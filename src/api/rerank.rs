//! `POST /v1/rerank`、`POST /v2/rerank`（Cohere 格式，SPEC §5）。

use std::sync::Arc;

use axum::extract::{OriginalUri, State};
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{Chunk, json_response, parse_object, passthrough, plan_chunks, sum_numbers};
use crate::backend::ServicePool;
use crate::server::App;
use crate::upstream::{Flavor, Forward, GatewayError, call_with_retry};

#[derive(Deserialize)]
pub struct RerankResponse {
    results: Vec<RerankItem>,
    #[serde(flatten)]
    rest: Map<String, Value>,
}

#[derive(Deserialize, Serialize)]
pub struct RerankItem {
    index: usize,
    relevance_score: f64,
    #[serde(flatten)]
    rest: Map<String, Value>,
}

pub fn parse_part(body: &[u8], len: usize) -> Result<RerankResponse, String> {
    let resp: RerankResponse = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    if resp.results.iter().any(|r| r.index >= len) {
        return Err(format!("result index out of range (chunk has {len} documents)"));
    }
    Ok(resp)
}

/// 改寫 index、依分數由高到低排序（同分時 index 小者在前）、套用 top_n。
pub fn merge(parts: Vec<(Chunk, RerankResponse)>, top_n: Option<usize>) -> Map<String, Value> {
    let mut results = Vec::new();
    let mut rest: Option<Map<String, Value>> = None;
    for (chunk, part) in parts {
        match rest.as_mut() {
            None => rest = Some(part.rest),
            Some(total) => {
                for key in ["usage", "meta"] {
                    if let (Some(a), Some(b)) = (total.get_mut(key), part.rest.get(key)) {
                        if key == "meta" {
                            // meta 只加總 billed_units，api_version 等取第一份
                            if let (Some(a), Some(b)) = (a.get_mut("billed_units"), b.get("billed_units")) {
                                sum_numbers(a, b);
                            }
                        } else {
                            sum_numbers(a, b);
                        }
                    }
                }
            }
        }
        results.extend(part.results.into_iter().map(|mut r| {
            r.index += chunk.offset;
            r
        }));
    }
    results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score).then(a.index.cmp(&b.index)));
    if let Some(n) = top_n {
        results.truncate(n);
    }
    let mut out = rest.unwrap_or_default();
    if out.contains_key("id") {
        out.insert("id".into(), Value::from(uuid::Uuid::new_v4().to_string()));
    }
    out.insert("results".into(), serde_json::to_value(results).expect("serialize results"));
    out
}

pub async fn handle(State(app): State<Arc<App>>, OriginalUri(uri): OriginalUri, headers: HeaderMap, body: Bytes) -> Response {
    let pool = app.pool.load_full();
    rerank(&pool.reranker, uri.path(), &headers, body).await.unwrap_or_else(|e| e.into_response(Flavor::Cohere))
}

async fn rerank(svc: &ServicePool, path: &str, headers: &HeaderMap, body: Bytes) -> Result<Response, GatewayError> {
    let fwd = Forward::from_headers(headers);
    let mut req = parse_object(&body)?;
    let n = match req.get("documents") {
        Some(Value::Array(docs)) if docs.len() > 1 => docs.len(),
        _ => 1,
    };
    let chunks = plan_chunks(svc, n)?;
    if chunks.len() == 1 {
        let buf = call_with_retry(svc, chunks[0].backend, path, body, &fwd, Ok).await?;
        return Ok(passthrough(buf));
    }

    // 與 vLLM 一致：top_n 為 0 視同未指定（回傳全部）
    let top_n = req.get("top_n").and_then(Value::as_u64).filter(|n| *n > 0).map(|n| n as usize);
    let Some(Value::Array(docs)) = req.remove("documents") else { unreachable!("checked above") };
    let calls = chunks.iter().map(|chunk| {
        let mut sub = req.clone();
        sub.insert("documents".into(), Value::Array(docs[chunk.offset..chunk.offset + chunk.len].to_vec()));
        // 全域前 n 名必然落在各份的前 n 名之中
        if let Some(n) = top_n {
            sub.insert("top_n".into(), Value::from(n.min(chunk.len)));
        }
        let sub_body = Bytes::from(serde_json::to_vec(&sub).expect("serialize sub-request"));
        let fwd = &fwd;
        let chunk = *chunk;
        async move {
            let part = call_with_retry(svc, chunk.backend, path, sub_body, fwd, |buf| parse_part(&buf.body, chunk.len)).await?;
            Ok::<_, GatewayError>((chunk, part))
        }
    });
    let parts = futures::future::try_join_all(calls).await?;
    let out = serde_json::to_vec(&merge(parts, top_n)).expect("serialize merged response");
    Ok(json_response(None, out.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn merges_by_score_with_top_n() {
        let a = br#"{"id":"a","results":[{"index":1,"relevance_score":0.9,"document":{"text":"d1"}},{"index":0,"relevance_score":0.1}],
            "meta":{"api_version":{"version":"2"},"billed_units":{"search_units":1}}}"#;
        let b = br#"{"id":"b","results":[{"index":0,"relevance_score":0.95},{"index":1,"relevance_score":0.9}],
            "meta":{"api_version":{"version":"2"},"billed_units":{"search_units":1}}}"#;
        let parts = vec![
            (Chunk { backend: 0, offset: 0, len: 2 }, parse_part(a, 2).unwrap()),
            (Chunk { backend: 1, offset: 2, len: 2 }, parse_part(b, 2).unwrap()),
        ];
        let out = Value::Object(merge(parts, Some(3)));
        assert_eq!(
            out["results"],
            json!([
                {"index": 2, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.9, "document": {"text": "d1"}},
                {"index": 3, "relevance_score": 0.9}
            ])
        );
        assert_eq!(out["meta"], json!({"api_version": {"version": "2"}, "billed_units": {"search_units": 2}}));
        assert_ne!(out["id"], json!("a"));
    }

    #[test]
    fn sums_usage_without_id() {
        let a = br#"{"model":"m","usage":{"prompt_tokens":3},"results":[{"index":0,"relevance_score":1.0}]}"#;
        let b = br#"{"model":"m","usage":{"prompt_tokens":4},"results":[{"index":0,"relevance_score":2.0}]}"#;
        let parts = vec![
            (Chunk { backend: 0, offset: 0, len: 1 }, parse_part(a, 1).unwrap()),
            (Chunk { backend: 1, offset: 1, len: 1 }, parse_part(b, 1).unwrap()),
        ];
        let out = Value::Object(merge(parts, None));
        assert_eq!(out["usage"], json!({"prompt_tokens": 7}));
        assert!(out.get("id").is_none());
        assert_eq!(out["results"][0]["index"], json!(1));
    }

    #[test]
    fn rejects_out_of_range_index() {
        assert!(parse_part(br#"{"results":[{"index":5,"relevance_score":1.0}]}"#, 2).is_err());
    }
}
