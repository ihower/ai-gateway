//! `POST /v2/rerank`（Cohere v2 格式，SPEC §5）。

use std::sync::Arc;

use axum::Extension;
use axum::extract::{OriginalUri, State};
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{Chunk, json_response, parse_object, passthrough, plan_chunks, split_text, sum_numbers};
use crate::backend::ServicePool;
use crate::config::ChunkingConfig;
use crate::server::App;
use crate::upstream::{CallLog, Flavor, Forward, GatewayError, call_once, call_with_retry};

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

/// 長文件切段後實際送出的 documents（SPEC §5.4）。
pub struct Expanded {
    docs: Vec<Value>,
    /// 第 i 段屬於哪一份原始 document。
    owner: Vec<usize>,
    /// 被切開的原始 document 全文（依原始 index），用來還原回應中的 `document.text`。
    originals: Vec<Option<String>>,
}

/// 把超過 `max_chars` 的 document 切成多段；沒有任何 document 需要切時回傳 `None`。
/// 字串與含 `text` 的物件都可切（物件的其他欄位保留在每一段）；其他形態原樣送出。
pub fn expand(docs: &[Value], chunking: &ChunkingConfig) -> Option<Expanded> {
    let mut ex = Expanded { docs: Vec::new(), owner: Vec::new(), originals: vec![None; docs.len()] };
    for (i, doc) in docs.iter().enumerate() {
        let text = match doc {
            Value::String(s) => Some(s.as_str()),
            Value::Object(o) => o.get("text").and_then(Value::as_str),
            _ => None,
        };
        let pieces = text.map(|t| split_text(t, Some(chunking))).filter(|p| p.len() > 1);
        let Some(pieces) = pieces else {
            ex.docs.push(doc.clone());
            ex.owner.push(i);
            continue;
        };
        for piece in pieces {
            ex.docs.push(match doc {
                Value::Object(o) => {
                    let mut o = o.clone();
                    o.insert("text".into(), Value::from(piece));
                    Value::Object(o)
                }
                _ => Value::from(piece),
            });
            ex.owner.push(i);
        }
        ex.originals[i] = text.map(String::from);
    }
    (ex.docs.len() > docs.len()).then_some(ex)
}

/// 各段的結果收回原始 document：分數取各段最高分，`document.text` 換回原文。
fn collapse(results: Vec<RerankItem>, ex: &Expanded) -> Vec<RerankItem> {
    let mut best: Vec<Option<RerankItem>> = (0..ex.originals.len()).map(|_| None).collect();
    for mut r in results {
        r.index = ex.owner[r.index];
        let slot = &mut best[r.index];
        if slot.as_ref().is_none_or(|b| r.relevance_score > b.relevance_score) {
            *slot = Some(r);
        }
    }
    best.into_iter()
        .flatten()
        .map(|mut r| {
            if let (Some(text), Some(doc)) = (&ex.originals[r.index], r.rest.get_mut("document")) {
                *doc = serde_json::json!({"text": text});
            }
            r
        })
        .collect()
}

/// 改寫 index（有切段時收回原始 document）、依分數由高到低排序（同分時 index 小者在前）、套用 top_n。
pub fn merge(parts: Vec<(Chunk, RerankResponse)>, top_n: Option<usize>, expanded: Option<&Expanded>) -> Map<String, Value> {
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
    if let Some(ex) = expanded {
        results = collapse(results, ex);
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

pub async fn handle(
    State(app): State<Arc<App>>,
    Extension(log): Extension<CallLog>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let pool = app.pool.load_full();
    let fwd = Forward::new(&headers, log);
    rerank(&pool.reranker, pool.config.reranker.chunking.as_ref(), uri.path(), fwd, body).await.unwrap_or_else(|e| e.into_response(Flavor::Cohere))
}

async fn rerank(
    svc: &ServicePool,
    chunking: Option<&ChunkingConfig>,
    path: &str,
    fwd: Forward,
    body: Bytes,
) -> Result<Response, GatewayError> {
    let mut req = parse_object(&body)?;
    let n = match req.get("documents") {
        Some(Value::Array(docs)) if docs.len() > 1 => docs.len(),
        _ => 1,
    };
    fwd.log.set_items(n);
    let mut expanded = match (chunking, req.get("documents")) {
        (Some(c), Some(Value::Array(docs))) => expand(docs, c),
        _ => None,
    };
    if let Some(ex) = &expanded {
        fwd.log.set_chunks(ex.docs.len());
    }
    let chunks = plan_chunks(svc, expanded.as_ref().map_or(n, |ex| ex.docs.len()))?;
    // 沒有切段且只需一份時透明轉發；一般後端都不可用時是整包送 fallback，fallback 收原始 request、不切段
    if chunks.len() == 1 && (expanded.is_none() || svc.backends[chunks[0].backend].fallback) {
        let buf = call_with_retry(svc, chunks[0].backend, path, body, &fwd, true, Ok).await?;
        return Ok(passthrough(buf));
    }

    // 與 vLLM 一致：top_n 為 0 視同未指定（回傳全部）
    let top_n = req.get("top_n").and_then(Value::as_u64).filter(|n| *n > 0).map(|n| n as usize);
    let Some(Value::Array(original_docs)) = req.remove("documents") else { unreachable!("checked above") };
    let docs = match expanded.as_mut() {
        Some(ex) => std::mem::take(&mut ex.docs),
        None => original_docs,
    };
    let calls = chunks.iter().map(|chunk| {
        let mut sub = req.clone();
        sub.insert("documents".into(), Value::Array(docs[chunk.offset..chunk.offset + chunk.len].to_vec()));
        // 全域前 n 名必然落在各份的前 n 名之中；有切段時同一份 document 的多段可能同時擠進前 n 名，
        // 縮減會讓其他 document 整個消失，所以要拿掉 client 的 top_n、讓後端回傳所有段
        if expanded.is_some() {
            sub.remove("top_n");
        } else if let Some(n) = top_n {
            sub.insert("top_n".into(), Value::from(n.min(chunk.len)));
        }
        let sub_body = Bytes::from(serde_json::to_vec(&sub).expect("serialize sub-request"));
        let fwd = &fwd;
        let chunk = *chunk;
        async move {
            // fallback 可能是不同模型，分數尺度不同，不能與一般後端的分數混排；失敗時整包改送 fallback
            let part =
                call_with_retry(svc, chunk.backend, path, sub_body, fwd, false, |buf| parse_part(&buf.body, chunk.len)).await?;
            Ok::<_, GatewayError>((chunk, part))
        }
    });
    let parts = match futures::future::try_join_all(calls).await {
        Ok(parts) => parts,
        Err(GatewayError::Upstream(e)) if e.is_backend_failure() && svc.fallback().is_some() => {
            let fb = svc.fallback().expect("checked above");
            tracing::warn!(service = %svc.kind, retry_to = %svc.backends[fb].url, reason = %e.reason(), "falling back with whole request");
            return match call_once(svc, fb, path, body, &fwd).await {
                Ok(buf) => Ok(passthrough(buf)),
                Err(e) => Err(e.into()),
            };
        }
        Err(e) => return Err(e),
    };
    let out = serde_json::to_vec(&merge(parts, top_n, expanded.as_ref())).expect("serialize merged response");
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
        let out = Value::Object(merge(parts, Some(3), None));
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
        let out = Value::Object(merge(parts, None, None));
        assert_eq!(out["usage"], json!({"prompt_tokens": 7}));
        assert!(out.get("id").is_none());
        assert_eq!(out["results"][0]["index"], json!(1));
    }

    #[test]
    fn expands_long_documents_and_keeps_max_score() {
        let c = ChunkingConfig { max_chars: 4, overlap_chars: 1 };
        let docs = vec![json!("abc"), json!("abcdefg"), json!({"text": "uvwxyz", "id": 7}), json!(1)];
        let ex = expand(&docs, &c).unwrap();
        assert_eq!(
            Value::Array(ex.docs.clone()),
            json!(["abc", "abcd", "defg", {"text": "uvwx", "id": 7}, {"text": "wxyz", "id": 7}, 1])
        );
        assert_eq!(ex.owner, vec![0, 1, 1, 2, 2, 3]);
        assert!(expand(&[json!("abcd"), json!(2)], &c).is_none());

        // 兩個後端各拿到 3 段
        let a = br#"{"results":[{"index":0,"relevance_score":0.5,"document":{"text":"abc"}},
            {"index":1,"relevance_score":0.2,"document":{"text":"abcd"}},{"index":2,"relevance_score":0.9,"document":{"text":"defg"}}]}"#;
        let b = br#"{"results":[{"index":0,"relevance_score":0.6},{"index":1,"relevance_score":0.1},{"index":2,"relevance_score":0.7}]}"#;
        let parts = vec![
            (Chunk { backend: 0, offset: 0, len: 3 }, parse_part(a, 3).unwrap()),
            (Chunk { backend: 1, offset: 3, len: 3 }, parse_part(b, 3).unwrap()),
        ];
        let out = Value::Object(merge(parts, Some(3), Some(&ex)));
        assert_eq!(
            out["results"],
            json!([
                {"index": 1, "relevance_score": 0.9, "document": {"text": "abcdefg"}},
                {"index": 3, "relevance_score": 0.7},
                {"index": 2, "relevance_score": 0.6}
            ])
        );
    }

    #[test]
    fn rejects_out_of_range_index() {
        assert!(parse_part(br#"{"results":[{"index":5,"relevance_score":1.0}]}"#, 2).is_err());
    }
}
