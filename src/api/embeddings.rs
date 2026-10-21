//! `POST /v1/embeddings`（OpenAI 格式，SPEC §4）。

use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::value::RawValue;

use super::{Chunk, json_response, parse_object, passthrough, plan_chunks, sum_numbers};
use crate::backend::ServicePool;
use crate::server::App;
use crate::upstream::{Flavor, Forward, GatewayError, call_with_retry};

const PATH: &str = "/v1/embeddings";

#[derive(Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default)]
    object: Option<Value>,
    data: Vec<EmbeddingItem>,
    #[serde(default)]
    model: Option<Value>,
    #[serde(default)]
    usage: Option<Value>,
}

/// 向量以 `RawValue` 原樣搬移，不重新解析浮點數。
#[derive(Deserialize, Serialize)]
pub struct EmbeddingItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    object: Option<Value>,
    index: usize,
    embedding: Box<RawValue>,
}

#[derive(Serialize)]
pub struct MergedEmbeddings {
    object: Value,
    data: Vec<EmbeddingItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Value>,
}

/// 可拆分的 input：字串陣列或 token 陣列的陣列。單一字串或單一 token 陣列（數字陣列）不可拆。
pub fn splittable_inputs(input: Option<&Value>) -> Option<&Vec<Value>> {
    match input {
        Some(Value::Array(items)) if items.len() > 1 && !items[0].is_number() => Some(items),
        _ => None,
    }
}

pub fn parse_part(body: &[u8], expected: usize) -> Result<EmbeddingResponse, String> {
    let resp: EmbeddingResponse = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    if resp.data.len() != expected || resp.data.iter().any(|d| d.index >= expected) {
        return Err(format!("expected {expected} embeddings, got {}", resp.data.len()));
    }
    Ok(resp)
}

/// 依原始順序串接，`index` 改寫為原始陣列中的位置，`usage` 加總。
pub fn merge(parts: Vec<(Chunk, EmbeddingResponse)>) -> MergedEmbeddings {
    let mut data = Vec::with_capacity(parts.iter().map(|(c, _)| c.len).sum());
    let mut merged = MergedEmbeddings { object: Value::from("list"), data: Vec::new(), model: None, usage: None };
    for (i, (chunk, mut part)) in parts.into_iter().enumerate() {
        if i == 0 {
            merged.object = part.object.take().unwrap_or_else(|| Value::from("list"));
            merged.model = part.model.take();
            merged.usage = part.usage.take();
        } else if let (Some(total), Some(u)) = (merged.usage.as_mut(), part.usage.as_ref()) {
            sum_numbers(total, u);
        }
        part.data.sort_by_key(|d| d.index);
        for mut item in part.data {
            item.index += chunk.offset;
            data.push(item);
        }
    }
    merged.data = data;
    merged
}

pub async fn handle(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    let pool = app.pool.load_full();
    embeddings(&pool.embedding, &headers, body).await.unwrap_or_else(|e| e.into_response(Flavor::OpenAi))
}

async fn embeddings(svc: &ServicePool, headers: &HeaderMap, body: Bytes) -> Result<Response, GatewayError> {
    let fwd = Forward::from_headers(headers);
    let mut req = parse_object(&body)?;
    let n = splittable_inputs(req.get("input")).map_or(1, Vec::len);
    let chunks = plan_chunks(svc, n)?;
    if chunks.len() == 1 {
        let buf = call_with_retry(svc, chunks[0].backend, PATH, body, &fwd, Ok).await?;
        return Ok(passthrough(buf));
    }

    let Some(Value::Array(items)) = req.remove("input") else { unreachable!("checked by splittable_inputs") };
    let calls = chunks.iter().map(|chunk| {
        let mut sub = req.clone();
        sub.insert("input".into(), Value::Array(items[chunk.offset..chunk.offset + chunk.len].to_vec()));
        let sub_body = Bytes::from(serde_json::to_vec(&sub).expect("serialize sub-request"));
        let fwd = &fwd;
        let chunk = *chunk;
        async move {
            let part = call_with_retry(svc, chunk.backend, PATH, sub_body, fwd, |buf| parse_part(&buf.body, chunk.len)).await?;
            Ok::<_, GatewayError>((chunk, part))
        }
    });
    let parts = futures::future::try_join_all(calls).await?;
    let out = serde_json::to_vec(&merge(parts)).expect("serialize merged response");
    Ok(json_response(None, out.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_splittable_inputs() {
        assert!(splittable_inputs(Some(&json!("a"))).is_none());
        assert!(splittable_inputs(Some(&json!(["a"]))).is_none());
        assert!(splittable_inputs(Some(&json!([1, 2, 3]))).is_none());
        assert_eq!(splittable_inputs(Some(&json!(["a", "b"]))).unwrap().len(), 2);
        assert_eq!(splittable_inputs(Some(&json!([[1], [2], [3]]))).unwrap().len(), 3);
        assert!(splittable_inputs(None).is_none());
    }

    #[test]
    fn merges_in_order_with_rewritten_indexes() {
        let a = br#"{"object":"list","model":"m","data":[
            {"object":"embedding","index":1,"embedding":[0.2]},
            {"object":"embedding","index":0,"embedding":[0.1]}],
            "usage":{"prompt_tokens":2,"total_tokens":2}}"#;
        let b = br#"{"object":"list","model":"m","data":[
            {"object":"embedding","index":0,"embedding":"AAAA"}],
            "usage":{"prompt_tokens":5,"total_tokens":5}}"#;
        let parts = vec![
            (Chunk { backend: 0, offset: 0, len: 2 }, parse_part(a, 2).unwrap()),
            (Chunk { backend: 1, offset: 2, len: 1 }, parse_part(b, 1).unwrap()),
        ];
        let out: Value = serde_json::to_value(merge(parts)).unwrap();
        assert_eq!(
            out,
            json!({"object":"list","model":"m",
                "data":[
                    {"object":"embedding","index":0,"embedding":[0.1]},
                    {"object":"embedding","index":1,"embedding":[0.2]},
                    {"object":"embedding","index":2,"embedding":"AAAA"}],
                "usage":{"prompt_tokens":7,"total_tokens":7}})
        );
    }

    #[test]
    fn rejects_wrong_count() {
        let a = br#"{"data":[{"index":0,"embedding":[0.1]}]}"#;
        assert!(parse_part(a, 2).is_err());
    }
}
