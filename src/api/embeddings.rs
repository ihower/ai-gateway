//! `POST /v1/embeddings`（OpenAI 格式，SPEC §4）。

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Extension;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use serde::Serialize;
use serde_json::Value;
use serde_json::value::RawValue;

use super::{Chunk, json_response, parse_object, passthrough, plan_chunks, sum_numbers};
use crate::backend::ServicePool;
use crate::server::App;
use crate::upstream::{CallLog, Flavor, Forward, GatewayError, call_with_retry};

const PATH: &str = "/v1/embeddings";

type RawMap = BTreeMap<String, Box<RawValue>>;

/// 後端回應拆成 `data` 與其餘欄位；所有值都以 `RawValue` 原樣保留，向量不重新解析浮點數，
/// 後端多回的欄位（例如 vLLM 的 `id`、`created`）也會保留。
pub struct EmbeddingResponse {
    rest: RawMap,
    data: Vec<(usize, RawMap)>,
}

#[derive(Serialize)]
pub struct MergedEmbeddings {
    #[serde(flatten)]
    rest: RawMap,
    data: Vec<RawMap>,
}

/// 可拆分的 input：字串陣列或 token 陣列的陣列。單一字串或單一 token 陣列（數字陣列）不可拆。
pub fn splittable_inputs(input: Option<&Value>) -> Option<&Vec<Value>> {
    match input {
        Some(Value::Array(items)) if items.len() > 1 && !items[0].is_number() => Some(items),
        _ => None,
    }
}

pub fn parse_part(body: &[u8], expected: usize) -> Result<EmbeddingResponse, String> {
    let mut rest: RawMap = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    let data = rest.remove("data").ok_or("missing \"data\"")?;
    let items: Vec<RawMap> = serde_json::from_str(data.get()).map_err(|e| format!("data: {e}"))?;
    let mut data = Vec::with_capacity(items.len());
    for item in items {
        let index = item.get("index").ok_or("missing \"index\"")?;
        let index: usize = serde_json::from_str(index.get()).map_err(|e| format!("index: {e}"))?;
        data.push((index, item));
    }
    if data.len() != expected || data.iter().any(|(i, _)| *i >= expected) {
        return Err(format!("expected {expected} embeddings, got {}", data.len()));
    }
    Ok(EmbeddingResponse { rest, data })
}

fn raw(v: &impl Serialize) -> Box<RawValue> {
    serde_json::value::to_raw_value(v).expect("serialize JSON value")
}

/// 依原始順序串接，`index` 改寫為原始陣列中的位置，`usage` 加總；其餘欄位取第一份，
/// `id` 改為新的 UUID（各份的 id 已無意義）。
pub fn merge(parts: Vec<(Chunk, EmbeddingResponse)>) -> MergedEmbeddings {
    let mut data = Vec::with_capacity(parts.iter().map(|(c, _)| c.len).sum());
    let mut rest: Option<RawMap> = None;
    let mut usage: Option<Value> = None;
    for (chunk, mut part) in parts {
        if let Some(u) = part.rest.get("usage").and_then(|u| serde_json::from_str::<Value>(u.get()).ok()) {
            match usage.as_mut() {
                Some(total) => sum_numbers(total, &u),
                None => usage = Some(u),
            }
        }
        if rest.is_none() {
            rest = Some(std::mem::take(&mut part.rest));
        }
        part.data.sort_by_key(|(i, _)| *i);
        for (i, mut item) in part.data {
            item.insert("index".into(), raw(&(chunk.offset + i)));
            data.push(item);
        }
    }
    let mut rest = rest.unwrap_or_default();
    if let Some(u) = usage {
        rest.insert("usage".into(), raw(&u));
    }
    if rest.contains_key("id") {
        rest.insert("id".into(), raw(&uuid::Uuid::new_v4().to_string()));
    }
    rest.entry("object".into()).or_insert_with(|| raw(&"list"));
    MergedEmbeddings { rest, data }
}

pub async fn handle(State(app): State<Arc<App>>, Extension(log): Extension<CallLog>, headers: HeaderMap, body: Bytes) -> Response {
    let pool = app.pool.load_full();
    let fwd = Forward::new(&headers, log);
    embeddings(&pool.embedding, fwd, body).await.unwrap_or_else(|e| e.into_response(Flavor::OpenAi))
}

async fn embeddings(svc: &ServicePool, fwd: Forward, body: Bytes) -> Result<Response, GatewayError> {
    let mut req = parse_object(&body)?;
    let n = splittable_inputs(req.get("input")).map_or(1, Vec::len);
    fwd.log.set_items(n);
    let chunks = plan_chunks(svc, n)?;
    if chunks.len() == 1 {
        let buf = call_with_retry(svc, chunks[0].backend, PATH, body, &fwd, true, Ok).await?;
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
            let part = call_with_retry(svc, chunk.backend, PATH, sub_body, fwd, true, |buf| parse_part(&buf.body, chunk.len)).await?;
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
        let a = br#"{"id":"embd-a","created":1,"object":"list","model":"m","data":[
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
        let mut out: Value = serde_json::to_value(merge(parts)).unwrap();
        let id = out.as_object_mut().unwrap().remove("id").unwrap();
        assert_ne!(id, json!("embd-a"));
        assert_eq!(
            out,
            json!({"object":"list","model":"m","created":1,
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
