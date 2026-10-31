//! 業務 API handler（SPEC §4–§6）。

pub mod decision_rerank;
pub mod embeddings;
pub mod rerank;

use std::sync::Arc;

use axum::Extension;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, header};
use axum::response::Response;
use bytes::Bytes;
use serde_json::Value;

use crate::backend::{ServicePool, split_sizes};
use crate::server::App;
use crate::upstream::{Buffered, CallLog, Flavor, Forward, GatewayError, forward_streaming};

/// 拆分後的一份：送往哪個後端、對應原始陣列的哪一段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    pub backend: usize,
    pub offset: usize,
    pub len: usize,
}

/// 把 `n` 筆依權重切給最多 `n` 個後端（SPEC §4.2）。
pub fn plan_chunks(svc: &ServicePool, n: usize) -> Result<Vec<Chunk>, GatewayError> {
    let chosen = svc.pick(n.max(1), &[]);
    if chosen.is_empty() {
        // 一般後端都不可用時，整個 request 交給 fallback，不拆分
        return match svc.fallback() {
            Some(backend) => Ok(vec![Chunk { backend, offset: 0, len: n.max(1) }]),
            None => Err(GatewayError::NoBackend),
        };
    }
    let weights: Vec<u32> = chosen.iter().map(|i| svc.backends[*i].weight).collect();
    let mut offset = 0;
    let mut chunks = Vec::with_capacity(chosen.len());
    for (backend, len) in chosen.into_iter().zip(split_sizes(n.max(1), &weights)) {
        if len > 0 {
            chunks.push(Chunk { backend, offset, len });
            offset += len;
        }
    }
    Ok(chunks)
}

/// 把 `b` 的數值加到 `a`（遞迴處理物件），用於合併 `usage`、`billed_units`。
pub fn sum_numbers(a: &mut Value, b: &Value) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, bv) in b {
                match a.get_mut(k) {
                    Some(av) => sum_numbers(av, bv),
                    None => {
                        a.insert(k.clone(), bv.clone());
                    }
                }
            }
        }
        (Value::Number(an), Value::Number(bn)) => {
            let sum = match (an.as_u64(), bn.as_u64()) {
                (Some(x), Some(y)) => Value::from(x + y),
                _ => Value::from(an.as_f64().unwrap_or(0.0) + bn.as_f64().unwrap_or(0.0)),
            };
            *an = match sum {
                Value::Number(n) => n,
                _ => return,
            };
        }
        _ => {}
    }
}

pub fn json_response(content_type: Option<HeaderValue>, body: Bytes) -> Response {
    let mut resp = Response::new(body.into());
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        content_type.unwrap_or_else(|| HeaderValue::from_static("application/json")),
    );
    resp
}

pub fn passthrough(buf: Buffered) -> Response {
    json_response(buf.content_type, buf.body)
}

pub fn parse_object(body: &[u8]) -> Result<serde_json::Map<String, Value>, GatewayError> {
    serde_json::from_slice(body).map_err(|e| GatewayError::BadRequest(format!("invalid JSON body: {e}")))
}

/// `POST /v1/systemone`：整包轉發到單一後端（SPEC §6）。
pub async fn decision(State(app): State<Arc<App>>, Extension(log): Extension<CallLog>, headers: HeaderMap, body: Bytes) -> Response {
    let pool = app.pool.load_full();
    let fwd = Forward::new(&headers, log);
    forward_streaming(&pool.decision, "/v1/systemone", body, &fwd)
        .await
        .unwrap_or_else(|e| e.into_response(Flavor::Plain))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sums_nested_numbers() {
        let mut a = json!({"prompt_tokens": 3, "total_tokens": 3, "x": "keep"});
        sum_numbers(&mut a, &json!({"prompt_tokens": 4, "total_tokens": 4, "extra": 1.5}));
        assert_eq!(a, json!({"prompt_tokens": 7, "total_tokens": 7, "x": "keep", "extra": 1.5}));
    }
}
