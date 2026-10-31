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
use crate::config::ChunkingConfig;
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

/// 依字元數切段（SPEC §5.4）：每段 `max_chars` 字元、相鄰兩段重疊 `overlap_chars` 字元；
/// 最後一段對齊結尾、取滿 `max_chars`，避免只剩幾個字的尾段，因此它與前一段的重疊可能較多。
/// 不超過 `max_chars` 的文字（含空字串）原樣成為唯一一段。
pub fn split_text<'a>(text: &'a str, chunking: Option<&ChunkingConfig>) -> Vec<&'a str> {
    let Some(c) = chunking else { return vec![text] };
    // 各字元的 byte 位置，最後補上結尾，方便以字元索引切出子字串
    let bounds: Vec<usize> = text.char_indices().map(|(i, _)| i).chain([text.len()]).collect();
    let n = bounds.len() - 1;
    if n <= c.max_chars {
        return vec![text];
    }
    let step = c.max_chars - c.overlap_chars;
    let mut starts: Vec<usize> = (0..).map(|k| k * step).take_while(|s| s + c.max_chars < n).collect();
    starts.push(n - c.max_chars);
    starts.into_iter().map(|s| &text[bounds[s]..bounds[s + c.max_chars]]).collect()
}

/// 截斷到前 `max_chars` 個字元；`None` = 不截斷。
pub fn truncate_chars(text: &str, max_chars: Option<usize>) -> &str {
    match max_chars.and_then(|n| text.char_indices().nth(n)) {
        Some((i, _)) => &text[..i],
        None => text,
    }
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
    fn splits_by_chars_with_overlap_and_end_aligned_tail() {
        let c = ChunkingConfig { max_chars: 4, overlap_chars: 1 };
        assert_eq!(split_text("abcdefghij", None), ["abcdefghij"]);
        assert_eq!(split_text("", Some(&c)), [""]);
        assert_eq!(split_text("abcd", Some(&c)), ["abcd"]);
        // 起點 0、3，最後一段對齊結尾（起點 6）
        assert_eq!(split_text("abcdefghij", Some(&c)), ["abcd", "defg", "ghij"]);
        // 尾段不足 max_chars 時往前取滿：最後一段與前一段重疊 3 個字元
        assert_eq!(split_text("abcdefgh", Some(&c)), ["abcd", "defg", "efgh"]);
        assert_eq!(split_text("abcde", Some(&c)), ["abcd", "bcde"]);
        // 以字元計算，不是 byte
        assert_eq!(split_text("一二三四五六七", Some(&c)), ["一二三四", "四五六七"]);
        let no_overlap = ChunkingConfig { max_chars: 3, overlap_chars: 0 };
        assert_eq!(split_text("abcdefghi", Some(&no_overlap)), ["abc", "def", "ghi"]);
    }

    #[test]
    fn truncates_by_chars() {
        assert_eq!(truncate_chars("一二三", Some(2)), "一二");
        assert_eq!(truncate_chars("一二三", Some(5)), "一二三");
        assert_eq!(truncate_chars("abc", None), "abc");
    }

    #[test]
    fn sums_nested_numbers() {
        let mut a = json!({"prompt_tokens": 3, "total_tokens": 3, "x": "keep"});
        sum_numbers(&mut a, &json!({"prompt_tokens": 4, "total_tokens": 4, "extra": 1.5}));
        assert_eq!(a, json!({"prompt_tokens": 7, "total_tokens": 7, "x": "keep", "extra": 1.5}));
    }
}
