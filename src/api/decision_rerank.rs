//! `POST /decision/v2/rerank`：Cohere v2 格式的 rerank，以 decision 模型逐筆判斷相關度（SPEC §6.4）。

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, stream};
use serde_json::{Value, json};

use super::{json_response, parse_object, split_text, truncate_chars};
use crate::backend::{ServicePool, smooth_wrr_schedule};
use crate::config::DecisionRerankerConfig;
use crate::server::App;
use crate::upstream::{CallError, CallLog, Flavor, Forward, GatewayError, call_once, call_with_retry};

const PATH: &str = "/v1/systemone";
/// 是非題的題目 ID。
const QUESTION: &str = "relevant";

pub async fn handle(
    State(app): State<Arc<App>>,
    Extension(log): Extension<CallLog>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let pool = app.pool.load_full();
    let Some(cfg) = pool.config.decision_reranker.as_ref() else {
        return (StatusCode::NOT_FOUND, Json(json!({"message": "decision reranker is not configured"}))).into_response();
    };
    let fwd = Forward::new(&headers, log);
    rerank(&pool.decision, cfg, fwd, body).await.unwrap_or_else(|e| e.into_response(Flavor::Cohere))
}

struct RerankRequest {
    query: String,
    documents: Vec<String>,
    top_n: Option<usize>,
    instructions: Option<String>,
}

fn parse_request(body: &[u8], cfg: &DecisionRerankerConfig) -> Result<RerankRequest, GatewayError> {
    let req = parse_object(body)?;
    let bad = |m: &str| GatewayError::BadRequest(m.to_string());
    let query = req.get("query").and_then(Value::as_str).ok_or_else(|| bad("query must be a string"))?.to_string();
    let Some(Value::Array(docs)) = req.get("documents") else {
        return Err(bad("documents must be an array"));
    };
    if docs.len() > cfg.max_documents {
        return Err(GatewayError::BadRequest(format!("too many documents ({} > {})", docs.len(), cfg.max_documents)));
    }
    let documents = docs
        .iter()
        .map(|d| match d {
            Value::String(s) => Some(s.clone()),
            Value::Object(o) => o.get("text").and_then(Value::as_str).map(String::from),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| bad("documents must be strings or objects with a text field"))?;
    // 與 vLLM 一致：top_n 為 0 視同未指定（回傳全部）
    let top_n = req.get("top_n").and_then(Value::as_u64).filter(|n| *n > 0).map(|n| n as usize);
    let instructions = match req.get("instructions") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if !s.trim().is_empty() => Some(s.clone()),
        Some(_) => return Err(bad("instructions must be a non-empty string")),
    };
    Ok(RerankRequest { query, documents, top_n, instructions })
}

/// 套用 state 樣板；先以 `{document}` 切開，避免 query 內容裡的 `{document}` 被誤換。
fn render_state(template: &str, query: &str, document: &str) -> String {
    template.split("{document}").map(|part| part.replace("{query}", query)).collect::<Vec<_>>().join(document)
}

fn payload(cfg: &DecisionRerankerConfig, query: &str, document: &str, instructions: &str) -> Bytes {
    let mut body = json!({
        "state": render_state(&cfg.state_template, query, document),
        "questions": {QUESTION: {"type": "noul", "instructions": instructions}},
    });
    if let Some(model) = &cfg.model {
        body["model"] = Value::from(model.as_str());
    }
    Bytes::from(serde_json::to_vec(&body).expect("serialize decision request"))
}

struct Judgment {
    score: f64,
    input_tokens: Option<u64>,
    model: Option<String>,
}

fn parse_judgment(body: &[u8]) -> Result<Judgment, String> {
    let v: Value = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    let answer = &v["answers"][QUESTION];
    let score = match (answer["type"].as_str(), answer["noul"].as_f64()) {
        (Some("noul"), Some(p)) => p,
        _ => return Err(format!("missing noul answer for {QUESTION:?}")),
    };
    Ok(Judgment { score, input_tokens: v["usage"]["input_tokens"].as_u64(), model: v["model"].as_str().map(String::from) })
}

/// 每筆 document 指定一個一般後端：依權重平滑輪流交錯，讓並行上限內的呼叫平均分到各後端。
fn assign_backends(svc: &ServicePool, n: usize) -> Vec<usize> {
    let chosen = svc.pick(n, &[]);
    if chosen.is_empty() {
        return chosen;
    }
    let weights: Vec<u32> = chosen.iter().map(|i| svc.backends[*i].weight).collect();
    let schedule = smooth_wrr_schedule(&weights);
    (0..n).map(|i| chosen[schedule[i % schedule.len()]]).collect()
}

/// 送出所有判斷並依原順序回傳；任一筆失敗即整體失敗（其餘進行中的呼叫隨之取消）。
async fn judge_all(
    svc: &ServicePool,
    payloads: &[Bytes],
    backend_for: impl Fn(usize) -> usize,
    limit: usize,
    fwd: &Forward,
) -> Result<Vec<Judgment>, GatewayError> {
    // 先收成 Vec：直接把借用 payloads 的 iterator 交給 stream 會讓 handler 的 future 無法滿足 axum 的 Send 推導
    let calls: Vec<_> = payloads.iter().cloned().enumerate().map(|(i, body)| {
        let first = backend_for(i);
        async move {
            let judgment = if svc.backends[first].fallback {
                // fallback 不重試到一般後端：同一個回應裡的分數必須來自同一個模型
                let buf = call_once(svc, first, PATH, body, fwd).await?;
                parse_judgment(&buf.body).map_err(CallError::Invalid)?
            } else {
                // 判斷相關度沒有副作用，可以像 rerank 一樣在 5xx / 逾時時改送其他後端
                call_with_retry(svc, first, PATH, body, fwd, false, |buf| parse_judgment(&buf.body)).await?
            };
            Ok::<_, GatewayError>((i, judgment))
        }
    }).collect();
    let mut out: Vec<(usize, Judgment)> = stream::iter(calls).buffer_unordered(limit).try_collect().await?;
    out.sort_by_key(|(i, _)| *i);
    Ok(out.into_iter().map(|(_, j)| j).collect())
}

async fn rerank(svc: &ServicePool, cfg: &DecisionRerankerConfig, fwd: Forward, body: Bytes) -> Result<Response, GatewayError> {
    let req = parse_request(&body, cfg)?;
    let n = req.documents.len();
    fwd.log.set_items(n);
    let instructions = req.instructions.as_deref().unwrap_or(&cfg.instructions);
    // 截斷或切段（設定上擇一，SPEC §6.4）：每一段各是一個 decision 呼叫，owner[i] 是第 i 段所屬的 document
    let mut owner = Vec::new();
    let mut payloads = Vec::new();
    for (i, doc) in req.documents.iter().enumerate() {
        for piece in split_text(truncate_chars(doc, cfg.max_document_chars), cfg.chunking.as_ref()) {
            owner.push(i);
            payloads.push(payload(cfg, &req.query, piece, instructions));
        }
    }
    let m = payloads.len();
    if m > n {
        fwd.log.set_chunks(m);
    }
    let limit = cfg.max_concurrency.unwrap_or(m).max(1);
    let fallback = svc.fallback().filter(|_| cfg.use_fallback);

    let judgments = if m == 0 {
        Vec::new()
    } else {
        let assigned = assign_backends(svc, m);
        if assigned.is_empty() {
            // 一般後端都不可用：整個 request 交給 fallback
            let fb = fallback.ok_or(GatewayError::NoBackend)?;
            judge_all(svc, &payloads, |_| fb, limit, &fwd).await?
        } else {
            match judge_all(svc, &payloads, |i| assigned[i], limit, &fwd).await {
                Ok(j) => j,
                // fallback 是不同模型，分數不能混排：任一筆失敗時整個 request 改送 fallback
                Err(GatewayError::Upstream(e)) if e.is_backend_failure() && fallback.is_some() => {
                    let fb = fallback.expect("checked above");
                    tracing::warn!(service = "decision_reranker", retry_to = %svc.backends[fb].url, reason = %e.reason(),
                        "falling back with whole request");
                    judge_all(svc, &payloads, |_| fb, limit, &fwd).await?
                }
                Err(e) => return Err(e),
            }
        }
    };

    let model = judgments.iter().find_map(|j| j.model.clone());
    let prompt_tokens = judgments.iter().filter_map(|j| j.input_tokens).reduce(|a, b| a + b);
    // 每份 document 的分數取其各段的最高分
    let mut scores = vec![f64::NEG_INFINITY; n];
    for (j, i) in judgments.iter().zip(&owner) {
        scores[*i] = scores[*i].max(j.score);
    }
    let mut ranked: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if let Some(top_n) = req.top_n {
        ranked.truncate(top_n);
    }
    let results: Vec<Value> = ranked
        .into_iter()
        .map(|(i, score)| json!({"index": i, "relevance_score": score, "document": {"text": req.documents[i]}}))
        .collect();
    let mut out = json!({"id": uuid::Uuid::new_v4().to_string(), "results": results});
    if let Some(model) = model {
        out["model"] = Value::from(model);
    }
    if let Some(tokens) = prompt_tokens {
        out["usage"] = json!({"prompt_tokens": tokens});
    }
    Ok(json_response(None, Bytes::from(serde_json::to_vec(&out).expect("serialize response"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_state_without_reinterpreting_content() {
        let t = "Query: {query}\n\nDocument: {document}";
        assert_eq!(render_state(t, "q {document}", "d {query}"), "Query: q {document}\n\nDocument: d {query}");
    }

    #[test]
    fn parses_noul_answer() {
        let j = parse_judgment(br#"{"model":"m","answers":{"relevant":{"type":"noul","noul":0.8}},"usage":{"input_tokens":12}}"#)
            .unwrap();
        assert_eq!((j.score, j.input_tokens, j.model.as_deref()), (0.8, Some(12), Some("m")));
        assert!(parse_judgment(br#"{"answers":{"relevant":{"type":"refusal"}}}"#).is_err());
        assert!(parse_judgment(b"not json").is_err());
    }
}
