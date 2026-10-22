//! 呼叫後端、重試與錯誤轉換（SPEC §8.3–§8.5）。

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;

use crate::backend::{Backend, ServicePool};

pub const REQUEST_ID: &str = "x-request-id";

/// 從 client request 帶到後端的 header，以及記錄這個 request 呼叫過哪些後端。
#[derive(Clone, Default)]
pub struct Forward {
    pub authorization: Option<HeaderValue>,
    pub request_id: Option<HeaderValue>,
    pub log: CallLog,
}

impl Forward {
    pub fn new(headers: &HeaderMap, log: CallLog) -> Self {
        Forward {
            authorization: headers.get(header::AUTHORIZATION).cloned(),
            request_id: headers.get(REQUEST_ID).cloned(),
            log,
        }
    }
}

/// 一個 client request 對後端的每次呼叫與結果，由 access log 輸出。
#[derive(Clone, Default)]
pub struct CallLog(Arc<Mutex<CallLogInner>>);

#[derive(Default)]
struct CallLogInner {
    items: Option<usize>,
    attempts: Vec<(String, String)>,
    fallback: bool,
}

impl CallLog {
    pub fn set_items(&self, n: usize) {
        self.0.lock().unwrap().items = Some(n);
    }

    fn record(&self, backend: &Backend, outcome: String) {
        let mut inner = self.0.lock().unwrap();
        inner.fallback |= backend.fallback;
        inner.attempts.push((backend.url.clone(), outcome));
    }

    /// 改寫該後端最近一次呼叫的結果（2xx 但回應無法解析）。
    fn amend(&self, backend: &Backend, outcome: String) {
        let mut inner = self.0.lock().unwrap();
        if let Some(a) = inner.attempts.iter_mut().rev().find(|(url, _)| *url == backend.url) {
            a.1 = outcome;
        }
    }

    /// (筆數, 「後端 結果」以逗號串接, 是否用到 fallback)
    pub fn summary(&self) -> (Option<usize>, String, bool) {
        let inner = self.0.lock().unwrap();
        let attempts = inner.attempts.iter().map(|(url, outcome)| format!("{url} {outcome}")).collect::<Vec<_>>().join(", ");
        (inner.items, attempts, inner.fallback)
    }
}

#[derive(Debug)]
pub enum CallError {
    Connect(String),
    Timeout,
    /// 後端回非 2xx。
    Status { status: StatusCode, content_type: Option<HeaderValue>, body: Bytes },
    /// 後端回 2xx 但內容無法解析。
    Invalid(String),
    Other(String),
}

impl CallError {
    /// 連線錯誤、逾時、5xx 與無效回應算後端失敗，可改送其他後端；4xx 是 client 的問題。
    pub fn is_backend_failure(&self) -> bool {
        match self {
            CallError::Status { status, .. } => status.is_server_error(),
            _ => true,
        }
    }

    pub fn reason(&self) -> String {
        match self {
            CallError::Connect(e) => format!("connect failed: {e}"),
            CallError::Timeout => "timeout".to_string(),
            CallError::Status { status, .. } => format!("HTTP {}", status.as_u16()),
            CallError::Invalid(e) => format!("invalid response: {e}"),
            CallError::Other(e) => e.clone(),
        }
    }
}

pub fn classify(e: &reqwest::Error) -> CallError {
    if e.is_timeout() {
        CallError::Timeout
    } else if e.is_connect() {
        CallError::Connect(root_cause(e))
    } else {
        CallError::Other(root_cause(e))
    }
}

fn root_cause(e: &(dyn std::error::Error + 'static)) -> String {
    let mut cur = e;
    while let Some(next) = cur.source() {
        cur = next;
    }
    cur.to_string()
}

#[derive(Debug)]
pub enum GatewayError {
    BadRequest(String),
    NoBackend,
    Upstream(CallError),
}

impl From<CallError> for GatewayError {
    fn from(e: CallError) -> Self {
        GatewayError::Upstream(e)
    }
}

/// 錯誤回應的格式依 API 而定。
#[derive(Clone, Copy)]
pub enum Flavor {
    OpenAi,
    Cohere,
    Plain,
}

impl GatewayError {
    pub fn into_response(self, flavor: Flavor) -> Response {
        let (status, message) = match self {
            GatewayError::Upstream(CallError::Status { status, content_type, body }) if status.is_client_error() => {
                // 後端的 4xx 原樣回傳
                let mut resp = Response::new(Body::from(body));
                *resp.status_mut() = status;
                if let Some(ct) = content_type {
                    resp.headers_mut().insert(header::CONTENT_TYPE, ct);
                }
                return resp;
            }
            GatewayError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            GatewayError::NoBackend => (StatusCode::SERVICE_UNAVAILABLE, "no available backend".to_string()),
            GatewayError::Upstream(CallError::Timeout) => (StatusCode::GATEWAY_TIMEOUT, "backend timed out".to_string()),
            GatewayError::Upstream(e) => (StatusCode::BAD_GATEWAY, format!("backend error: {}", e.reason())),
        };
        let body = match flavor {
            Flavor::OpenAi => {
                let (kind, code) = match status {
                    StatusCode::BAD_REQUEST => ("invalid_request_error", "bad_request"),
                    StatusCode::SERVICE_UNAVAILABLE => ("upstream_error", "service_unavailable"),
                    StatusCode::GATEWAY_TIMEOUT => ("upstream_error", "gateway_timeout"),
                    _ => ("upstream_error", "bad_gateway"),
                };
                json!({"error": {"message": message, "type": kind, "code": code}})
            }
            Flavor::Cohere => json!({"message": message}),
            Flavor::Plain => json!({"error": message}),
        };
        (status, axum::Json(body)).into_response()
    }
}

pub struct Buffered {
    pub status: StatusCode,
    pub content_type: Option<HeaderValue>,
    pub body: Bytes,
}

fn request(svc: &ServicePool, idx: usize, path: &str, body: Bytes, fwd: &Forward) -> reqwest::RequestBuilder {
    let backend = &svc.backends[idx];
    let mut req = svc
        .client
        .post(format!("{}{}", backend.url, path))
        .timeout(Duration::from_millis(backend.request_timeout_ms.unwrap_or(svc.tuning.request_timeout_ms)))
        .header(header::CONTENT_TYPE, "application/json")
        .body(body);
    if let Some(v) = &fwd.authorization {
        req = req.header(header::AUTHORIZATION, v.clone());
    }
    if let Some(v) = &fwd.request_id {
        req = req.header(REQUEST_ID, v.clone());
    }
    req
}

/// 對單一後端送出並讀完整個回應。
pub async fn call_once(svc: &ServicePool, idx: usize, path: &str, body: Bytes, fwd: &Forward) -> Result<Buffered, CallError> {
    let backend = &svc.backends[idx];
    let _guard = backend.start_call();
    let result = async {
        let resp = request(svc, idx, path, body, fwd).send().await.map_err(|e| classify(&e))?;
        let status = resp.status();
        let content_type = resp.headers().get(header::CONTENT_TYPE).cloned();
        let body = resp.bytes().await.map_err(|e| classify(&e))?;
        if status.is_success() {
            Ok(Buffered { status, content_type, body })
        } else {
            Err(CallError::Status { status, content_type, body })
        }
    }
    .await;
    match &result {
        Err(e) if e.is_backend_failure() => backend.record_failure(&svc.tuning, &e.reason()),
        _ => backend.record_success(),
    }
    let outcome = match &result {
        Ok(buf) => buf.status.as_u16().to_string(),
        Err(CallError::Status { status, .. }) => status.as_u16().to_string(),
        Err(e) => e.reason(),
    };
    fwd.log.record(backend, outcome);
    result
}

/// 送出並在後端失敗時改送其他後端，最多 `max_retries` 次（embedding / rerank 為冪等運算）；
/// `use_fallback` 時，一般後端都失敗後若有 fallback 再送一次（不計入 `max_retries`）。
/// `parse` 檢查 2xx 回應；檢查失敗視同後端失敗。
pub async fn call_with_retry<T>(
    svc: &ServicePool,
    first: usize,
    path: &str,
    body: Bytes,
    fwd: &Forward,
    use_fallback: bool,
    parse: impl Fn(Buffered) -> Result<T, String>,
) -> Result<T, CallError> {
    let mut tried = vec![first];
    let mut idx = first;
    loop {
        let err = match call_once(svc, idx, path, body.clone(), fwd).await {
            Ok(buf) => match parse(buf) {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let err = CallError::Invalid(e);
                    svc.backends[idx].record_failure(&svc.tuning, &err.reason());
                    fwd.log.amend(&svc.backends[idx], err.reason());
                    err
                }
            },
            Err(e) => e,
        };
        if !err.is_backend_failure() {
            return Err(err);
        }
        let retry = if tried.len() <= svc.tuning.max_retries as usize { svc.pick(1, &tried).first().copied() } else { None };
        match retry.or_else(|| svc.fallback().filter(|f| use_fallback && !tried.contains(f))) {
            Some(next) => {
                let msg = if svc.backends[next].fallback { "falling back" } else { "retrying on another backend" };
                tracing::warn!(service = %svc.kind, backend = %svc.backends[idx].url, retry_to = %svc.backends[next].url,
                    reason = %err.reason(), "{msg}");
                idx = next;
                tried.push(idx);
            }
            None => return Err(err),
        }
    }
}

/// 透明轉發並以串流回傳 body；只在連線建立失敗時改送其他後端（decision 不保證冪等），
/// 一般後端都連不上（或都不可用）時改送 fallback。
pub async fn forward_streaming(svc: &ServicePool, path: &str, body: Bytes, fwd: &Forward) -> Result<Response, GatewayError> {
    let mut tried = Vec::new();
    let mut last_err: Option<CallError> = None;
    loop {
        let retry = if tried.len() <= svc.tuning.max_retries as usize { svc.pick(1, &tried).first().copied() } else { None };
        let Some(idx) = retry.or_else(|| svc.fallback().filter(|f| !tried.contains(f))) else {
            return Err(last_err.map(GatewayError::from).unwrap_or(GatewayError::NoBackend));
        };
        tried.push(idx);
        let backend = &svc.backends[idx];
        let guard = backend.start_call();
        match request(svc, idx, path, body.clone(), fwd).send().await {
            Ok(resp) => {
                let status = resp.status();
                fwd.log.record(backend, status.as_u16().to_string());
                if status.is_server_error() {
                    backend.record_failure(&svc.tuning, &format!("HTTP {}", status.as_u16()));
                } else {
                    backend.record_success();
                }
                let content_type = resp.headers().get(header::CONTENT_TYPE).cloned();
                // guard 隨串流結束才釋放，in-flight 數涵蓋整個回應傳輸
                let stream = resp.bytes_stream().map(move |chunk| {
                    let _ = &guard;
                    chunk
                });
                let mut out = Response::new(Body::from_stream(stream));
                *out.status_mut() = status;
                if let Some(ct) = content_type {
                    out.headers_mut().insert(header::CONTENT_TYPE, ct);
                }
                return Ok(out);
            }
            Err(e) => {
                let err = classify(&e);
                backend.record_failure(&svc.tuning, &err.reason());
                fwd.log.record(backend, err.reason());
                if !matches!(err, CallError::Connect(_)) {
                    return Err(err.into());
                }
                tracing::warn!(service = %svc.kind, backend = %backend.url, reason = %err.reason(), "connect failed, retrying");
                last_err = Some(err);
            }
        }
    }
}
