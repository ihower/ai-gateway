//! 以假後端驗證拆分合併、探測停用、重試與熱重載。

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use ai_gateway::config;
use ai_gateway::probe::build_pool;
use ai_gateway::server::{App, router};
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use serde_json::{Value, json};

const OK: u64 = 0;
const FAIL: u64 = 1;
const BAD_REQUEST: u64 = 2;

/// 假後端：embedding 向量為 `[字串長度, 後端 id]`；rerank 分數為文件長度；decision 回傳後端 id，
/// 若是相關度題目（`relevant`）則以 state 中 document 的長度為機率。
struct Mock {
    id: u64,
    hits: AtomicUsize,
    mode: AtomicU64,
    delay_ms: AtomicU64,
    /// 同時處理中的 decision request 數與其峰值。
    active: AtomicUsize,
    peak: AtomicUsize,
}

impl Mock {
    async fn pause(&self) {
        let ms = self.delay_ms.load(Ordering::Relaxed);
        if ms > 0 {
            tokio::time::sleep(Duration::from_millis(ms)).await;
        }
    }

    fn failure(&self, probe: bool) -> Option<Response> {
        if probe {
            return None;
        }
        self.hits.fetch_add(1, Ordering::Relaxed);
        match self.mode.load(Ordering::Relaxed) {
            FAIL => Some((StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response()),
            BAD_REQUEST => Some((StatusCode::BAD_REQUEST, axum::Json(json!({"error": "too long"}))).into_response()),
            _ => None,
        }
    }
}

async fn mock_embeddings(State(m): State<Arc<Mock>>, axum::Json(req): axum::Json<Value>) -> Response {
    let inputs: Vec<String> = match &req["input"] {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect(),
        _ => vec![],
    };
    if let Some(r) = m.failure(inputs == ["ping"]) {
        return r;
    }
    m.pause().await;
    // 刻意倒序回傳，驗證 gateway 依 index 排序
    let data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .rev()
        .map(|(i, s)| json!({"object": "embedding", "index": i, "embedding": [s.len() as f64, m.id as f64]}))
        .collect();
    let n = inputs.len();
    axum::Json(json!({"object": "list", "model": "mock", "data": data, "usage": {"prompt_tokens": n, "total_tokens": n}}))
        .into_response()
}

async fn mock_rerank(State(m): State<Arc<Mock>>, axum::Json(req): axum::Json<Value>) -> Response {
    let docs: Vec<String> = req["documents"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    if let Some(r) = m.failure(req["query"] == "ping") {
        return r;
    }
    m.pause().await;
    let with_docs = req["return_documents"] == true;
    let mut results: Vec<Value> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut r = json!({"index": i, "relevance_score": d.len() as f64 / 100.0});
            if with_docs {
                r["document"] = json!({"text": d});
            }
            r
        })
        .collect();
    results.sort_by(|a, b| b["relevance_score"].as_f64().partial_cmp(&a["relevance_score"].as_f64()).unwrap());
    if let Some(n) = req["top_n"].as_u64() {
        results.truncate(n as usize);
    }
    axum::Json(json!({"id": format!("mock-{}", m.id), "results": results, "meta": {"billed_units": {"search_units": 1}}}))
        .into_response()
}

async fn mock_decision(State(m): State<Arc<Mock>>, axum::Json(req): axum::Json<Value>) -> Response {
    if let Some(r) = m.failure(req["state"] == "ping") {
        return r;
    }
    let now = m.active.fetch_add(1, Ordering::SeqCst) + 1;
    m.peak.fetch_max(now, Ordering::SeqCst);
    m.pause().await;
    m.active.fetch_sub(1, Ordering::SeqCst);
    if req["questions"]["relevant"]["type"] == "noul" {
        let state = req["state"].as_str().unwrap();
        let doc = state.split_once("Document: ").map_or("", |(_, d)| d);
        return axum::Json(json!({"model": format!("mock-{}", m.id),
            "answers": {"relevant": {"type": "noul", "noul": doc.len() as f64 / 100.0}}, "usage": {"input_tokens": 10}}))
        .into_response();
    }
    axum::Json(json!({"model": "mock", "answers": {"backend": {"type": "choice", "choice": m.id.to_string()}}})).into_response()
}

async fn spawn_mock(id: u64) -> (String, Arc<Mock>) {
    let mock = Arc::new(Mock {
        id,
        hits: AtomicUsize::new(0),
        mode: AtomicU64::new(OK),
        delay_ms: AtomicU64::new(0),
        active: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
    });
    let app = Router::new()
        .route("/v1/embeddings", post(mock_embeddings))
        .route("/v2/rerank", post(mock_rerank))
        .route("/v1/systemone", post(mock_decision))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, mock)
}

/// 一個沒有人在聽的位址。
async fn dead_url() -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    drop(l);
    url
}

fn config_yaml(dir: &std::path::Path, emb: &[&str], rr: &[&str], dec: &[&str]) -> String {
    let list = |urls: &[&str]| urls.iter().map(|u| format!("\n      - url: \"{u}\"")).collect::<String>();
    format!(
        r#"
server:
  listen: "127.0.0.1:0"
  control_socket: "{sock}"
defaults:
  connect_timeout_ms: 300
  request_timeout_ms: 5000
probe:
  timeout_ms: 1000
services:
  embedding:
    backends:{e}
  reranker:
    backends:{r}
  decision:
    backends:{d}
"#,
        sock = dir.join("control.sock").display(),
        e = list(emb),
        r = list(rr),
        d = list(dec)
    )
}

struct Gateway {
    url: String,
    app: Arc<App>,
    config_path: std::path::PathBuf,
    _dir: tempfile::TempDir,
}

async fn start_gateway(yaml: impl FnOnce(&std::path::Path) -> String) -> (Gateway, ai_gateway::probe::LoadReport) {
    logs();
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.yaml");
    std::fs::write(&config_path, yaml(dir.path())).unwrap();
    let cfg = config::load(&config_path).unwrap();
    let (pool, report) = build_pool(cfg, 1, None).await;
    let app = Arc::new(App::new(pool, config_path.clone()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let r = router(app.clone());
    let r = r.into_make_service_with_connect_info::<std::net::SocketAddr>();
    tokio::spawn(async move { axum::serve(listener, r).await.unwrap() });
    (Gateway { url, app, config_path, _dir: dir }, report)
}

async fn post_json(url: &str, body: Value) -> (StatusCode, Value) {
    let resp = reqwest::Client::new().post(url).json(&body).send().await.unwrap();
    let status = resp.status();
    (status, resp.json().await.unwrap_or(Value::Null))
}

#[tokio::test]
async fn embeddings_are_split_and_merged_in_order() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (gw, report) = start_gateway(|d| config_yaml(d, &[&u1, &u2], &[&u1], &[&u1])).await;
    assert!(report.disabled.is_empty());

    let inputs = ["a", "bb", "ccc", "dddd", "eeeee"];
    let (status, body) = post_json(&format!("{}/v1/embeddings", gw.url), json!({"model": "x", "input": inputs})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 5);
    for (i, item) in data.iter().enumerate() {
        assert_eq!(item["index"], json!(i));
        assert_eq!(item["embedding"][0], json!((i + 1) as f64));
    }
    // 兩台各分到一段
    let backends: Vec<f64> = data.iter().map(|d| d["embedding"][1].as_f64().unwrap()).collect();
    assert!(backends.contains(&1.0) && backends.contains(&2.0));
    assert_eq!(body["usage"], json!({"prompt_tokens": 5, "total_tokens": 5}));
    assert_eq!(m1.hits.load(Ordering::Relaxed) + m2.hits.load(Ordering::Relaxed), 2);

    // 單一字串：只打一台
    let (status, body) = post_json(&format!("{}/v1/embeddings", gw.url), json!({"input": "hello"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"][0]["embedding"][0], json!(5.0));
    assert_eq!(m1.hits.load(Ordering::Relaxed) + m2.hits.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn unreachable_backend_is_disabled_at_startup() {
    let (u1, m1) = spawn_mock(1).await;
    let dead = dead_url().await;
    let (gw, report) = start_gateway(|d| config_yaml(d, &[&u1, &dead], &[&u1], &[&u1])).await;
    assert_eq!(report.disabled.len(), 1);
    assert_eq!(report.disabled[0].url, dead);

    let (status, body) = post_json(&format!("{}/v1/embeddings", gw.url), json!({"input": ["a", "b", "c"]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(m1.hits.load(Ordering::Relaxed), 1, "all inputs go to the only enabled backend");
    let errors = gw.app.pool.load().embedding.backends[1].stats.errors.load(Ordering::Relaxed);
    assert_eq!(errors, 0, "disabled backend is never called");
}

#[tokio::test]
async fn all_disabled_returns_503() {
    let (u1, _m1) = spawn_mock(1).await;
    let dead = dead_url().await;
    let (gw, _) = start_gateway(|d| config_yaml(d, &[&u1], &[&u1], &[&dead])).await;
    let (status, _) = post_json(&format!("{}/v1/systemone", gw.url), json!({"state": "x", "questions": {}})).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let resp = reqwest::get(format!("{}/readyz", gw.url)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn rerank_merges_by_score_with_top_n() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (gw, _) = start_gateway(|d| config_yaml(d, &[&u1], &[&u1, &u2], &[&u1])).await;
    let docs = ["aaa", "a", "aaaaa", "aa", "aaaa"];
    let (status, body) =
        post_json(&format!("{}/v2/rerank", gw.url), json!({"query": "q", "documents": docs, "top_n": 3})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let idx: Vec<u64> = body["results"].as_array().unwrap().iter().map(|r| r["index"].as_u64().unwrap()).collect();
    assert_eq!(idx, vec![2, 4, 0]);
    assert_eq!(body["meta"]["billed_units"]["search_units"], json!(2));
    assert!(!body["id"].as_str().unwrap().starts_with("mock-"));
    assert_eq!(m1.hits.load(Ordering::Relaxed), 1);
    assert_eq!(m2.hits.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn decision_round_robins_whole_request() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (gw, _) = start_gateway(|d| config_yaml(d, &[&u1], &[&u1], &[&u1, &u2])).await;
    let mut seen = Vec::new();
    for _ in 0..4 {
        let resp = reqwest::Client::new()
            .post(format!("{}/v1/systemone", gw.url))
            .header("x-request-id", "abc")
            .json(&json!({"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.headers()["x-request-id"], "abc");
        let body: Value = resp.json().await.unwrap();
        seen.push(body["answers"]["backend"]["choice"].as_str().unwrap().to_string());
    }
    assert_eq!(seen, vec!["1", "2", "1", "2"]);
    assert_eq!(m1.hits.load(Ordering::Relaxed), 2);
    assert_eq!(m2.hits.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn failures_retry_elsewhere_and_mark_unhealthy() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (gw, _) = start_gateway(|d| config_yaml(d, &[&u1, &u2], &[&u1], &[&u1])).await;
    m1.mode.store(FAIL, Ordering::Relaxed);
    for _ in 0..4 {
        let (status, body) = post_json(&format!("{}/v1/embeddings", gw.url), json!({"input": "x"})).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["data"][0]["embedding"][1], json!(2.0));
    }
    let pool = gw.app.pool.load();
    assert_eq!(pool.embedding.backends[0].state(), ai_gateway::backend::BackendState::Unhealthy);
    // 第 4 次起直接送到健康的後端
    let m1_hits = m1.hits.load(Ordering::Relaxed);
    assert!(m1_hits <= 3, "backend 1 got {m1_hits} hits");
    assert_eq!(m2.hits.load(Ordering::Relaxed), 4);

    // 兩台都壞：502
    m2.mode.store(FAIL, Ordering::Relaxed);
    let (status, body) = post_json(&format!("{}/v1/embeddings", gw.url), json!({"input": "x"})).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"]["code"], json!("bad_gateway"));
}

#[tokio::test]
async fn client_errors_pass_through_without_retry() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (gw, _) = start_gateway(|d| config_yaml(d, &[&u1, &u2], &[&u1], &[&u1])).await;
    m1.mode.store(BAD_REQUEST, Ordering::Relaxed);
    m2.mode.store(BAD_REQUEST, Ordering::Relaxed);
    let (status, body) = post_json(&format!("{}/v1/embeddings", gw.url), json!({"input": ["a", "b"]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body, json!({"error": "too long"}));
    let (status, _) = post_json(&format!("{}/v1/embeddings", gw.url), json!("not an object")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(m1.hits.load(Ordering::Relaxed) + m2.hits.load(Ordering::Relaxed) <= 2);
}

#[tokio::test]
async fn reload_swaps_backends_and_lets_inflight_requests_finish() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (gw, _) = start_gateway(|d| config_yaml(d, &[&u1], &[&u1], &[&u1])).await;

    // 進行中的 request 打在 u1，且很慢
    m1.delay_ms.store(500, Ordering::Relaxed);
    let url = format!("{}/v1/embeddings", gw.url);
    let inflight = tokio::spawn(async move { post_json(&url, json!({"input": "slow"})).await });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 改成只有 u2 與一台連不上的後端
    let dead = dead_url().await;
    let dir = gw.config_path.parent().unwrap().to_path_buf();
    std::fs::write(&gw.config_path, config_yaml(&dir, &[&u2, &dead], &[&u2], &[&u2])).unwrap();
    let report = gw.app.reload().await.unwrap();
    assert_eq!(report.version, 2);
    assert_eq!(report.changes["embedding"].added, vec![u2.clone(), dead.clone()]);
    assert_eq!(report.changes["embedding"].removed, vec![u1.clone()]);
    assert_eq!(report.disabled.len(), 1);

    let (status, body) = inflight.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"][0]["embedding"][1], json!(1.0), "in-flight request finished on the removed backend");

    let before = m1.hits.load(Ordering::Relaxed);
    for _ in 0..3 {
        let (status, body) = post_json(&format!("{}/v1/embeddings", gw.url), json!({"input": "x"})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"][0]["embedding"][1], json!(2.0));
    }
    assert_eq!(m1.hits.load(Ordering::Relaxed), before);
    assert_eq!(m2.hits.load(Ordering::Relaxed), 3);

    // 設定檔壞掉：保留舊設定
    std::fs::write(&gw.config_path, "server: [").unwrap();
    assert!(gw.app.reload().await.is_err());
    assert_eq!(gw.app.pool.load().version, 2);
}

fn with_fallback(yaml: String, url: &str) -> String {
    with_fallback_on(yaml, "embedding", url)
}

/// 在 `service` 區塊末尾（下一個 service 標題之前）加上 fallback。
fn with_fallback_on(yaml: String, service: &str, url: &str) -> String {
    let block = format!("    fallback:\n      url: \"{url}\"\n      request_timeout_ms: 8000\n");
    let mut out = String::new();
    let mut inside = false;
    for line in yaml.lines() {
        let is_header = line.starts_with("  ") && !line.starts_with("   ") && line.ends_with(':');
        if inside && is_header {
            out.push_str(&block);
            inside = false;
        }
        if is_header && line.trim() == format!("{service}:") {
            inside = true;
        }
        out.push_str(line);
        out.push('\n');
    }
    if inside {
        out.push_str(&block);
    }
    out
}

#[tokio::test]
async fn embedding_falls_back_when_all_backends_fail() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (fb, mf) = spawn_mock(9).await;
    let (gw, report) = start_gateway(|d| with_fallback(config_yaml(d, &[&u1, &u2], &[&u1], &[&u1]), &fb)).await;
    assert!(report.disabled.is_empty());
    assert_eq!(report.changes["embedding"].added, vec![u1.clone(), u2.clone(), fb.clone()]);
    let url = format!("{}/v1/embeddings", gw.url);

    // 一般後端正常時不使用 fallback
    let (status, _) = post_json(&url, json!({"input": ["a", "b", "c"]})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(mf.hits.load(Ordering::Relaxed), 0);

    // 一台壞：在另一台重試，仍不使用 fallback
    m1.mode.store(FAIL, Ordering::Relaxed);
    let (status, body) = post_json(&url, json!({"input": ["a", "b", "c", "d"]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(mf.hits.load(Ordering::Relaxed), 0);

    // 兩台都壞：失敗的那份改送 fallback，結果依序合併
    m2.mode.store(FAIL, Ordering::Relaxed);
    let (status, body) = post_json(&url, json!({"input": ["a", "bb", "ccc"]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.iter().map(|d| d["embedding"][0].as_f64().unwrap()).collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
    assert!(data.iter().all(|d| d["embedding"][1] == json!(9.0)));

    // 兩台都進入 unhealthy 後，直接送 fallback，不再先打壞掉的後端
    for _ in 0..3 {
        post_json(&url, json!({"input": "x"})).await;
    }
    let pool = gw.app.pool.load();
    assert!(pool.embedding.backends[..2].iter().all(|b| b.state() == ai_gateway::backend::BackendState::Unhealthy));
    let before = m1.hits.load(Ordering::Relaxed) + m2.hits.load(Ordering::Relaxed);
    let fb_before = mf.hits.load(Ordering::Relaxed);
    let (status, body) = post_json(&url, json!({"input": ["a", "b", "c", "d"]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(m1.hits.load(Ordering::Relaxed) + m2.hits.load(Ordering::Relaxed), before);
    assert_eq!(mf.hits.load(Ordering::Relaxed), fb_before + 1, "whole request goes to fallback unsplit");
    assert!(pool.status().services["embedding"][2].fallback);

    // fallback 也壞：502
    mf.mode.store(FAIL, Ordering::Relaxed);
    let (status, _) = post_json(&url, json!({"input": "x"})).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn fallback_serves_when_all_backends_disabled_at_startup() {
    let (fb, mf) = spawn_mock(9).await;
    let dead = dead_url().await;
    let (gw, report) = start_gateway(|d| with_fallback(config_yaml(d, &[&dead], &[&fb], &[&fb]), &fb)).await;
    assert_eq!(report.disabled.len(), 1);
    let resp = reqwest::get(format!("{}/readyz", gw.url)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let (status, body) = post_json(&format!("{}/v1/embeddings", gw.url), json!({"input": ["a", "b"]})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"][1]["embedding"], json!([1.0, 9.0]));
    assert_eq!(mf.hits.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn rerank_falls_back_with_whole_request() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (fb, mf) = spawn_mock(9).await;
    let (gw, _) = start_gateway(|d| with_fallback_on(config_yaml(d, &[&u1], &[&u1, &u2], &[&u1]), "reranker", &fb)).await;
    assert!(gw.app.pool.load().embedding.fallback().is_none());
    let url = format!("{}/v2/rerank", gw.url);
    let body = json!({"query": "q", "documents": ["aaa", "a", "aaaaa", "aa"], "top_n": 2});

    let (status, out) = post_json(&url, body.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_ne!(out["id"], json!("mock-9"));
    assert_eq!(mf.hits.load(Ordering::Relaxed), 0);

    // 拆分後一份在兩台都失敗：不只補那一份，而是整個 request 改送 fallback（分數尺度不同，不能混排）
    m1.mode.store(FAIL, Ordering::Relaxed);
    m2.mode.store(FAIL, Ordering::Relaxed);
    let (status, out) = post_json(&url, body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(out["id"], json!("mock-9"), "fallback response passed through unmerged");
    let idx: Vec<u64> = out["results"].as_array().unwrap().iter().map(|r| r["index"].as_u64().unwrap()).collect();
    assert_eq!(idx, vec![2, 0]);
    assert_eq!(mf.hits.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn decision_falls_back_when_backends_unavailable() {
    let (u1, m1) = spawn_mock(1).await;
    let (fb, mf) = spawn_mock(9).await;
    let dead = dead_url().await;
    let (gw, report) = start_gateway(|d| with_fallback_on(config_yaml(d, &[&u1], &[&u1], &[&u1, &dead]), "decision", &fb)).await;
    assert_eq!(report.disabled.len(), 1);
    let url = format!("{}/v1/systemone", gw.url);
    let body = json!({"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}});
    let choice = |v: &Value| v["answers"]["backend"]["choice"].as_str().unwrap().to_string();

    let (_, out) = post_json(&url, body.clone()).await;
    assert_eq!(choice(&out), "1");

    // 5xx 不重試、不改送（decision 不保證冪等），原樣回傳；連續失敗後標記 unhealthy
    m1.mode.store(FAIL, Ordering::Relaxed);
    for _ in 0..3 {
        let resp = reqwest::Client::new().post(&url).json(&body).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
    assert_eq!(mf.hits.load(Ordering::Relaxed), 0);
    // 唯一啟用的後端 unhealthy → 改送 fallback
    let (status, out) = post_json(&url, body.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(choice(&out), "9");
    assert_eq!(mf.hits.load(Ordering::Relaxed), 1);
}

/// 在 `services.<service>` 底下加上 `chunking` 設定。
fn with_chunking(yaml: String, service: &str, chunking: &str) -> String {
    let anchor = format!("  {service}:\n");
    yaml.replace(&anchor, &format!("{anchor}    chunking: {chunking}\n"))
}

// 「aaaa一一一一」共 8 個字元，以 4 / 重疊 1 切成「aaaa」「a一一一」「一一一一」；
// 假後端以 byte 數為分數，三段分別是 0.04、0.10、0.12，取最高分應為 0.12（不切則是 16 bytes 的 0.16）
const MIXED: &str = "aaaa一一一一";

#[tokio::test]
async fn rerank_chunking_does_not_forward_top_n_to_sub_requests() {
    let (u1, _m1) = spawn_mock(1).await;
    let (gw, _) = start_gateway(|d| {
        with_chunking(config_yaml(d, &[&u1], &[&u1], &[&u1]), "reranker", "{max_chars: 4, overlap_chars: 1}")
    })
    .await;
    // "aaaaaaa" 切成兩段（各 0.04），加上 "a"（0.01）共 3 段送同一台；
    // 若把 top_n=2 原樣轉給後端，只會回兩段（都屬於 index 0），index 1 就從結果裡消失
    let (status, out) = post_json(
        &format!("{}/v2/rerank", gw.url),
        json!({"query": "q", "documents": ["aaaaaaa", "a"], "top_n": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(out["results"], json!([{"index": 0, "relevance_score": 0.04}, {"index": 1, "relevance_score": 0.01}]));
}

#[tokio::test]
async fn rerank_chunks_long_documents_and_takes_max() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (fb, mf) = spawn_mock(9).await;
    let (gw, _) = start_gateway(|d| {
        let yaml = with_fallback_on(config_yaml(d, &[&u1], &[&u1, &u2], &[&u1]), "reranker", &fb);
        with_chunking(yaml, "reranker", "{max_chars: 4, overlap_chars: 1}")
    })
    .await;
    let url = format!("{}/v2/rerank", gw.url);
    let body = json!({"query": "q", "documents": ["aaa", MIXED, "aa", "aaaaaaa"], "top_n": 2, "return_documents": true});
    let resp = reqwest::Client::new().post(&url).header("x-request-id", "log-chunks").json(&body).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let out: Value = resp.json().await.unwrap();
    // top_n 在收回原始 document 之後才套用；document 換回原文
    assert_eq!(
        out["results"],
        json!([
            {"index": 1, "relevance_score": 0.12, "document": {"text": MIXED}},
            {"index": 3, "relevance_score": 0.04, "document": {"text": "aaaaaaa"}}
        ])
    );
    // 1 + 3 + 1 + 2 = 7 段，依權重分給兩台
    assert_eq!((m1.hits.load(Ordering::Relaxed), m2.hits.load(Ordering::Relaxed)), (1, 1));
    let f = access_line("log-chunks");
    assert_eq!((f["items"].as_u64(), f["chunks"].as_u64()), (Some(4), Some(7)));

    // 只有一份 document 也會切段、合併
    let (status, out) = post_json(&url, json!({"query": "q", "documents": [MIXED]})).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(out["results"], json!([{"index": 0, "relevance_score": 0.12}]));

    // 沒有 document 需要切時照舊，access log 不記 chunks
    let resp = reqwest::Client::new()
        .post(&url)
        .header("x-request-id", "log-no-chunks")
        .json(&json!({"query": "q", "documents": ["a", "aa"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(access_line("log-no-chunks").get("chunks").is_none());

    // 兩台都失敗：整個原始 request（未切段）改送 fallback
    m1.mode.store(FAIL, Ordering::Relaxed);
    m2.mode.store(FAIL, Ordering::Relaxed);
    let (status, out) = post_json(&url, body).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(out["id"], json!("mock-9"));
    assert_eq!(out["results"][0]["relevance_score"], json!(0.16));
    assert_eq!(mf.hits.load(Ordering::Relaxed), 1);
}

/// 在設定檔末尾（`services` 底下）加上 `decision_reranker` 區塊。
fn with_decision_reranker(yaml: String, body: &str) -> String {
    format!("{yaml}  decision_reranker:\n{body}")
}

#[tokio::test]
async fn decision_rerank_judges_each_document_and_sorts() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (gw, _) = start_gateway(|d| {
        with_decision_reranker(config_yaml(d, &[&u1], &[&u1], &[&u1, &u2]), "    max_document_chars: 4\n")
    })
    .await;
    let url = format!("{}/decision/v2/rerank", gw.url);
    let docs = json!(["aaa", "a", "aaaaaaaa", {"text": "aa"}]);
    let (status, out) = post_json(&url, json!({"query": "q", "documents": docs, "top_n": 3})).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    // 第 3 份截斷到 4 個字元後才判斷，回傳的 document 仍是原文
    assert_eq!(
        out["results"],
        json!([
            {"index": 2, "relevance_score": 0.04, "document": {"text": "aaaaaaaa"}},
            {"index": 0, "relevance_score": 0.03, "document": {"text": "aaa"}},
            {"index": 3, "relevance_score": 0.02, "document": {"text": "aa"}}
        ])
    );
    assert_eq!(out["usage"], json!({"prompt_tokens": 40}));
    // 每份 document 一個 decision 呼叫，依權重分到兩台
    assert_eq!((m1.hits.load(Ordering::Relaxed), m2.hits.load(Ordering::Relaxed)), (2, 2));

    let (status, out) = post_json(&url, json!({"query": "q", "documents": []})).await;
    assert_eq!((status, out["results"].clone()), (StatusCode::OK, json!([])));
    let (status, _) = post_json(&url, json!({"query": "q", "documents": [1]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // 一般 rerank 端點不受影響
    let (status, _) = post_json(&format!("{}/v2/rerank", gw.url), json!({"query": "q", "documents": ["a"]})).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn decision_rerank_is_404_when_not_configured() {
    let (u1, _) = spawn_mock(1).await;
    let (gw, _) = start_gateway(|d| config_yaml(d, &[&u1], &[&u1], &[&u1])).await;
    let (status, out) = post_json(&format!("{}/decision/v2/rerank", gw.url), json!({"query": "q", "documents": ["a"]})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(out["message"].as_str().unwrap().contains("not configured"));
}

#[tokio::test]
async fn decision_rerank_limits_concurrency() {
    let (u1, m1) = spawn_mock(1).await;
    m1.delay_ms.store(50, Ordering::Relaxed);
    let (gw, _) = start_gateway(|d| {
        with_decision_reranker(config_yaml(d, &[&u1], &[&u1], &[&u1]), "    max_concurrency: 2\n    max_documents: 6\n")
    })
    .await;
    let url = format!("{}/decision/v2/rerank", gw.url);
    let (status, out) = post_json(&url, json!({"query": "q", "documents": ["a", "b", "c", "d", "e", "f"]})).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(m1.hits.load(Ordering::Relaxed), 6);
    assert_eq!(m1.peak.load(Ordering::SeqCst), 2);

    let (status, out) = post_json(&url, json!({"query": "q", "documents": vec!["a"; 7]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(out["message"].as_str().unwrap().contains("too many documents"));
}

#[tokio::test]
async fn decision_rerank_falls_back_with_whole_request() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (fb, mf) = spawn_mock(9).await;
    let yaml = |d: &std::path::Path, extra: &str| {
        with_decision_reranker(with_fallback_on(config_yaml(d, &[&u1], &[&u1], &[&u1, &u2]), "decision", &fb), extra)
    };
    let (gw, _) = start_gateway(|d| yaml(d, "    max_concurrency: 1\n")).await;
    let url = format!("{}/decision/v2/rerank", gw.url);
    let body = json!({"query": "q", "documents": ["aaa", "a", "aa"]});

    // 只有一台壞：該筆在另一台重試，不用 fallback
    m1.mode.store(FAIL, Ordering::Relaxed);
    let (status, out) = post_json(&url, body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(out["model"], json!("mock-2"));
    assert_eq!(mf.hits.load(Ordering::Relaxed), 0);

    // 兩台都壞：不只補失敗那筆，整個 request 改送 fallback（分數不能混排）
    m2.mode.store(FAIL, Ordering::Relaxed);
    let (status, out) = post_json(&url, body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(out["model"], json!("mock-9"));
    let idx: Vec<u64> = out["results"].as_array().unwrap().iter().map(|r| r["index"].as_u64().unwrap()).collect();
    assert_eq!(idx, vec![0, 2, 1]);
    assert_eq!(mf.hits.load(Ordering::Relaxed), 3);

    // use_fallback: false 時直接回錯誤
    let (gw, _) = start_gateway(|d| yaml(d, "    use_fallback: false\n")).await;
    m1.mode.store(FAIL, Ordering::Relaxed);
    m2.mode.store(FAIL, Ordering::Relaxed);
    let (status, out) = post_json(&format!("{}/decision/v2/rerank", gw.url), body).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY, "{out}");
    assert!(out["message"].is_string());
    assert_eq!(mf.hits.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn decision_rerank_chunks_long_documents_and_takes_max() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (fb, mf) = spawn_mock(9).await;
    let (gw, _) = start_gateway(|d| {
        let yaml = with_fallback_on(config_yaml(d, &[&u1], &[&u1], &[&u1, &u2]), "decision", &fb);
        with_decision_reranker(yaml, "    chunking: {max_chars: 4, overlap_chars: 1}\n")
    })
    .await;
    let url = format!("{}/decision/v2/rerank", gw.url);
    let body = json!({"query": "q", "documents": ["aaa", MIXED, "a"]});
    let (status, out) = post_json(&url, body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(
        out["results"],
        json!([
            {"index": 1, "relevance_score": 0.12, "document": {"text": MIXED}},
            {"index": 0, "relevance_score": 0.03, "document": {"text": "aaa"}},
            {"index": 2, "relevance_score": 0.01, "document": {"text": "a"}}
        ])
    );
    // 1 + 3 + 1 = 5 段，每段一個 decision 呼叫
    assert_eq!(m1.hits.load(Ordering::Relaxed) + m2.hits.load(Ordering::Relaxed), 5);
    assert_eq!(out["usage"], json!({"prompt_tokens": 50}));

    // 兩台都失敗：fallback 也照樣切段
    m1.mode.store(FAIL, Ordering::Relaxed);
    m2.mode.store(FAIL, Ordering::Relaxed);
    let (status, out) = post_json(&url, body).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(out["model"], json!("mock-9"));
    assert_eq!(out["results"][0], json!({"index": 1, "relevance_score": 0.12, "document": {"text": MIXED}}));
    assert_eq!(mf.hits.load(Ordering::Relaxed), 5);
}

/// 收集 tracing 輸出（JSON），供檢查 access log。
#[derive(Clone, Default)]
struct LogBuf(Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for LogBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// 全域 subscriber 在第一個 gateway 啟動前裝好（scoped subscriber 收不到其他測試已快取的 callsite）。
fn logs() -> &'static LogBuf {
    static LOGS: std::sync::OnceLock<LogBuf> = std::sync::OnceLock::new();
    LOGS.get_or_init(|| {
        let buf = LogBuf::default();
        let writer = buf.clone();
        tracing_subscriber::fmt().json().with_writer(move || writer.clone()).init();
        buf
    })
}

/// 並行的測試共用同一份 log，以 request id 篩選。
fn access_line(request_id: &str) -> Value {
    let text = String::from_utf8(logs().0.lock().unwrap().clone()).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .filter(|v| v["target"] == "access" && v["fields"]["request_id"] == request_id)
        .collect();
    assert_eq!(lines.len(), 1, "{request_id}: {lines:?}");
    lines[0]["fields"].clone()
}

#[tokio::test]
async fn access_log_records_each_request_with_backends() {
    let (u1, m1) = spawn_mock(1).await;
    let (u2, m2) = spawn_mock(2).await;
    let (fb, _mf) = spawn_mock(9).await;
    let (gw, _) = start_gateway(|d| with_fallback(config_yaml(d, &[&u1, &u2], &[&u1], &[&u1]), &fb)).await;
    let send = |path: &str, id: &str, body: Value| {
        reqwest::Client::new().post(format!("{}{path}", gw.url)).header("x-request-id", id).json(&body).send()
    };

    let resp = send("/v1/embeddings", "log-ok", json!({"input": ["a", "b", "c"]})).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let health = reqwest::Client::new().get(format!("{}/healthz", gw.url)).header("x-request-id", "log-health");
    health.send().await.unwrap();
    m1.mode.store(FAIL, Ordering::Relaxed);
    m2.mode.store(FAIL, Ordering::Relaxed);
    let resp = send("/v1/embeddings", "log-fallback", json!({"input": "x"})).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = send("/v1/systemone", "log-decision", json!({"state": "s", "questions": {}})).await.unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let f = access_line("log-ok");
    assert_eq!(f["client"], "127.0.0.1");
    assert_eq!((f["method"].as_str(), f["path"].as_str(), f["status"].as_u64()), (Some("POST"), Some("/v1/embeddings"), Some(200)));
    assert_eq!(f["items"], 3);
    assert_eq!(f["fallback"], false);
    let backends = f["backends"].as_str().unwrap();
    assert!(backends.contains(&format!("{u1} 200")) && backends.contains(&format!("{u2} 200")), "{backends}");
    assert!(f["duration_ms"].as_f64().unwrap() >= 0.0);

    // 兩台都失敗後改送 fallback：依序記錄每次嘗試
    let f = access_line("log-fallback");
    assert_eq!(f["fallback"], true);
    let backends = f["backends"].as_str().unwrap();
    assert!(backends.ends_with(&format!("{fb} 200")), "{backends}");
    assert_eq!(backends.matches(" 500").count(), 2, "{backends}");

    let f = access_line("log-decision");
    assert_eq!(f["status"], 500);
    assert_eq!(f["backends"], format!("{u1} 500"));
    assert!(f.get("items").is_none());

    // 健康檢查不記錄
    let text = String::from_utf8(logs().0.lock().unwrap().clone()).unwrap();
    assert!(!text.contains("log-health"));
}
