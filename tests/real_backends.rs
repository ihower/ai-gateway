//! 對 `config.yaml` 中的真實後端測試：同一個 request 直接打後端與經過 gateway，比對回應格式與內容。
//! 需要後端都在執行，因此預設略過：
//!
//! ```bash
//! cargo test --test real_backends -- --ignored --nocapture
//! ```

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use ai_gateway::config::{self, ServiceKind};
use ai_gateway::probe::build_pool;
use ai_gateway::server::{App, router};
use base64::Engine;
use reqwest::StatusCode;
use serde_json::{Value, json};

struct Env {
    gateway: String,
    app: Arc<App>,
    http: reqwest::Client,
}

impl Env {
    async fn start() -> Env {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.yaml");
        let cfg = config::load(&path).unwrap();
        let (pool, report) = build_pool(cfg, 1, None).await;
        assert!(report.disabled.is_empty(), "all real backends must be up: {:?}", report.disabled);
        let app = Arc::new(App::new(pool, path));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway = format!("http://{}", listener.local_addr().unwrap());
        let r = router(app.clone());
        tokio::spawn(async move { axum::serve(listener, r).await.unwrap() });
        Env { gateway, app, http: reqwest::Client::new() }
    }

    fn backends(&self, kind: ServiceKind) -> Vec<String> {
        self.app.pool.load().service(kind).backends.iter().map(|b| b.url.clone()).collect()
    }

    fn requests(&self, kind: ServiceKind) -> Vec<u64> {
        self.app.pool.load().service(kind).backends.iter().map(|b| b.stats.requests.load(Ordering::Relaxed)).collect()
    }

    async fn post(&self, base: &str, path: &str, body: &Value) -> (StatusCode, Value) {
        let resp = self.http.post(format!("{base}{path}")).json(body).send().await.unwrap();
        let status = resp.status();
        let text = resp.text().await.unwrap();
        (status, serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }

    async fn via(&self, path: &str, body: &Value) -> (StatusCode, Value) {
        self.post(&self.gateway, path, body).await
    }

    /// 直接打每一台後端。
    async fn direct_all(&self, kind: ServiceKind, path: &str, body: &Value) -> Vec<(StatusCode, Value)> {
        let mut out = Vec::new();
        for b in self.backends(kind) {
            out.push(self.post(&b, path, body).await);
        }
        out
    }
}

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default()
}

fn floats(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap()).collect(),
        Value::String(s) => base64::engine::general_purpose::STANDARD
            .decode(s)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c) as f64)
            .collect(),
        _ => panic!("unexpected embedding {v}"),
    }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// gateway 的每一筆向量都要與某台後端對同一 input 的結果幾乎相同（兩台後端數值略有差異）。
fn assert_embeddings_match(gw: &Value, directs: &[(StatusCode, Value)], label: &str) {
    let data = gw["data"].as_array().unwrap();
    for (i, item) in data.iter().enumerate() {
        assert_eq!(item["index"], json!(i), "{label}: index order");
        let v = floats(&item["embedding"]);
        let best = directs
            .iter()
            .map(|(_, d)| cosine(&v, &floats(&d["data"][i]["embedding"])))
            .fold(f64::MIN, f64::max);
        assert!(best > 0.999, "{label}: input {i} cosine {best}");
    }
}

/// 格式比對：頂層欄位與每筆 data / results 的欄位集合與後端相同。
fn assert_same_shape(gw: &Value, direct: &Value, list_key: &str, label: &str) {
    assert_eq!(keys(gw), keys(direct), "{label}: top-level keys");
    let (g, d) = (&gw[list_key][0], &direct[list_key][0]);
    assert_eq!(keys(g), keys(d), "{label}: {list_key} item keys");
    if direct.get("usage").is_some() {
        assert_eq!(keys(&gw["usage"]), keys(&direct["usage"]), "{label}: usage keys");
    }
}

const TEXTS: [&str; 7] = [
    "The capital of France is Paris.",
    "Rust is a systems programming language.",
    "台北是台灣的首都。",
    "Bananas are rich in potassium.",
    "The quick brown fox jumps over the lazy dog.",
    "Embedding models map text to vectors.",
    "東京は日本の首都です。",
];

#[tokio::test]
#[ignore]
async fn embedding_requests() {
    let env = Env::start().await;
    let path = "/v1/embeddings";
    let before = env.requests(ServiceKind::Embedding);

    let cases = [
        ("single string", json!({"input": TEXTS[0]})),
        ("single item list", json!({"input": [TEXTS[1]]})),
        ("list of 2", json!({"input": &TEXTS[..2]})),
        ("list of 7", json!({"input": TEXTS})),
        ("with model", json!({"model": "voyage-4-nano", "input": TEXTS})),
        ("base64", json!({"input": TEXTS, "encoding_format": "base64"})),
        ("float explicit", json!({"input": &TEXTS[..3], "encoding_format": "float"})),
        ("dimensions 256", json!({"input": TEXTS, "dimensions": 256})),
        ("input_type query", json!({"input": &TEXTS[..4], "input_type": "query"})),
        ("input_type document", json!({"input": &TEXTS[..4], "input_type": "document"})),
        ("token ids", json!({"input": [[101, 102, 103], [104, 105], [106]], "input_type": "none"})),
        ("user field", json!({"input": &TEXTS[..3], "user": "tester"})),
    ];
    for (label, body) in cases {
        let directs = env.direct_all(ServiceKind::Embedding, path, &body).await;
        let (status, gw) = env.via(path, &body).await;
        assert_eq!(status, StatusCode::OK, "{label}: {gw}");
        assert_eq!(directs[0].0, StatusCode::OK, "{label}: direct {}", directs[0].1);
        let direct = &directs[0].1;
        assert_same_shape(&gw, direct, "data", label);
        assert_eq!(gw["data"].as_array().unwrap().len(), direct["data"].as_array().unwrap().len(), "{label}: count");
        assert_eq!(gw["model"], direct["model"], "{label}: model");
        assert_eq!(gw["object"], direct["object"], "{label}: object");
        assert_eq!(gw["usage"]["prompt_tokens"], direct["usage"]["prompt_tokens"], "{label}: prompt_tokens");
        assert_eq!(gw["usage"]["total_tokens"], direct["usage"]["total_tokens"], "{label}: total_tokens");
        assert_embeddings_match(&gw, &directs, label);
        if label == "dimensions 256" {
            assert_eq!(floats(&gw["data"][0]["embedding"]).len(), 256);
        }
        if label == "base64" {
            assert!(gw["data"][0]["embedding"].is_string());
        }
        println!("ok  embedding  {label:<20} n={} keys={:?}", gw["data"].as_array().unwrap().len(), keys(&gw));
    }

    // 64 筆：順序正確
    let many: Vec<String> = (0..64).map(|i| format!("sentence number {i} about topic {}", i % 7)).collect();
    let body = json!({"input": many});
    let directs = env.direct_all(ServiceKind::Embedding, path, &body).await;
    let (status, gw) = env.via(path, &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_embeddings_match(&gw, &directs, "64 inputs");
    assert_eq!(gw["usage"]["prompt_tokens"], directs[0].1["usage"]["prompt_tokens"]);
    println!("ok  embedding  64 inputs            order and vectors match");

    let after = env.requests(ServiceKind::Embedding);
    let used: Vec<u64> = after.iter().zip(&before).map(|(a, b)| a - b).collect();
    assert!(used.iter().all(|n| *n > 0), "both backends should get traffic: {used:?}");
    println!("ok  embedding  traffic per backend {used:?}");

    // 錯誤：後端錯誤原樣回傳；gateway 自身錯誤用 OpenAI 格式
    let errors = [
        ("unknown model", json!({"input": ["a", "b"], "model": "nope"})),
        ("empty list", json!({"input": []})),
        ("missing input", json!({})),
        ("token ids without input_type", json!({"input": [[1, 2], [3]]})),
    ];
    for (label, body) in errors {
        let (dstatus, direct) = env.post(&env.backends(ServiceKind::Embedding)[0], path, &body).await;
        let (status, gw) = env.via(path, &body).await;
        assert_eq!(status, dstatus, "{label}: {gw}");
        assert_eq!(keys(&gw["error"]), keys(&direct["error"]), "{label}: error keys");
        println!("ok  embedding  error {label:<22} HTTP {}", status.as_u16());
    }
    let resp = env.http.post(format!("{}{path}", env.gateway)).body("{not json").send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let gw: Value = resp.json().await.unwrap();
    assert_eq!(gw["error"]["code"], json!("bad_request"));
    println!("ok  embedding  error invalid JSON         HTTP 400 {}", gw["error"]["message"]);
}

const QUERY: &str = "What is the capital of France?";
const DOCS: [&str; 8] = [
    "Berlin is the capital of Germany.",
    "Paris is the capital and largest city of France.",
    "Bananas are rich in potassium.",
    "France is a country in Western Europe; its capital is Paris.",
    "The Eiffel Tower is located in Paris.",
    "Rust is a systems programming language.",
    "Lyon is the third-largest city in France.",
    "Madrid is the capital of Spain.",
];

fn result_indexes(v: &Value) -> Vec<u64> {
    v["results"].as_array().unwrap().iter().map(|r| r["index"].as_u64().unwrap()).collect()
}

#[tokio::test]
#[ignore]
async fn rerank_requests() {
    let env = Env::start().await;
    let before = env.requests(ServiceKind::Reranker);

    let cases = [
        ("v1 all docs", "/v1/rerank", json!({"query": QUERY, "documents": DOCS})),
        ("v2 all docs", "/v2/rerank", json!({"query": QUERY, "documents": DOCS})),
        ("top_n 3", "/v1/rerank", json!({"query": QUERY, "documents": DOCS, "top_n": 3})),
        ("top_n 1", "/v2/rerank", json!({"query": QUERY, "documents": DOCS, "top_n": 1})),
        ("top_n 0", "/v1/rerank", json!({"query": QUERY, "documents": DOCS, "top_n": 0})),
        ("top_n > docs", "/v1/rerank", json!({"query": QUERY, "documents": DOCS, "top_n": 20})),
        ("return_documents", "/v1/rerank", json!({"query": QUERY, "documents": DOCS, "return_documents": true})),
        ("return_documents false", "/v2/rerank", json!({"query": QUERY, "documents": DOCS, "return_documents": false})),
        ("with model", "/v1/rerank", json!({"model": "nvidia/llama-nemotron-rerank-1b-v2", "query": QUERY, "documents": DOCS})),
        ("2 docs", "/v1/rerank", json!({"query": QUERY, "documents": &DOCS[..2]})),
        ("single doc", "/v1/rerank", json!({"query": QUERY, "documents": [DOCS[1]]})),
    ];
    for (label, path, body) in cases {
        let directs = env.direct_all(ServiceKind::Reranker, path, &body).await;
        let (status, gw) = env.via(path, &body).await;
        assert_eq!(status, StatusCode::OK, "{label}: {gw}");
        let direct = &directs[0].1;
        assert_eq!(directs[0].0, StatusCode::OK, "{label}: direct {direct}");
        assert_same_shape(&gw, direct, "results", label);
        assert_eq!(gw["model"], direct["model"], "{label}: model");
        // 後端本身不是確定性的：同一台、同一個 request 連打兩次，中段分數（約 0.5）可差到 0.012，
        // 批次組成不同也會浮動。因此逐份文件比對分數（容忍 0.05），只在差距明顯時要求名次一致。
        const TOL: f64 = 0.05;
        let score_of = |d: &Value, index: u64| {
            d["results"].as_array().unwrap().iter().find(|r| r["index"] == json!(index)).map(|r| r["relevance_score"].as_f64().unwrap())
        };
        let results = gw["results"].as_array().unwrap();
        assert_eq!(results.len(), direct["results"].as_array().unwrap().len(), "{label}: result count");
        let gws: Vec<f64> = results.iter().map(|r| r["relevance_score"].as_f64().unwrap()).collect();
        assert!(gws.windows(2).all(|w| w[0] >= w[1]), "{label}: scores sorted");
        for r in results {
            let (index, s) = (r["index"].as_u64().unwrap(), r["relevance_score"].as_f64().unwrap());
            let close = directs.iter().any(|(_, d)| score_of(d, index).is_some_and(|ds| (ds - s).abs() < TOL));
            assert!(close, "{label}: document {index} score {s} far from direct scores");
            // index 重新對應後，document 必須是原本那一份
            if let Some(text) = r["document"]["text"].as_str() {
                let docs = body["documents"].as_array().unwrap();
                assert_eq!(text, docs[index as usize].as_str().unwrap(), "{label}: document at index");
            }
        }
        // 名次：直接打後端時分數差距明顯（> TOL）的兩份文件，在 gateway 結果中的先後必須相同
        let direct_ranked = result_indexes(direct);
        let gw_pos = |index: u64| result_indexes(&gw).iter().position(|x| *x == index);
        for (a, b) in direct_ranked.iter().zip(direct_ranked.iter().skip(1)) {
            let (sa, sb) = (score_of(direct, *a).unwrap(), score_of(direct, *b).unwrap());
            if sa - sb > TOL
                && let (Some(pa), Some(pb)) = (gw_pos(*a), gw_pos(*b))
            {
                assert!(pa < pb, "{label}: document {a} ({sa}) should rank above {b} ({sb})");
            }
        }
        // 拆分後 query 在每份都算一次，usage 可能大於單次呼叫
        let (g, d) = (gw["usage"]["prompt_tokens"].as_u64().unwrap(), direct["usage"]["prompt_tokens"].as_u64().unwrap());
        assert!(g >= d, "{label}: usage {g} < {d}");
        println!("ok  rerank     {label:<24} ranking={:?} usage gw={g} direct={d}", result_indexes(&gw));
    }

    let after = env.requests(ServiceKind::Reranker);
    let used: Vec<u64> = after.iter().zip(&before).map(|(a, b)| a - b).collect();
    assert!(used.iter().all(|n| *n > 0), "both backends should get traffic: {used:?}");
    println!("ok  rerank     traffic per backend {used:?}");

    let errors = [
        ("empty documents", json!({"query": QUERY, "documents": []})),
        ("missing query", json!({"documents": DOCS})),
        ("object documents", json!({"query": QUERY, "documents": [{"text": "a"}, {"text": "b"}]})),
        ("unknown model", json!({"model": "nope", "query": QUERY, "documents": DOCS})),
    ];
    for (label, body) in errors {
        let (dstatus, direct) = env.post(&env.backends(ServiceKind::Reranker)[0], "/v1/rerank", &body).await;
        let (status, gw) = env.via("/v1/rerank", &body).await;
        assert_eq!(status, dstatus, "{label}: {gw}");
        assert_eq!(keys(&gw), keys(&direct), "{label}: error keys");
        println!("ok  rerank     error {label:<18} HTTP {}", status.as_u16());
    }
}

#[tokio::test]
#[ignore]
async fn decision_requests() {
    let env = Env::start().await;
    let path = "/v1/systemone";
    let before = env.requests(ServiceKind::Decision);

    let cases = [
        (
            "all question types",
            json!({"state": "Checkout has been failing for every customer for the last hour.",
                "questions": {
                    "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
                    "team": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"billing": "Payments", "technical": "Outages"}},
                    "severity": {"type": "score", "instructions": "How severe is the impact?",
                                 "criteria": ["No impact", "Minor", "Major", "Critical"]}}}),
        ),
        (
            "json state + noul criteria",
            json!({"state": {"user": "bob", "message": "I want my money back"},
                "questions": {"refund": {"type": "noul", "instructions": "Is this a refund request?",
                                         "criteria": {"true": "asks for money back", "false": "anything else"}}}}),
        ),
        (
            "array state",
            json!({"state": [{"role": "user", "content": "my order never arrived"}],
                "questions": {"shipping": {"type": "noul", "instructions": "Is this about shipping?"}}}),
        ),
        (
            "with model",
            json!({"model": "clef-flash", "state": "hello", "questions": {"greeting": {"type": "noul", "instructions": "Is this a greeting?"}}}),
        ),
    ];
    for (label, body) in cases {
        let directs = env.direct_all(ServiceKind::Decision, path, &body).await;
        // 打兩次，確認兩台都會被用到
        for _ in 0..2 {
            let resp = env
                .http
                .post(format!("{}{path}", env.gateway))
                .header("x-request-id", "real-test")
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{label}");
            assert_eq!(resp.headers()["x-request-id"], "real-test");
            assert!(resp.headers()["content-type"].to_str().unwrap().starts_with("application/json"));
            let gw: Value = resp.json().await.unwrap();
            let direct = &directs[0].1;
            assert_eq!(keys(&gw), keys(direct), "{label}: top-level keys");
            assert_eq!(keys(&gw["answers"]), keys(&direct["answers"]), "{label}: answer names");
            for (name, a) in gw["answers"].as_object().unwrap() {
                assert_eq!(keys(a), keys(&direct["answers"][name]), "{label}: {name} keys");
                assert_eq!(a["type"], direct["answers"][name]["type"]);
            }
            assert_eq!(gw["usage"], direct["usage"], "{label}: usage");
        }
        println!("ok  decision   {label:<28} answers={:?}", keys(&directs[0].1["answers"]));
    }

    let after = env.requests(ServiceKind::Decision);
    let used: Vec<u64> = after.iter().zip(&before).map(|(a, b)| a - b).collect();
    assert_eq!(used.iter().sum::<u64>(), 8, "every request reaches a backend");
    assert!(used.iter().all(|n| *n > 0), "both backends should get traffic: {used:?}");
    println!("ok  decision   traffic per backend {used:?}");

    // 權重分配：依 config.yaml 的 weight 比例（12371:12372 = 45:55）
    let weights: Vec<u64> = env.app.pool.load().decision.backends.iter().map(|b| b.weight as u64).collect();
    let total: u64 = weights.iter().sum();
    let small = json!({"state": "hi", "questions": {"q": {"type": "noul", "instructions": "Is this a greeting?"}}});
    let before = env.requests(ServiceKind::Decision);
    for _ in 0..total {
        let (status, _) = env.via(path, &small).await;
        assert_eq!(status, StatusCode::OK);
    }
    let used: Vec<u64> = env.requests(ServiceKind::Decision).iter().zip(&before).map(|(a, b)| a - b).collect();
    assert_eq!(used, weights, "weighted distribution over {total} requests");
    println!("ok  decision   weighted distribution {used:?} for weights {weights:?}");

    let errors = [
        ("bad question type", json!({"state": "x", "questions": {"q": {"type": "bogus", "instructions": "?"}}})),
        ("missing questions", json!({"state": "x"})),
        ("choice with one option", json!({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": "only"}}}})),
    ];
    for (label, body) in errors {
        let (dstatus, direct) = env.post(&env.backends(ServiceKind::Decision)[0], path, &body).await;
        let (status, gw) = env.via(path, &body).await;
        assert_eq!(status, dstatus, "{label}: {gw}");
        assert_eq!(gw, direct, "{label}: body passed through unchanged");
        println!("ok  decision   error {label:<24} HTTP {}", status.as_u16());
    }
}
