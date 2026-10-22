//! 啟動 / reload 時的後端探測與建立後端池（SPEC §7.1、§8.2）。

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::backend::{Backend, Pool, ServicePool, unix_now};
use crate::config::{Config, ServiceKind};
use crate::upstream::classify;

fn probe_request(kind: ServiceKind, model: Option<&str>) -> (&'static str, Value, &'static str) {
    let (path, mut body, key) = match kind {
        ServiceKind::Embedding => ("/v1/embeddings", json!({"input": "ping"}), "data"),
        ServiceKind::Reranker => ("/v1/rerank", json!({"query": "ping", "documents": ["ping"]}), "results"),
        ServiceKind::Decision => (
            "/v1/systemone",
            json!({"state": "ping", "questions": {"probe": {"type": "noul", "instructions": "Is this a test?"}}}),
            "answers",
        ),
    };
    if let Some(m) = model {
        body["model"] = Value::from(m);
    }
    (path, body, key)
}

/// 對後端實際打一次該服務的 API；回 2xx 且回應含預期欄位才算通過。
pub async fn probe_backend(client: &reqwest::Client, kind: ServiceKind, url: &str, model: Option<&str>) -> Result<(), String> {
    let (path, body, key) = probe_request(kind, model);
    let resp = client.post(format!("{url}{path}")).json(&body).send().await.map_err(|e| reason(&e))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("HTTP {}", status.as_u16()));
    }
    let value: Value = resp.json().await.map_err(|e| {
        if e.is_timeout() { "timeout".to_string() } else { "invalid response: not JSON".to_string() }
    })?;
    if value.get(key).is_none() {
        return Err(format!("invalid response: missing \"{key}\""));
    }
    Ok(())
}

fn reason(e: &reqwest::Error) -> String {
    match classify(e) {
        crate::upstream::CallError::Timeout => "timeout".to_string(),
        crate::upstream::CallError::Connect(m) | crate::upstream::CallError::Other(m) => m,
        other => format!("{other:?}"),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Changes {
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisabledBackend {
    pub service: String,
    pub url: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadReport {
    pub version: u64,
    pub config_path: String,
    pub changes: BTreeMap<String, Changes>,
    pub disabled: Vec<DisabledBackend>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// 並行探測所有後端並建立新的後端池；連不上的後端標記為停用。
/// `prev` 為目前生效的池：用來計算差異，並讓 URL 相同的後端沿用統計數字。
pub async fn build_pool(config: Config, version: u64, prev: Option<&Pool>) -> (Pool, LoadReport) {
    let probe_client = reqwest::Client::builder()
        .timeout(Duration::from_millis(config.probe.timeout_ms))
        .build()
        .expect("reqwest client");
    let prev_stats = prev.map(|p| p.stats_by_url()).unwrap_or_default();

    let probes = ServiceKind::ALL.iter().flat_map(|kind| {
        let svc = config.service(*kind);
        let client = &probe_client;
        svc.urls().map(move |url| async move {
            let result = probe_backend(client, *kind, url, svc.probe_model.as_deref()).await;
            (*kind, url.to_string(), result.err())
        })
    });
    let results = futures::future::join_all(probes).await;

    let mut disabled = Vec::new();
    let mut changes = BTreeMap::new();
    let mut services = Vec::new();
    for kind in ServiceKind::ALL {
        let svc = config.service(kind);
        let mut make = |url: &str, weight: u32| {
            let failure = results.iter().find(|(k, u, _)| *k == kind && u == url).and_then(|(_, _, e)| e.clone());
            if let Some(reason) = &failure {
                tracing::warn!(service = %kind, backend = %url, reason, "backend disabled: probe failed");
                disabled.push(DisabledBackend { service: kind.to_string(), url: url.to_string(), reason: reason.clone() });
            }
            let stats = prev_stats.get(&(kind, url.to_string())).cloned().unwrap_or_default();
            Backend::new(kind, url.to_string(), weight, failure, stats)
        };
        let mut backends: Vec<Arc<Backend>> = svc.backends.iter().map(|b| Arc::new(make(&b.url, b.weight))).collect();
        if let Some(f) = &svc.fallback {
            backends.push(Arc::new(make(&f.url, 1).into_fallback(f.request_timeout_ms)));
        }
        if backends.iter().all(|b| b.disabled.is_some()) {
            tracing::error!(service = %kind, "all backends disabled; requests will get 503");
        }
        let old: Vec<String> = prev.map(|p| p.service(kind).backends.iter().map(|b| b.url.clone()).collect()).unwrap_or_default();
        let new: Vec<String> = svc.urls().map(String::from).collect();
        changes.insert(
            kind.to_string(),
            Changes {
                added: if prev.is_some() { new.iter().filter(|u| !old.contains(u)).cloned().collect() } else { new.clone() },
                removed: old.iter().filter(|u| !new.contains(u)).cloned().collect(),
            },
        );
        services.push(ServicePool::new(kind, svc, backends));
    }

    let report = LoadReport { version, config_path: config.path.display().to_string(), changes, disabled, warnings: Vec::new() };
    let mut services = services.into_iter();
    let pool = Pool {
        version,
        loaded_at_unix: unix_now(),
        config: Arc::new(config),
        embedding: services.next().unwrap(),
        reranker: services.next().unwrap(),
        decision: services.next().unwrap(),
    };
    (pool, report)
}
