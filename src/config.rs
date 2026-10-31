//! 設定檔 `config.yaml` 的載入與驗證（SPEC §2）。

use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceKind {
    Embedding,
    Reranker,
    Decision,
}

impl ServiceKind {
    pub const ALL: [ServiceKind; 3] = [ServiceKind::Embedding, ServiceKind::Reranker, ServiceKind::Decision];

    pub fn as_str(self) -> &'static str {
        match self {
            ServiceKind::Embedding => "embedding",
            ServiceKind::Reranker => "reranker",
            ServiceKind::Decision => "decision",
        }
    }
}

impl fmt::Display for ServiceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Config {
    pub path: PathBuf,
    pub server: ServerConfig,
    pub probe: ProbeConfig,
    pub embedding: ServiceConfig,
    pub reranker: ServiceConfig,
    pub decision: ServiceConfig,
    /// 以 decision 後端提供 rerank（SPEC §6.4）；`None` 表示未啟用。
    pub decision_reranker: Option<DecisionRerankerConfig>,
}

impl Config {
    pub fn service(&self, kind: ServiceKind) -> &ServiceConfig {
        match kind {
            ServiceKind::Embedding => &self.embedding,
            ServiceKind::Reranker => &self.reranker,
            ServiceKind::Decision => &self.decision,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub listen: SocketAddr,
    #[serde(default)]
    pub admin_listen: Option<SocketAddr>,
    pub control_socket: PathBuf,
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout_secs: u64,
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeConfig {
    #[serde(default = "default_probe_timeout")]
    pub timeout_ms: u64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        ProbeConfig { timeout_ms: default_probe_timeout() }
    }
}

/// 每個 service 生效的調校參數（`defaults` 套上 service 層級的覆寫後的結果）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tuning {
    pub connect_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub max_retries: u32,
    pub unhealthy_threshold: u32,
    pub unhealthy_cooldown_secs: u64,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            connect_timeout_ms: 1000,
            request_timeout_ms: 30000,
            max_retries: 1,
            unhealthy_threshold: 3,
            unhealthy_cooldown_secs: 600,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ServiceConfig {
    pub tuning: Tuning,
    pub probe_model: Option<String>,
    pub backends: Vec<BackendConfig>,
    /// 所有 `backends` 都失效時才使用的後端（SPEC §8.6）。
    pub fallback: Option<FallbackConfig>,
}

impl ServiceConfig {
    /// 所有後端 URL，fallback 排在最後。
    pub fn urls(&self) -> impl Iterator<Item = &str> {
        self.backends.iter().map(|b| b.url.as_str()).chain(self.fallback.iter().map(|f| f.url.as_str()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    pub url: String,
    #[serde(default = "default_weight")]
    pub weight: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FallbackConfig {
    pub url: String,
    /// 覆寫 service 的 `request_timeout_ms`；CPU 推論較慢，通常要設得較長。
    #[serde(default)]
    pub request_timeout_ms: Option<u64>,
}

/// `POST /decision/v2/rerank` 的設定：沒有自己的後端，共用 `services.decision` 的後端池。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRerankerConfig {
    /// 每個 (query, document) 配對的 state；`{query}`、`{document}` 會被替換。
    #[serde(default = "default_state_template")]
    pub state_template: String,
    /// 是非題的題目，「是」的機率即相關度；client 可用非標準欄位 `instructions` 覆寫。
    #[serde(default = "default_relevance_instructions")]
    pub instructions: String,
    /// 送給 decision 後端的 `model` 欄位；`None` = 不帶。
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_max_documents")]
    pub max_documents: usize,
    /// 單一 rerank request 同時送出的 decision 呼叫上限；`None` = 不限制。
    #[serde(default)]
    pub max_concurrency: Option<usize>,
    /// 放進 state 前把 document 截斷到這個字元數；`None` = 不截斷。
    #[serde(default)]
    pub max_document_chars: Option<usize>,
    /// 一般後端失敗時，是否把整個 request 改送 `services.decision.fallback`。
    #[serde(default = "default_true")]
    pub use_fallback: bool,
}

impl DecisionRerankerConfig {
    fn validate(&self) -> anyhow::Result<()> {
        if !self.state_template.contains("{document}") {
            bail!("services.decision_reranker.state_template must contain {{document}}");
        }
        if self.instructions.trim().is_empty() {
            bail!("services.decision_reranker.instructions must not be empty");
        }
        if self.max_documents == 0 || self.max_concurrency == Some(0) || self.max_document_chars == Some(0) {
            bail!("services.decision_reranker: max_documents / max_concurrency / max_document_chars must be at least 1");
        }
        Ok(())
    }
}

fn default_state_template() -> String {
    "Query: {query}\n\nDocument: {document}".to_string()
}
fn default_relevance_instructions() -> String {
    "Does the document contain information that helps answer the query?".to_string()
}
fn default_max_documents() -> usize {
    1000
}
fn default_true() -> bool {
    true
}

fn default_shutdown_timeout() -> u64 {
    60
}
fn default_max_body_bytes() -> usize {
    32 * 1024 * 1024
}
fn default_probe_timeout() -> u64 {
    10000
}
fn default_weight() -> u32 {
    1
}

// ---- 檔案格式 ----

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    server: ServerConfig,
    #[serde(default)]
    defaults: RawTuning,
    #[serde(default)]
    probe: ProbeConfig,
    services: RawServices,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawServices {
    embedding: RawService,
    reranker: RawService,
    decision: RawService,
    #[serde(default)]
    decision_reranker: Option<DecisionRerankerConfig>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawTuning {
    connect_timeout_ms: Option<u64>,
    request_timeout_ms: Option<u64>,
    max_retries: Option<u32>,
    unhealthy_threshold: Option<u32>,
    unhealthy_cooldown_secs: Option<u64>,
}

impl RawTuning {
    fn apply(&self, base: &Tuning) -> Tuning {
        Tuning {
            connect_timeout_ms: self.connect_timeout_ms.unwrap_or(base.connect_timeout_ms),
            request_timeout_ms: self.request_timeout_ms.unwrap_or(base.request_timeout_ms),
            max_retries: self.max_retries.unwrap_or(base.max_retries),
            unhealthy_threshold: self.unhealthy_threshold.unwrap_or(base.unhealthy_threshold),
            unhealthy_cooldown_secs: self.unhealthy_cooldown_secs.unwrap_or(base.unhealthy_cooldown_secs),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawService {
    #[serde(default)]
    probe_model: Option<String>,
    backends: Vec<BackendConfig>,
    #[serde(default)]
    fallback: Option<FallbackConfig>,
    // service 層級可覆寫 defaults 的任一欄位
    connect_timeout_ms: Option<u64>,
    request_timeout_ms: Option<u64>,
    max_retries: Option<u32>,
    unhealthy_threshold: Option<u32>,
    unhealthy_cooldown_secs: Option<u64>,
}

impl RawService {
    fn into_config(self, kind: ServiceKind, defaults: &Tuning) -> anyhow::Result<ServiceConfig> {
        let overrides = RawTuning {
            connect_timeout_ms: self.connect_timeout_ms,
            request_timeout_ms: self.request_timeout_ms,
            max_retries: self.max_retries,
            unhealthy_threshold: self.unhealthy_threshold,
            unhealthy_cooldown_secs: self.unhealthy_cooldown_secs,
        };
        let tuning = overrides.apply(defaults);
        if tuning.unhealthy_threshold == 0 {
            bail!("services.{kind}.unhealthy_threshold must be at least 1");
        }
        if self.backends.is_empty() {
            bail!("services.{kind}.backends must contain at least one backend");
        }
        let mut backends = Vec::with_capacity(self.backends.len());
        for b in self.backends {
            let url = normalize_url(&b.url).with_context(|| format!("services.{kind}.backends: invalid url {:?}", b.url))?;
            if b.weight == 0 {
                bail!("services.{kind}.backends: weight of {url} must be at least 1");
            }
            if backends.iter().any(|x: &BackendConfig| x.url == url) {
                bail!("services.{kind}.backends: duplicate url {url}");
            }
            backends.push(BackendConfig { url, weight: b.weight });
        }
        let fallback = match self.fallback {
            Some(f) => {
                let url = normalize_url(&f.url).with_context(|| format!("services.{kind}.fallback: invalid url {:?}", f.url))?;
                if backends.iter().any(|b| b.url == url) {
                    bail!("services.{kind}.fallback: {url} is also listed in backends");
                }
                Some(FallbackConfig { url, request_timeout_ms: f.request_timeout_ms })
            }
            None => None,
        };
        Ok(ServiceConfig { tuning, probe_model: self.probe_model, backends, fallback })
    }
}

fn normalize_url(raw: &str) -> anyhow::Result<String> {
    let parsed = url::Url::parse(raw.trim())?;
    // 沒有編入 TLS，只支援 http 後端
    if parsed.scheme() != "http" {
        bail!("only http:// backends are supported");
    }
    if parsed.host_str().is_none() {
        bail!("missing host");
    }
    Ok(parsed.as_str().trim_end_matches('/').to_string())
}

/// 相對路徑以設定檔所在目錄為基準。
fn resolve_relative(base_dir: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() { p.to_path_buf() } else { base_dir.join(p) }
}

pub fn load(path: &Path) -> anyhow::Result<Config> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    parse(&text, path)
}

pub fn parse(text: &str, path: &Path) -> anyhow::Result<Config> {
    let raw: RawConfig = serde_yaml::from_str(text).with_context(|| format!("invalid config {}", path.display()))?;
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let base_dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let defaults = raw.defaults.apply(&Tuning::default());
    if let Some(dr) = &raw.services.decision_reranker {
        dr.validate()?;
    }
    let mut server = raw.server;
    server.control_socket = resolve_relative(&base_dir, &server.control_socket);
    Ok(Config {
        server,
        probe: raw.probe,
        embedding: raw.services.embedding.into_config(ServiceKind::Embedding, &defaults)?,
        reranker: raw.services.reranker.into_config(ServiceKind::Reranker, &defaults)?,
        decision: raw.services.decision.into_config(ServiceKind::Decision, &defaults)?,
        decision_reranker: raw.services.decision_reranker,
        path,
    })
}

/// 只讀出 `server.control_socket`，供 `reload` / `status` 找到執行中的 gateway。
/// 刻意不做完整驗證：設定檔有錯時仍要能送出 reload，讓 gateway 回報錯誤。
pub fn read_control_socket(path: &Path) -> anyhow::Result<PathBuf> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let value: serde_yaml::Value = serde_yaml::from_str(&text).with_context(|| format!("invalid YAML {}", path.display()))?;
    let socket = value
        .get("server")
        .and_then(|s| s.get("control_socket"))
        .and_then(|s| s.as_str())
        .with_context(|| format!("server.control_socket not found in {}", path.display()))?;
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let base_dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    Ok(resolve_relative(&base_dir, Path::new(socket)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
server:
  listen: "127.0.0.1:8080"
  control_socket: "run/control.sock"
defaults:
  max_retries: 2
services:
  embedding:
    backends:
      - url: "http://10.0.0.1:1/"
      - url: "http://10.0.0.2:1"
        weight: 3
  reranker:
    request_timeout_ms: 5000
    backends: [{url: "http://10.0.0.3:1"}]
  decision:
    probe_model: clef-flash
    backends: [{url: "http://10.0.0.4:1"}]
"#;

    #[test]
    fn parses_and_applies_overrides() {
        let c = parse(SAMPLE, Path::new("/etc/gw/config.yaml")).unwrap();
        assert_eq!(c.server.control_socket, PathBuf::from("/etc/gw/run/control.sock"));
        assert_eq!(c.embedding.backends[0].url, "http://10.0.0.1:1");
        assert_eq!(c.embedding.backends[1].weight, 3);
        assert_eq!(c.embedding.tuning.max_retries, 2);
        assert_eq!(c.reranker.tuning.request_timeout_ms, 5000);
        assert_eq!(c.reranker.tuning.max_retries, 2);
        assert_eq!(c.embedding.tuning.unhealthy_threshold, 3);
        assert_eq!(c.embedding.tuning.unhealthy_cooldown_secs, 600);
        assert_eq!(c.decision.probe_model.as_deref(), Some("clef-flash"));
        assert!(c.embedding.fallback.is_none());

        let with_fallback = SAMPLE.replace(
            "        weight: 3\n",
            "        weight: 3\n    fallback:\n      url: \"http://127.0.0.1:9/\"\n      request_timeout_ms: 120000\n",
        );
        let c = parse(&with_fallback, Path::new("c.yaml")).unwrap();
        let f = c.embedding.fallback.unwrap();
        assert_eq!((f.url.as_str(), f.request_timeout_ms), ("http://127.0.0.1:9", Some(120000)));
    }

    #[test]
    fn parses_decision_reranker() {
        assert!(parse(SAMPLE, Path::new("c.yaml")).unwrap().decision_reranker.is_none());
        let c = parse(&format!("{SAMPLE}  decision_reranker: {{}}\n"), Path::new("c.yaml")).unwrap();
        let dr = c.decision_reranker.unwrap();
        assert_eq!(dr.state_template, "Query: {query}\n\nDocument: {document}");
        assert_eq!((dr.max_documents, dr.max_concurrency, dr.use_fallback), (1000, None, true));
        let c = parse(&format!("{SAMPLE}  decision_reranker:\n    max_concurrency: 4\n    use_fallback: false\n"), Path::new("c.yaml"))
            .unwrap();
        let dr = c.decision_reranker.unwrap();
        assert_eq!((dr.max_concurrency, dr.use_fallback), (Some(4), false));
        for bad in ["state_template: \"{query}\"", "max_concurrency: 0", "instructions: \" \"", "unknown: 1"] {
            assert!(parse(&format!("{SAMPLE}  decision_reranker:\n    {bad}\n"), Path::new("c.yaml")).is_err(), "{bad}");
        }
    }

    #[test]
    fn rejects_bad_configs() {
        let dup = SAMPLE.replace("http://10.0.0.2:1", "http://10.0.0.1:1");
        assert!(parse(&dup, Path::new("c.yaml")).unwrap_err().to_string().contains("duplicate"));
        let empty = SAMPLE.replace(r#"backends: [{url: "http://10.0.0.3:1"}]"#, "backends: []");
        assert!(parse(&empty, Path::new("c.yaml")).unwrap_err().to_string().contains("at least one"));
        let bad_url = SAMPLE.replace("http://10.0.0.3:1", "ftp://x");
        assert!(parse(&bad_url, Path::new("c.yaml")).is_err());
        let https = SAMPLE.replace("http://10.0.0.3:1", "https://10.0.0.3:1");
        assert!(format!("{:#}", parse(&https, Path::new("c.yaml")).unwrap_err()).contains("only http://"));
        let unknown = SAMPLE.replace("max_retries: 2", "max_retry: 2");
        assert!(parse(&unknown, Path::new("c.yaml")).is_err());
        let dup_fallback = SAMPLE.replace("        weight: 3\n", "        weight: 3\n    fallback: {url: \"http://10.0.0.1:1\"}\n");
        assert!(parse(&dup_fallback, Path::new("c.yaml")).unwrap_err().to_string().contains("also listed"));
        let rerank_fallback = SAMPLE.replace("    request_timeout_ms: 5000\n", "    request_timeout_ms: 5000\n    fallback: {url: \"http://10.0.0.9:1\"}\n");
        assert_eq!(parse(&rerank_fallback, Path::new("c.yaml")).unwrap().reranker.fallback.unwrap().url, "http://10.0.0.9:1");
    }
}
