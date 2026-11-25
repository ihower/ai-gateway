//! 設定檔 `config.yaml` 的載入與驗證（SPEC §2）。

use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceKind {
    Embedding,
    Reranker,
    Decision,
    /// OpenAI `/v1/chat/completions`；選填，未設定時沒有後端。
    Chat,
}

impl ServiceKind {
    pub const ALL: [ServiceKind; 4] = [ServiceKind::Embedding, ServiceKind::Reranker, ServiceKind::Decision, ServiceKind::Chat];

    pub fn as_str(self) -> &'static str {
        match self {
            ServiceKind::Embedding => "embedding",
            ServiceKind::Reranker => "reranker",
            ServiceKind::Decision => "decision",
            ServiceKind::Chat => "chat",
        }
    }

    /// 是否有「處理筆數」（`items`）的概念；decision 與 chat 整包轉發，沒有。
    pub fn counts_items(self) -> bool {
        matches!(self, ServiceKind::Embedding | ServiceKind::Reranker)
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
    /// 選填；未設定時 `backends` 為空（見 `ServiceConfig::is_configured`）。
    pub chat: ServiceConfig,
    /// 以 decision 後端提供 rerank（SPEC §6.4）；`None` 表示未啟用。
    pub decision_reranker: Option<DecisionRerankerConfig>,
}

impl Config {
    pub fn service(&self, kind: ServiceKind) -> &ServiceConfig {
        match kind {
            ServiceKind::Embedding => &self.embedding,
            ServiceKind::Reranker => &self.reranker,
            ServiceKind::Decision => &self.decision,
            ServiceKind::Chat => &self.chat,
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
    /// 長文件切段（SPEC §5.4）；只有 reranker 可設定。
    pub chunking: Option<ChunkingConfig>,
}

impl ServiceConfig {
    /// 未設定的選填 service（目前只有 chat）。
    fn unconfigured(defaults: &Tuning) -> Self {
        ServiceConfig { tuning: defaults.clone(), probe_model: None, backends: Vec::new(), fallback: None, chunking: None }
    }

    /// 設定檔中有這個 service（至少一個後端）。
    pub fn is_configured(&self) -> bool {
        !self.backends.is_empty()
    }

    /// 所有後端 URL，fallback 排在最後。
    pub fn urls(&self) -> impl Iterator<Item = &str> {
        self.backends.iter().map(|b| b.url.as_str()).chain(self.fallback.iter().map(|f| f.url.as_str()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    pub url: String,
    /// 顯示用名稱（status、`/admin/backends`）；未填則用 url。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default = "default_weight")]
    pub weight: u32,
    /// 可接受的輸入字元數上限（request 中所有 JSON 字串值的字元數總和）；超過時不挑這個後端。
    /// `None` = 不限制。只有 decision 與 chat 可設定。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_chars: Option<usize>,
}

impl BackendConfig {
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.url)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FallbackConfig {
    pub url: String,
    /// 顯示用名稱；未填則用 url。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 覆寫 service 的 `request_timeout_ms`；CPU 推論較慢，通常要設得較長。
    #[serde(default)]
    pub request_timeout_ms: Option<u64>,
    /// 同 `BackendConfig::max_input_chars`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_chars: Option<usize>,
}

/// rerank 的長文件切段：超過 `max_chars` 個字元的 document 切成多段分別算分，取最高分。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkingConfig {
    pub max_chars: usize,
    #[serde(default)]
    pub overlap_chars: usize,
}

impl ChunkingConfig {
    fn validate(&self, section: &str) -> anyhow::Result<()> {
        if self.max_chars == 0 || self.overlap_chars >= self.max_chars {
            bail!("{section}.chunking: max_chars must be at least 1 and overlap_chars must be less than max_chars");
        }
        Ok(())
    }
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
    /// 放進 state 前把 document 截斷到這個字元數；`None` = 不截斷。與 `chunking` 擇一。
    #[serde(default)]
    pub max_document_chars: Option<usize>,
    /// 一般後端失敗時，是否把整個 request 改送 `services.decision.fallback`。
    #[serde(default = "default_true")]
    pub use_fallback: bool,
    /// 長文件切段；`None` = 不切。與 `max_document_chars` 擇一。
    #[serde(default)]
    pub chunking: Option<ChunkingConfig>,
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
        if self.max_document_chars.is_some() && self.chunking.is_some() {
            bail!("services.decision_reranker: max_document_chars and chunking are mutually exclusive");
        }
        if let Some(c) = &self.chunking {
            c.validate("services.decision_reranker")?;
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
    chat: Option<RawService>,
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
    #[serde(default)]
    chunking: Option<ChunkingConfig>,
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
            let name = check_name(kind, b.name, &url, &backends)?;
            check_max_input_chars(kind, &url, b.max_input_chars)?;
            backends.push(BackendConfig { url, name, weight: b.weight, max_input_chars: b.max_input_chars });
        }
        let fallback = match self.fallback {
            Some(f) => {
                let url = normalize_url(&f.url).with_context(|| format!("services.{kind}.fallback: invalid url {:?}", f.url))?;
                if backends.iter().any(|b| b.url == url) {
                    bail!("services.{kind}.fallback: {url} is also listed in backends");
                }
                let name = check_name(kind, f.name, &url, &backends)?;
                check_max_input_chars(kind, &url, f.max_input_chars)?;
                Some(FallbackConfig { url, name, request_timeout_ms: f.request_timeout_ms, max_input_chars: f.max_input_chars })
            }
            None => None,
        };
        if let Some(c) = &self.chunking {
            if kind != ServiceKind::Reranker {
                bail!("services.{kind}.chunking is only supported for reranker");
            }
            c.validate(&format!("services.{kind}"))?;
        }
        Ok(ServiceConfig { tuning, probe_model: self.probe_model, backends, fallback, chunking: self.chunking })
    }
}

/// `max_input_chars` 只對整包轉發的 decision / chat 有意義；embedding / reranker 由 gateway 拆分，另有 chunking。
fn check_max_input_chars(kind: ServiceKind, url: &str, limit: Option<usize>) -> anyhow::Result<()> {
    match limit {
        None => Ok(()),
        Some(_) if !matches!(kind, ServiceKind::Decision | ServiceKind::Chat) => {
            bail!("services.{kind}: max_input_chars is only supported for decision and chat")
        }
        Some(0) => bail!("services.{kind}: max_input_chars of {url} must be at least 1"),
        Some(_) => Ok(()),
    }
}

/// 驗證選填的 `name`：去除前後空白、不可為空字串，同一 service 內不可與其他後端的顯示名稱重複。
fn check_name(kind: ServiceKind, name: Option<String>, url: &str, others: &[BackendConfig]) -> anyhow::Result<Option<String>> {
    let name = match name {
        Some(n) if n.trim().is_empty() => bail!("services.{kind}: name of {url} must not be empty"),
        Some(n) => Some(n.trim().to_string()),
        None => None,
    };
    let display = name.as_deref().unwrap_or(url);
    if others.iter().any(|b| b.display_name() == display) {
        bail!("services.{kind}: duplicate backend name {display:?}");
    }
    Ok(name)
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
        chat: match raw.services.chat {
            Some(c) => c.into_config(ServiceKind::Chat, &defaults)?,
            None => ServiceConfig::unconfigured(&defaults),
        },
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
        let c = parse(&format!("{SAMPLE}  decision_reranker:\n    chunking: {{max_chars: 10, overlap_chars: 2}}\n"), Path::new("c.yaml"))
            .unwrap();
        assert_eq!(c.decision_reranker.unwrap().chunking, Some(ChunkingConfig { max_chars: 10, overlap_chars: 2 }));
        for bad in [
            "state_template: \"{query}\"",
            "max_concurrency: 0",
            "instructions: \" \"",
            "unknown: 1",
            "chunking: {max_chars: 10, overlap_chars: 10}",
            "chunking: {max_chars: 10}\n    max_document_chars: 100",
        ] {
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

    #[test]
    fn parses_max_input_chars() {
        let dec = r#"backends: [{url: "http://10.0.0.4:1"}]"#;
        let with = |b: &str| SAMPLE.replace(dec, b);
        let c = parse(&with(r#"backends: [{url: "http://10.0.0.4:1", max_input_chars: 7000}]
    fallback: {url: "http://10.0.0.5:1", max_input_chars: 100}"#), Path::new("c.yaml")).unwrap();
        assert_eq!(c.decision.backends[0].max_input_chars, Some(7000));
        assert_eq!(c.decision.fallback.unwrap().max_input_chars, Some(100));
        assert_eq!(parse(SAMPLE, Path::new("c.yaml")).unwrap().decision.backends[0].max_input_chars, None);
        let zero = with(r#"backends: [{url: "http://10.0.0.4:1", max_input_chars: 0}]"#);
        assert!(parse(&zero, Path::new("c.yaml")).unwrap_err().to_string().contains("at least 1"));
        let rr = SAMPLE.replace(r#"[{url: "http://10.0.0.3:1"}]"#, r#"[{url: "http://10.0.0.3:1", max_input_chars: 10}]"#);
        assert!(parse(&rr, Path::new("c.yaml")).unwrap_err().to_string().contains("only supported for decision and chat"));
    }

    #[test]
    fn chat_is_optional() {
        let c = parse(SAMPLE, Path::new("c.yaml")).unwrap();
        assert!(!c.chat.is_configured());
        assert_eq!(c.chat.urls().count(), 0);
        let with_chat = format!("{SAMPLE}  chat:\n    probe_model: qwen\n    request_timeout_ms: 600000\n    backends: [{{url: \"http://10.0.0.5:1/\"}}]\n");
        let c = parse(&with_chat, Path::new("c.yaml")).unwrap();
        assert!(c.chat.is_configured());
        assert_eq!(c.chat.backends[0].url, "http://10.0.0.5:1");
        assert_eq!(c.chat.probe_model.as_deref(), Some("qwen"));
        assert_eq!((c.chat.tuning.request_timeout_ms, c.chat.tuning.max_retries), (600000, 2));
        let empty = format!("{SAMPLE}  chat:\n    backends: []\n");
        assert!(parse(&empty, Path::new("c.yaml")).unwrap_err().to_string().contains("services.chat.backends"));
    }

    #[test]
    fn parses_reranker_chunking() {
        assert!(parse(SAMPLE, Path::new("c.yaml")).unwrap().reranker.chunking.is_none());
        let with = |service: &str, chunking: &str| {
            let anchor = format!("  {service}:\n");
            SAMPLE.replace(&anchor, &format!("{anchor}    chunking: {chunking}\n"))
        };
        let c = parse(&with("reranker", "{max_chars: 2000}"), Path::new("c.yaml")).unwrap();
        assert_eq!(c.reranker.chunking, Some(ChunkingConfig { max_chars: 2000, overlap_chars: 0 }));
        assert!(parse(&with("reranker", "{max_chars: 0}"), Path::new("c.yaml")).is_err());
        assert!(parse(&with("reranker", "{max_chars: 5, overlap_chars: 6}"), Path::new("c.yaml")).is_err());
        let err = parse(&with("embedding", "{max_chars: 5}"), Path::new("c.yaml")).unwrap_err();
        assert!(err.to_string().contains("only supported for reranker"), "{err}");
    }

    #[test]
    fn backend_name_is_optional_trimmed_and_unique() {
        let named = SAMPLE.replace(
            "      - url: \"http://10.0.0.1:1/\"\n",
            "      - url: \"http://10.0.0.1:1/\"\n        name: \" gpu-1 \"\n",
        );
        let c = parse(&named, Path::new("c.yaml")).unwrap();
        assert_eq!(c.embedding.backends[0].name.as_deref(), Some("gpu-1"));
        assert_eq!(c.embedding.backends[1].name, None);
        assert_eq!(c.embedding.backends[1].display_name(), "http://10.0.0.2:1");

        let empty = SAMPLE.replace("      - url: \"http://10.0.0.1:1/\"\n", "      - url: \"http://10.0.0.1:1/\"\n        name: \"  \"\n");
        assert!(format!("{:#}", parse(&empty, Path::new("c.yaml")).unwrap_err()).contains("must not be empty"));

        let dup = SAMPLE
            .replace("      - url: \"http://10.0.0.1:1/\"\n", "      - url: \"http://10.0.0.1:1/\"\n        name: gpu\n")
            .replace("        weight: 3\n", "        weight: 3\n        name: gpu\n");
        assert!(format!("{:#}", parse(&dup, Path::new("c.yaml")).unwrap_err()).contains("duplicate backend name"));

        let fb = named.replace("        weight: 3\n", "        weight: 3\n    fallback:\n      url: \"http://127.0.0.1:9\"\n      name: cpu\n");
        let c = parse(&fb, Path::new("c.yaml")).unwrap();
        assert_eq!(c.embedding.fallback.unwrap().name.as_deref(), Some("cpu"));
    }
}
