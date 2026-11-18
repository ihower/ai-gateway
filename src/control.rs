//! CLI 與執行中 gateway 之間的 control socket（Unix domain socket，一行 JSON 請求、一行 JSON 回應）。

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::backend::PoolStatus;
use crate::config::ServiceKind;
use crate::probe::LoadReport;
use crate::server::App;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "lowercase")]
pub enum Command {
    Reload,
    Status,
    Reset,
    Shutdown,
    /// 手動停用後端直到下次 reload；`backend` 為 name 或 url，`service` 省略時套用到所有符合的 service。
    Disable {
        backend: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        service: Option<ServiceKind>,
    },
    /// 強制啟用後端（含探測失敗被停用者）直到下次 reload；參數同 `Disable`。
    Enable {
        backend: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        service: Option<ServiceKind>,
    },
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Reply {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<LoadReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<PoolStatus>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// 建立 control socket。若檔案已存在：連得上表示已有 gateway 在跑，拒絕啟動；否則視為殘留檔並移除。
pub fn bind(path: &Path) -> anyhow::Result<UnixListener> {
    if path.exists() {
        if std::os::unix::net::UnixStream::connect(path).is_ok() {
            anyhow::bail!("another ai-gateway is already running on control socket {}", path.display());
        }
        std::fs::remove_file(path).with_context(|| format!("cannot remove stale socket {}", path.display()))?;
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    }
    let listener = UnixListener::bind(path).with_context(|| format!("cannot bind control socket {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

pub async fn serve(listener: UnixListener, app: Arc<App>) {
    loop {
        let Ok((stream, _)) = listener.accept().await else { continue };
        let app = app.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(stream, app).await {
                tracing::warn!(error = %e, "control connection failed");
            }
        });
    }
}

async fn handle(stream: UnixStream, app: Arc<App>) -> anyhow::Result<()> {
    let (read, mut write) = stream.into_split();
    let mut line = String::new();
    BufReader::new(read).read_line(&mut line).await?;
    let reply = match serde_json::from_str::<Command>(&line) {
        Ok(Command::Reload) => match app.reload().await {
            Ok(report) => Reply { ok: true, report: Some(report), ..Default::default() },
            Err(e) => {
                tracing::error!(error = %e, "reload failed; keeping current config");
                Reply { ok: false, error: Some(e), ..Default::default() }
            }
        },
        Ok(Command::Status) => Reply { ok: true, status: Some(app.pool.load().status()), ..Default::default() },
        Ok(Command::Reset) => {
            let pool = app.pool.load();
            pool.reset_stats();
            tracing::info!(version = pool.version, "backend stats reset");
            Reply { ok: true, status: Some(pool.status()), ..Default::default() }
        }
        Ok(Command::Shutdown) => {
            tracing::info!("shutdown requested via control socket");
            app.begin_shutdown();
            Reply { ok: true, ..Default::default() }
        }
        Ok(Command::Disable { backend, service }) => match app.disable_backend(&backend, service).await {
            Ok(o) => Reply { ok: true, messages: o.messages, warnings: o.warnings, ..Default::default() },
            Err(e) => Reply { ok: false, error: Some(e), ..Default::default() },
        },
        Ok(Command::Enable { backend, service }) => match app.enable_backend(&backend, service).await {
            Ok(o) => Reply { ok: true, messages: o.messages, warnings: o.warnings, ..Default::default() },
            Err(e) => Reply { ok: false, error: Some(e), ..Default::default() },
        },
        Err(e) => Reply { ok: false, error: Some(format!("invalid command: {e}")), ..Default::default() },
    };
    let mut out = serde_json::to_vec(&reply)?;
    out.push(b'\n');
    write.write_all(&out).await?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("cannot connect to ai-gateway at {path}: {source}")]
    Connect { path: String, source: std::io::Error },
    #[error("control socket error: {0}")]
    Protocol(String),
}

pub async fn send(path: &Path, cmd: &Command) -> Result<Reply, ClientError> {
    let stream = UnixStream::connect(path)
        .await
        .map_err(|source| ClientError::Connect { path: path.display().to_string(), source })?;
    let (read, mut write) = stream.into_split();
    let mut msg = serde_json::to_vec(cmd).map_err(|e| ClientError::Protocol(e.to_string()))?;
    msg.push(b'\n');
    write.write_all(&msg).await.map_err(|e| ClientError::Protocol(e.to_string()))?;
    let mut line = String::new();
    BufReader::new(read).read_line(&mut line).await.map_err(|e| ClientError::Protocol(e.to_string()))?;
    serde_json::from_str(&line).map_err(|e| ClientError::Protocol(format!("invalid reply: {e}")))
}
