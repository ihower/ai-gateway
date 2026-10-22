//! HTTP server、熱重載與優雅關閉（SPEC §3、§7）。

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use axum::Router;
use axum::extract::{ConnectInfo, DefaultBodyLimit, Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::json;
use tokio::net::{TcpListener, TcpSocket};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{Mutex, watch};

use crate::backend::Pool;
use crate::config::{self, ServerConfig, ServiceKind};
use crate::probe::{LoadReport, build_pool};
use crate::upstream::{CallLog, REQUEST_ID};
use crate::{admin, api, control};

pub struct App {
    pub pool: ArcSwap<Pool>,
    pub draining: AtomicBool,
    pub config_path: PathBuf,
    /// 啟動時的 server 設定；監聽位址等無法熱重載。
    pub boot_server: ServerConfig,
    reload_lock: Mutex<()>,
}

impl App {
    pub fn new(pool: Pool, config_path: PathBuf) -> Self {
        let boot_server = pool.config.server.clone();
        App {
            pool: ArcSwap::from_pointee(pool),
            draining: AtomicBool::new(false),
            config_path,
            boot_server,
            reload_lock: Mutex::new(()),
        }
    }

    /// 重新載入設定並原子替換後端池。進行中的 request 持有舊池的 `Arc`，會用舊池完成。
    /// 設定檔有錯時保留舊設定。
    pub async fn reload(&self) -> Result<LoadReport, String> {
        let _serialized = self.reload_lock.lock().await;
        let config = config::load(&self.config_path).map_err(|e| format!("{e:#}"))?;
        let prev = self.pool.load_full();
        let mut warnings = Vec::new();
        let (a, b) = (&self.boot_server, &config.server);
        if a.listen != b.listen || a.admin_listen != b.admin_listen || a.control_socket != b.control_socket {
            warnings.push("server.listen / admin_listen / control_socket changes need a restart to take effect".to_string());
        }
        if a.max_body_bytes != b.max_body_bytes || a.shutdown_timeout_secs != b.shutdown_timeout_secs {
            warnings.push("server.max_body_bytes / shutdown_timeout_secs changes need a restart to take effect".to_string());
        }
        let (pool, mut report) = build_pool(config, prev.version + 1, Some(&prev)).await;
        report.warnings = warnings;
        self.pool.store(Arc::new(pool));
        let changes = serde_json::to_string(&report.changes).unwrap_or_default();
        tracing::info!(version = report.version, changes, disabled = report.disabled.len(), "config reloaded");
        Ok(report)
    }
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/v1/embeddings", post(api::embeddings::handle))
        .route("/v1/rerank", post(api::rerank::handle))
        .route("/v2/rerank", post(api::rerank::handle))
        .route("/v1/systemone", post(api::decision))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .layer(middleware::from_fn(access))
        .layer(DefaultBodyLimit::max(app.boot_server.max_body_bytes))
        .with_state(app)
}

/// 沿用 client 的 `x-request-id`，沒有則產生；轉送給後端並回在回應上。
/// 每個 request（健康檢查除外）寫一行 access log（target `access`），含呼叫過的後端與結果。
async fn access(mut req: Request, next: Next) -> Response {
    let started = Instant::now();
    let id = match req.headers().get(REQUEST_ID) {
        Some(v) => v.clone(),
        None => {
            let v = HeaderValue::from_str(&uuid::Uuid::new_v4().to_string()).expect("uuid is a valid header");
            req.headers_mut().insert(REQUEST_ID, v.clone());
            v
        }
    };
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let client = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|c| c.0.ip().to_string());
    let log = CallLog::default();
    req.extensions_mut().insert(log.clone());

    let mut resp = next.run(req).await;

    if path != "/healthz" && path != "/readyz" {
        let (items, backends, fallback) = log.summary();
        // decision 以串流回傳，duration 是收到後端回應 header 的時間
        tracing::info!(
            target: "access",
            request_id = id.to_str().unwrap_or(""),
            client = client.as_deref().unwrap_or(""),
            method = %method,
            path,
            status = resp.status().as_u16(),
            duration_ms = started.elapsed().as_secs_f64() * 1000.0,
            items,
            backends,
            fallback,
            "request"
        );
    }
    resp.headers_mut().insert(REQUEST_ID, id);
    resp
}

async fn healthz(State(app): State<Arc<App>>) -> Response {
    if app.draining.load(Ordering::Relaxed) {
        return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"status": "draining"}))).into_response();
    }
    axum::Json(json!({"status": "ok"})).into_response()
}

async fn readyz(State(app): State<Arc<App>>) -> Response {
    if app.draining.load(Ordering::Relaxed) {
        return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"status": "draining"}))).into_response();
    }
    let pool = app.pool.load();
    let unavailable: Vec<&str> =
        ServiceKind::ALL.iter().filter(|k| !pool.service(**k).has_usable()).map(|k| k.as_str()).collect();
    if unavailable.is_empty() {
        axum::Json(json!({"status": "ready"})).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"status": "not_ready", "unavailable": unavailable}))).into_response()
    }
}

/// 設定 `SO_REUSEPORT`，升級 binary 時新舊 process 可同時監聽同一個埠。
fn bind_reuseport(addr: SocketAddr) -> std::io::Result<TcpListener> {
    let socket = if addr.is_ipv4() { TcpSocket::new_v4()? } else { TcpSocket::new_v6()? };
    socket.set_reuseaddr(true)?;
    socket.set_reuseport(true)?;
    socket.set_nodelay(true)?;
    socket.bind(addr)?;
    socket.listen(1024)
}

pub async fn serve(config_path: PathBuf) -> anyhow::Result<()> {
    let config = config::load(&config_path)?;
    tracing::info!(config = %config.path.display(), "probing backends");
    let (pool, report) = build_pool(config, 1, None).await;
    tracing::info!(version = report.version, disabled = report.disabled.len(), "config loaded");
    let app = Arc::new(App::new(pool, config_path));
    let server = app.boot_server.clone();

    let control = control::bind(&server.control_socket)?;
    let listener = bind_reuseport(server.listen).map_err(|e| anyhow::anyhow!("cannot listen on {}: {e}", server.listen))?;
    let admin_listener = match server.admin_listen {
        Some(addr) => Some(TcpListener::bind(addr).await.map_err(|e| anyhow::anyhow!("cannot listen on {addr}: {e}"))?),
        None => None,
    };
    tracing::info!(listen = %server.listen, admin = ?server.admin_listen, control = %server.control_socket.display(), "ai-gateway started");

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let wait_shutdown = |mut rx: watch::Receiver<bool>| async move {
        let _ = rx.wait_for(|v| *v).await;
    };

    tokio::spawn(control::serve(control, app.clone()));
    tokio::spawn(handle_signals(app.clone(), shutdown_tx));
    if let Some(l) = admin_listener {
        let admin = axum::serve(l, admin::router(app.clone())).with_graceful_shutdown(wait_shutdown(shutdown_rx.clone()));
        tokio::spawn(async move { admin.await });
    }
    let main = tokio::spawn(
        axum::serve(listener, router(app.clone()).into_make_service_with_connect_info::<SocketAddr>())
            .with_graceful_shutdown(wait_shutdown(shutdown_rx.clone()))
            .into_future(),
    );

    wait_shutdown(shutdown_rx).await;
    tracing::info!(timeout_secs = server.shutdown_timeout_secs, "draining in-flight requests");
    match tokio::time::timeout(Duration::from_secs(server.shutdown_timeout_secs), main).await {
        Ok(_) => tracing::info!("all in-flight requests finished"),
        Err(_) => {
            let pool = app.pool.load();
            let inflight: i64 = ServiceKind::ALL
                .iter()
                .flat_map(|k| pool.service(*k).backends.iter())
                .map(|b| b.stats.inflight.load(Ordering::Relaxed))
                .sum();
            tracing::warn!(inflight, "shutdown timeout reached; exiting with unfinished requests");
        }
    }
    let _ = std::fs::remove_file(&server.control_socket);
    Ok(())
}

/// SIGHUP：熱重載；SIGTERM / SIGINT：進入 draining 並優雅關閉。
async fn handle_signals(app: Arc<App>, shutdown: watch::Sender<bool>) {
    let mut hup = signal(SignalKind::hangup()).expect("install SIGHUP handler");
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    loop {
        tokio::select! {
            _ = hup.recv() => {
                let app = app.clone();
                tokio::spawn(async move {
                    if let Err(e) = app.reload().await {
                        tracing::error!(error = %e, "reload failed; keeping current config");
                    }
                });
            }
            _ = term.recv() => break,
            _ = int.recv() => break,
        }
    }
    app.draining.store(true, Ordering::Relaxed);
    let _ = shutdown.send(true);
}
