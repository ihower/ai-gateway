//! 管理端點（唯讀）：後端狀態、目前設定與 Prometheus metrics（SPEC §3.2、§9）。

use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::Router;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;
use serde_json::json;

use crate::backend::{BackendState, LATENCY_BUCKETS};
use crate::config::ServiceKind;
use crate::server::App;

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/admin/backends", get(backends))
        .route("/admin/config", get(config))
        .route("/metrics", get(metrics))
        .with_state(app)
}

async fn backends(State(app): State<Arc<App>>) -> impl IntoResponse {
    axum::Json(app.pool.load().status())
}

async fn config(State(app): State<Arc<App>>) -> impl IntoResponse {
    let pool = app.pool.load();
    axum::Json(json!({"version": pool.version, "loaded_at_unix": pool.loaded_at_unix, "config": &*pool.config}))
}

async fn metrics(State(app): State<Arc<App>>) -> impl IntoResponse {
    let pool = app.pool.load();
    let mut out = String::new();
    let _ = writeln!(out, "# TYPE ai_gateway_config_version gauge\nai_gateway_config_version {}", pool.version);
    let headers = [
        ("ai_gateway_backend_up", "gauge"),
        ("ai_gateway_backend_requests_total", "counter"),
        ("ai_gateway_backend_errors_total", "counter"),
        ("ai_gateway_backend_inflight", "gauge"),
        ("ai_gateway_backend_request_duration_seconds", "histogram"),
    ];
    for (name, kind) in headers {
        let _ = writeln!(out, "# TYPE {name} {kind}");
        for svc in ServiceKind::ALL {
            for b in &pool.service(svc).backends {
                let labels = format!("service=\"{svc}\",backend=\"{}\"", b.url);
                let s = &b.stats;
                match name {
                    "ai_gateway_backend_up" => {
                        let _ = writeln!(out, "{name}{{{labels}}} {}", (b.state() == BackendState::Enabled) as u8);
                    }
                    "ai_gateway_backend_requests_total" => {
                        let _ = writeln!(out, "{name}{{{labels}}} {}", s.requests.load(Ordering::Relaxed));
                    }
                    "ai_gateway_backend_errors_total" => {
                        let _ = writeln!(out, "{name}{{{labels}}} {}", s.errors.load(Ordering::Relaxed));
                    }
                    "ai_gateway_backend_inflight" => {
                        let _ = writeln!(out, "{name}{{{labels}}} {}", s.inflight.load(Ordering::Relaxed));
                    }
                    _ => {
                        let mut cumulative = 0;
                        for (le, count) in LATENCY_BUCKETS.iter().zip(&s.latency_buckets) {
                            cumulative += count.load(Ordering::Relaxed);
                            let _ = writeln!(out, "{name}_bucket{{{labels},le=\"{le}\"}} {cumulative}");
                        }
                        let total = s.latency_count.load(Ordering::Relaxed);
                        let _ = writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {total}");
                        let sum = s.latency_sum_micros.load(Ordering::Relaxed) as f64 / 1e6;
                        let _ = writeln!(out, "{name}_sum{{{labels}}} {sum}\n{name}_count{{{labels}}} {total}");
                    }
                }
            }
        }
    }
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], out)
}
