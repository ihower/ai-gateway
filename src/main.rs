use std::path::PathBuf;
use std::process::ExitCode;

use ai_gateway::control::{self, ClientError, Command};
use ai_gateway::probe::{LoadReport, build_pool};
use ai_gateway::config::ServiceKind;
use ai_gateway::{config, server};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ai-gateway", version, about = "Embedding / reranker / decision API gateway")]
struct Cli {
    /// 設定檔路徑
    #[arg(long, global = true, env = "AI_GATEWAY_CONFIG", default_value = "config.yaml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 啟動 gateway
    Serve,
    /// 通知執行中的 gateway 重新載入設定
    Reload {
        #[arg(long)]
        json: bool,
    },
    /// 驗證設定檔並探測所有後端，不影響執行中的 gateway
    Check {
        #[arg(long)]
        json: bool,
    },
    /// 印出執行中 gateway 的設定版本與各後端狀態
    Status {
        #[arg(long)]
        json: bool,
    },
    /// 將執行中 gateway 各後端的 requests / items / errors / 延遲統計歸零
    Reset,
    /// 通知執行中的 gateway 優雅關閉（等同送 SIGTERM）
    Shutdown,
    /// 手動停用後端（不再分派新的 request），直到下次 reload 重新探測
    Disable {
        /// `status` 顯示的編號（如 3），或後端的 name / url
        backend: String,
        /// 只套用到指定的 service；省略時套用到所有符合的 service
        #[arg(long, value_parser = parse_service)]
        service: Option<ServiceKind>,
    },
    /// 強制啟用後端（含 reload 探測失敗被停用者），並清除 unhealthy 狀態，直到下次 reload 重新探測
    Enable {
        /// `status` 顯示的編號（如 3），或後端的 name / url
        backend: String,
        /// 只套用到指定的 service；省略時套用到所有符合的 service
        #[arg(long, value_parser = parse_service)]
        service: Option<ServiceKind>,
    },
}

fn parse_service(s: &str) -> Result<ServiceKind, String> {
    ServiceKind::ALL.into_iter().find(|k| k.as_str() == s).ok_or_else(|| "expected one of: embedding, reranker, decision".to_string())
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Serve => {
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
                )
                .init();
            match server::serve(cli.config).await {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    tracing::error!(error = %format!("{e:#}"), "ai-gateway failed");
                    ExitCode::from(1)
                }
            }
        }
        Cmd::Reload { json } => reload(&cli.config, json).await,
        Cmd::Check { json } => check(&cli.config, json).await,
        Cmd::Status { json } => status(&cli.config, json).await,
        Cmd::Reset => simple_command(&cli.config, Command::Reset, "stats reset to 0").await,
        Cmd::Shutdown => {
            simple_command(&cli.config, Command::Shutdown, "shutdown requested; gateway is draining in-flight requests").await
        }
        Cmd::Disable { backend, service } => backend_command(&cli.config, Command::Disable { backend, service }).await,
        Cmd::Enable { backend, service } => backend_command(&cli.config, Command::Enable { backend, service }).await,
    }
}

async fn control_send(config_path: &std::path::Path, cmd: Command) -> Result<control::Reply, ExitCode> {
    let socket = match config::read_control_socket(config_path) {
        Ok(s) => s,
        Err(e) => {
            // 設定檔讀不到或格式錯誤屬於設定錯誤
            eprintln!("config error: {e:#}");
            return Err(ExitCode::from(1));
        }
    };
    control::send(&socket, &cmd).await.map_err(|e| {
        eprintln!("error: {e}");
        match e {
            ClientError::Connect { .. } => ExitCode::from(2),
            ClientError::Protocol(_) => ExitCode::from(1),
        }
    })
}

async fn reload(config_path: &std::path::Path, json: bool) -> ExitCode {
    let reply = match control_send(config_path, Command::Reload).await {
        Ok(r) => r,
        Err(code) => return code,
    };
    match reply.report {
        Some(report) if reply.ok => {
            if json {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                println!("reloaded config v{} from {}", report.version, report.config_path);
                print_report(&report);
            }
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("reload failed (current config kept): {}", reply.error.unwrap_or_default());
            ExitCode::from(1)
        }
    }
}

fn print_report(report: &LoadReport) {
    for (service, changes) in &report.changes {
        if changes.added.is_empty() && changes.removed.is_empty() {
            println!("{service:<10} (no change)");
        }
        for url in &changes.added {
            println!("{service:<10} + {url}");
        }
        for url in &changes.removed {
            println!("{service:<10} - {url}");
        }
    }
    for d in &report.disabled {
        println!("{:<10} ! {}  disabled ({})", d.service, d.url, d.reason);
    }
    for w in &report.warnings {
        println!("warning: {w}");
    }
}

async fn check(config_path: &std::path::Path, json: bool) -> ExitCode {
    let config = match config::load(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {e:#}");
            return ExitCode::from(1);
        }
    };
    let services: Vec<_> = config::ServiceKind::ALL
        .iter()
        .flat_map(|k| config.service(*k).urls().map(move |u| (*k, u.to_string())))
        .collect();
    let path = config.path.display().to_string();
    let (_, report) = build_pool(config, 0, None).await;
    if json {
        println!("{}", serde_json::to_string_pretty(&report.disabled).unwrap());
    } else {
        println!("config OK: {path}");
        for (kind, url) in services {
            match report.disabled.iter().find(|d| d.service == kind.as_str() && d.url == url) {
                Some(d) => println!("{kind:<10} {url}  DISABLED ({})", d.reason),
                None => println!("{kind:<10} {url}  ok"),
            }
        }
    }
    ExitCode::SUCCESS
}

/// 送出不需額外輸出的命令，成功時印出 `done`。
async fn simple_command(config_path: &std::path::Path, cmd: Command, done: &str) -> ExitCode {
    match control_send(config_path, cmd).await {
        Ok(reply) if reply.ok => {
            println!("{done}");
            ExitCode::SUCCESS
        }
        Ok(reply) => {
            eprintln!("error: {}", reply.error.unwrap_or_default());
            ExitCode::from(1)
        }
        Err(code) => code,
    }
}

/// 送出 disable / enable，印出每個符合的後端與警告。
async fn backend_command(config_path: &std::path::Path, cmd: Command) -> ExitCode {
    match control_send(config_path, cmd).await {
        Ok(reply) if reply.ok => {
            for m in &reply.messages {
                println!("{m}");
            }
            for w in &reply.warnings {
                println!("warning: {w}");
            }
            ExitCode::SUCCESS
        }
        Ok(reply) => {
            eprintln!("error: {}", reply.error.unwrap_or_default());
            ExitCode::from(1)
        }
        Err(code) => code,
    }
}

async fn status(config_path: &std::path::Path, json: bool) -> ExitCode {
    let reply = match control_send(config_path, Command::Status).await {
        Ok(r) => r,
        Err(code) => return code,
    };
    let Some(status) = reply.status else {
        eprintln!("error: {}", reply.error.unwrap_or_default());
        return ExitCode::from(1);
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&status).unwrap());
        return ExitCode::SUCCESS;
    }
    println!("config v{} from {} (loaded at unix {})", status.version, status.config_path, status.loaded_at_unix);
    // 有設定 name 時顯示「name (url)」，否則只顯示 url
    let label = |b: &ai_gateway::backend::BackendStatus| {
        if b.name.is_empty() || b.name == b.url { b.url.clone() } else { format!("{} ({})", b.name, b.url) }
    };
    let width = status.services.values().flatten().map(|b| label(b).chars().count()).max().unwrap_or(0).max(32);
    for (service, backends) in &status.services {
        for b in backends {
            let state = serde_json::to_value(b.state).unwrap();
            let reason = b.reason.as_deref().map(|r| format!(" ({r})")).unwrap_or_default();
            let role = if b.fallback { " [fallback]" } else { "" };
            // decision 沒有筆數概念，不顯示 items
            let items = if service == "decision" { String::new() } else { format!(" items={}", b.items) };
            println!(
                "#{:<3} {service:<10} {:<width$} {:<10} inflight={} requests={}{items} errors={}{role}{reason}",
                b.id,
                label(b),
                state.as_str().unwrap_or(""),
                b.inflight,
                b.requests,
                b.errors
            );
        }
    }
    ExitCode::SUCCESS
}
