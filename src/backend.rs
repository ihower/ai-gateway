//! 後端池：後端狀態、加權輪流挑選與拆分比例（SPEC §4.2、§8.1、§8.2）。

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::config::{Config, ServiceConfig, ServiceKind, Tuning};

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

fn now_ms() -> u64 {
    START.elapsed().as_millis() as u64
}

/// 延遲分佈的 bucket 上界（秒），對應 Prometheus histogram。
pub const LATENCY_BUCKETS: [f64; 12] = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0];

/// 累計統計；reload 後 URL 相同的後端沿用同一份。
#[derive(Default)]
pub struct BackendStats {
    pub requests: AtomicU64,
    pub errors: AtomicU64,
    pub inflight: AtomicI64,
    pub latency_buckets: [AtomicU64; LATENCY_BUCKETS.len()],
    pub latency_sum_micros: AtomicU64,
    pub latency_count: AtomicU64,
}

impl BackendStats {
    fn observe(&self, elapsed: Duration) {
        let secs = elapsed.as_secs_f64();
        if let Some(i) = LATENCY_BUCKETS.iter().position(|b| secs <= *b) {
            self.latency_buckets[i].fetch_add(1, Ordering::Relaxed);
        }
        self.latency_sum_micros.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// 一次對後端的呼叫期間持有；drop 時更新 in-flight 數與延遲。
pub struct InflightGuard {
    stats: Arc<BackendStats>,
    started: Instant,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.stats.inflight.fetch_sub(1, Ordering::Relaxed);
        self.stats.observe(self.started.elapsed());
    }
}

pub struct Backend {
    pub service: ServiceKind,
    pub url: String,
    pub weight: u32,
    /// 啟動 / reload 時探測失敗的原因；`Some` 表示停用，直到下次 reload。
    pub disabled: Option<String>,
    /// 只在所有一般後端都失效時使用（SPEC §8.6）。
    pub fallback: bool,
    /// 覆寫 service 的 `request_timeout_ms`。
    pub request_timeout_ms: Option<u64>,
    consecutive_failures: AtomicU32,
    unhealthy_until_ms: AtomicU64,
    pub stats: Arc<BackendStats>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendState {
    Enabled,
    Unhealthy,
    Disabled,
}

impl Backend {
    pub fn new(service: ServiceKind, url: String, weight: u32, disabled: Option<String>, stats: Arc<BackendStats>) -> Self {
        Backend {
            service,
            url,
            weight,
            disabled,
            fallback: false,
            request_timeout_ms: None,
            consecutive_failures: AtomicU32::new(0),
            unhealthy_until_ms: AtomicU64::new(0),
            stats,
        }
    }

    pub fn into_fallback(mut self, request_timeout_ms: Option<u64>) -> Self {
        self.fallback = true;
        self.request_timeout_ms = request_timeout_ms;
        self
    }

    pub fn state(&self) -> BackendState {
        if self.disabled.is_some() {
            BackendState::Disabled
        } else if self.unhealthy_until_ms.load(Ordering::Relaxed) > now_ms() {
            BackendState::Unhealthy
        } else {
            BackendState::Enabled
        }
    }

    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures.load(Ordering::Relaxed)
    }

    pub fn start_call(&self) -> InflightGuard {
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        self.stats.inflight.fetch_add(1, Ordering::Relaxed);
        InflightGuard { stats: self.stats.clone(), started: Instant::now() }
    }

    pub fn record_success(&self) {
        if self.consecutive_failures.swap(0, Ordering::Relaxed) != 0 {
            self.unhealthy_until_ms.store(0, Ordering::Relaxed);
        }
    }

    /// 連續失敗達門檻即標記為 unhealthy；冷卻結束後若再失敗，計數仍在門檻以上，會立即再次標記。
    pub fn record_failure(&self, tuning: &Tuning, reason: &str) {
        self.stats.errors.fetch_add(1, Ordering::Relaxed);
        let n = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if n >= tuning.unhealthy_threshold {
            let now = now_ms();
            let until = now + tuning.unhealthy_cooldown_secs * 1000;
            let prev = self.unhealthy_until_ms.swap(until, Ordering::Relaxed);
            if prev <= now {
                tracing::warn!(service = %self.service, backend = %self.url, consecutive_failures = n,
                    cooldown_secs = tuning.unhealthy_cooldown_secs, reason, "backend marked unhealthy");
            }
        }
    }
}

pub struct ServicePool {
    pub kind: ServiceKind,
    pub tuning: Tuning,
    pub probe_model: Option<String>,
    /// 一般後端在前，fallback（若有）排在最後。
    pub backends: Vec<Arc<Backend>>,
    pub client: reqwest::Client,
    /// 平滑加權輪流的排程（後端索引序列）。
    schedule: Vec<usize>,
    rr: AtomicUsize,
}

impl ServicePool {
    pub fn new(kind: ServiceKind, cfg: &ServiceConfig, backends: Vec<Arc<Backend>>) -> Self {
        debug_assert!(backends.iter().rev().skip(1).all(|b| !b.fallback), "fallback must be last");
        let weights: Vec<u32> = backends.iter().filter(|b| !b.fallback).map(|b| b.weight).collect();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(cfg.tuning.connect_timeout_ms))
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(256)
            .tcp_nodelay(true)
            .build()
            .expect("reqwest client");
        ServicePool {
            kind,
            tuning: cfg.tuning.clone(),
            probe_model: cfg.probe_model.clone(),
            backends,
            client,
            schedule: smooth_wrr_schedule(&weights),
            rr: AtomicUsize::new(0),
        }
    }

    /// 可被選中的一般後端：啟用中且健康者；若全部不健康，有 fallback 時回空（改用 fallback），
    /// 否則退回所有啟用中的後端（SPEC §8.1）。
    pub fn eligible(&self, exclude: &[usize]) -> Vec<usize> {
        let enabled: Vec<usize> = (0..self.backends.len())
            .filter(|i| !self.backends[*i].fallback && self.backends[*i].disabled.is_none() && !exclude.contains(i))
            .collect();
        let healthy: Vec<usize> =
            enabled.iter().copied().filter(|i| self.backends[*i].state() == BackendState::Enabled).collect();
        if !healthy.is_empty() || self.fallback().is_some() { healthy } else { enabled }
    }

    /// 未停用的 fallback 後端索引。fallback 是最後手段，unhealthy 時仍會使用。
    pub fn fallback(&self) -> Option<usize> {
        self.backends.iter().position(|b| b.fallback && b.disabled.is_none())
    }

    /// 依加權輪流挑出最多 `k` 個不重複的後端。
    pub fn pick(&self, k: usize, exclude: &[usize]) -> Vec<usize> {
        let eligible = self.eligible(exclude);
        let k = k.min(eligible.len());
        let mut chosen = Vec::with_capacity(k);
        if k == 0 {
            return chosen;
        }
        let start = self.rr.fetch_add(1, Ordering::Relaxed);
        for step in 0..self.schedule.len() {
            let idx = self.schedule[(start + step) % self.schedule.len()];
            if eligible.contains(&idx) && !chosen.contains(&idx) {
                chosen.push(idx);
                if chosen.len() == k {
                    break;
                }
            }
        }
        chosen
    }

    pub fn has_usable(&self) -> bool {
        self.backends.iter().any(|b| b.state() == BackendState::Enabled)
    }
}

/// nginx 的平滑加權輪流：權重 {a:3, b:1} 產生 a a b a 而非 a a a b。
pub fn smooth_wrr_schedule(weights: &[u32]) -> Vec<usize> {
    let total: i64 = weights.iter().map(|w| *w as i64).sum();
    let mut current = vec![0i64; weights.len()];
    let mut schedule = Vec::with_capacity(total as usize);
    for _ in 0..total {
        for (c, w) in current.iter_mut().zip(weights) {
            *c += *w as i64;
        }
        let best = (0..weights.len()).max_by_key(|i| (current[*i], std::cmp::Reverse(*i))).unwrap();
        current[best] -= total;
        schedule.push(best);
    }
    schedule
}

/// 依權重把 `n` 筆切成連續區段的筆數（最大餘數法），各份差距最小。
pub fn split_sizes(n: usize, weights: &[u32]) -> Vec<usize> {
    let total: u64 = weights.iter().map(|w| *w as u64).sum();
    let mut sizes: Vec<usize> = weights.iter().map(|w| (n as u64 * *w as u64 / total) as usize).collect();
    let mut remainders: Vec<(u64, usize)> =
        weights.iter().enumerate().map(|(i, w)| (n as u64 * *w as u64 % total, i)).collect();
    // 餘數大者優先；相同時前面的後端優先（挑選順序已依輪流輪替）
    remainders.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut left = n - sizes.iter().sum::<usize>();
    for (_, i) in remainders {
        if left == 0 {
            break;
        }
        sizes[i] += 1;
        left -= 1;
    }
    sizes
}

pub struct Pool {
    pub version: u64,
    pub loaded_at_unix: u64,
    pub config: Arc<Config>,
    pub embedding: ServicePool,
    pub reranker: ServicePool,
    pub decision: ServicePool,
}

impl Pool {
    pub fn service(&self, kind: ServiceKind) -> &ServicePool {
        match kind {
            ServiceKind::Embedding => &self.embedding,
            ServiceKind::Reranker => &self.reranker,
            ServiceKind::Decision => &self.decision,
        }
    }

    pub fn stats_by_url(&self) -> HashMap<(ServiceKind, String), Arc<BackendStats>> {
        ServiceKind::ALL
            .iter()
            .flat_map(|k| self.service(*k).backends.iter().map(move |b| ((*k, b.url.clone()), b.stats.clone())))
            .collect()
    }

    pub fn status(&self) -> PoolStatus {
        let services = ServiceKind::ALL
            .iter()
            .map(|k| {
                let backends = self
                    .service(*k)
                    .backends
                    .iter()
                    .map(|b| BackendStatus {
                        url: b.url.clone(),
                        fallback: b.fallback,
                        weight: b.weight,
                        state: b.state(),
                        reason: b.disabled.clone(),
                        consecutive_failures: b.consecutive_failures(),
                        inflight: b.stats.inflight.load(Ordering::Relaxed),
                        requests: b.stats.requests.load(Ordering::Relaxed),
                        errors: b.stats.errors.load(Ordering::Relaxed),
                    })
                    .collect();
                (k.as_str().to_string(), backends)
            })
            .collect();
        PoolStatus {
            version: self.version,
            loaded_at_unix: self.loaded_at_unix,
            config_path: self.config.path.display().to_string(),
            services,
        }
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct PoolStatus {
    pub version: u64,
    pub loaded_at_unix: u64,
    pub config_path: String,
    pub services: std::collections::BTreeMap<String, Vec<BackendStatus>>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct BackendStatus {
    pub url: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fallback: bool,
    pub weight: u32,
    pub state: BackendState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub consecutive_failures: u32,
    pub inflight: i64,
    pub requests: u64,
    pub errors: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BackendConfig;

    fn pool(weights: &[u32], disabled: &[usize]) -> ServicePool {
        pool_with_fallback(weights, disabled, None)
    }

    /// `fallback`：`Some(停用原因)`，`Some(None)` 為啟用中的 fallback。
    fn pool_with_fallback(weights: &[u32], disabled: &[usize], fallback: Option<Option<&str>>) -> ServicePool {
        let cfg = ServiceConfig {
            tuning: Tuning::default(),
            probe_model: None,
            backends: weights.iter().enumerate().map(|(i, w)| BackendConfig { url: format!("http://b{i}"), weight: *w }).collect(),
            fallback: None,
        };
        let mut backends: Vec<Arc<Backend>> = cfg
            .backends
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let reason = disabled.contains(&i).then(|| "down".to_string());
                Arc::new(Backend::new(ServiceKind::Embedding, b.url.clone(), b.weight, reason, Default::default()))
            })
            .collect();
        if let Some(reason) = fallback {
            let b = Backend::new(ServiceKind::Embedding, "http://fb".into(), 1, reason.map(String::from), Default::default());
            backends.push(Arc::new(b.into_fallback(None)));
        }
        ServicePool::new(ServiceKind::Embedding, &cfg, backends)
    }

    #[test]
    fn smooth_schedule_interleaves() {
        assert_eq!(smooth_wrr_schedule(&[1, 1]), vec![0, 1]);
        assert_eq!(smooth_wrr_schedule(&[3, 1]), vec![0, 0, 1, 0]);
    }

    #[test]
    fn weighted_pick_follows_ratio() {
        let p = pool(&[45, 55], &[]);
        let mut counts = [0; 2];
        for _ in 0..100 {
            counts[p.pick(1, &[])[0]] += 1;
        }
        assert_eq!(counts, [45, 55]);
        // 平滑分佈：任意連續 20 次的比例也接近 9:11，不會先連續打同一台
        let first20 = smooth_wrr_schedule(&[45, 55])[..20].iter().filter(|i| **i == 0).count();
        assert!((8..=10).contains(&first20), "{first20}");
    }

    #[test]
    fn split_sizes_are_balanced() {
        assert_eq!(split_sizes(5, &[1, 1]), vec![3, 2]);
        assert_eq!(split_sizes(2, &[1, 1]), vec![1, 1]);
        assert_eq!(split_sizes(7, &[1, 1, 1]), vec![3, 2, 2]);
        assert_eq!(split_sizes(8, &[3, 1]), vec![6, 2]);
        assert_eq!(split_sizes(20, &[45, 55]), vec![9, 11]);
        assert_eq!(split_sizes(100, &[55, 45]), vec![55, 45]);
        assert_eq!(split_sizes(10, &[1, 1]).iter().sum::<usize>(), 10);
    }

    #[test]
    fn pick_rotates_and_skips_disabled() {
        let p = pool(&[1, 1], &[]);
        assert_eq!(p.pick(1, &[]), vec![0]);
        assert_eq!(p.pick(1, &[]), vec![1]);
        assert_eq!(p.pick(2, &[]), vec![0, 1]);
        assert_eq!(p.pick(2, &[]), vec![1, 0]);
        assert_eq!(p.pick(1, &[0]), vec![1]);

        let p = pool(&[1, 1], &[0]);
        assert_eq!(p.pick(2, &[]), vec![1]);
        let p = pool(&[1], &[0]);
        assert!(p.pick(1, &[]).is_empty());
    }

    #[test]
    fn unhealthy_after_threshold_and_fallback() {
        let p = pool(&[1, 1], &[]);
        let t = Tuning::default();
        p.backends[0].record_failure(&t, "x");
        p.backends[0].record_failure(&t, "x");
        assert_eq!(p.backends[0].state(), BackendState::Enabled);
        p.backends[0].record_failure(&t, "x");
        assert_eq!(p.backends[0].state(), BackendState::Unhealthy);
        assert_eq!(p.eligible(&[]), vec![1]);
        // 全部不健康時仍可選
        for _ in 0..3 {
            p.backends[1].record_failure(&t, "x");
        }
        assert_eq!(p.eligible(&[]), vec![0, 1]);
        p.backends[0].record_success();
        assert_eq!(p.backends[0].state(), BackendState::Enabled);
    }

    #[test]
    fn fallback_only_when_primaries_fail() {
        let p = pool_with_fallback(&[1, 1], &[], Some(None));
        assert_eq!(p.fallback(), Some(2));
        // fallback 不參與輪流
        for _ in 0..4 {
            assert!(!p.pick(3, &[]).contains(&2));
        }
        let t = Tuning::default();
        for _ in 0..3 {
            p.backends[0].record_failure(&t, "x");
        }
        assert_eq!(p.eligible(&[]), vec![1]);
        // 一般後端全部不健康 → 不再勉強嘗試，改用 fallback
        for _ in 0..3 {
            p.backends[1].record_failure(&t, "x");
        }
        assert!(p.eligible(&[]).is_empty());
        // fallback 不健康仍可用（最後手段）
        for _ in 0..3 {
            p.backends[2].record_failure(&t, "x");
        }
        assert_eq!(p.fallback(), Some(2));

        let p = pool_with_fallback(&[1, 1], &[0, 1], Some(None));
        assert!(p.pick(1, &[]).is_empty());
        assert!(p.has_usable());
        // fallback 停用時維持原本行為
        let p = pool_with_fallback(&[1], &[], Some(Some("down")));
        assert_eq!(p.fallback(), None);
        p.backends[0].record_failure(&t, "x");
        for _ in 0..3 {
            p.backends[0].record_failure(&t, "x");
        }
        assert_eq!(p.eligible(&[]), vec![0]);
    }
}
