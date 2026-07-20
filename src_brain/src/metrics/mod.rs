// ── Metrics system — §9.3 Turn-level telemetry + §12 Observability ────────
// Prometheus-style counters and histogram buckets exposed via /metrics endpoint.
// All counters use atomic operations for zero-lock reads from any thread.

use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::sync::RwLock;

// ── Atomic counters ──────────────────────────────────────────────────────────

pub struct Metrics {
    /// requests_total counter: key = "worker:model:status"
    pub requests_total: RwLock<HashMap<String, AtomicU64>>,

    /// ttft_ms bucket: key = "worker:model:le", bucket values are cumulative
    /// Buckets: 50, 100, 250, 500, 1000, 2500, 5000, 10000, +Inf
    pub ttft_ms: RwLock<HashMap<String, AtomicU64>>,

    /// tokens_per_sec sum: key = "worker:model"
    pub tokens_per_sec_sum: RwLock<HashMap<String, AtomicU64>>,
    pub tokens_per_sec_count: RwLock<HashMap<String, AtomicU64>>,

    /// cache_hit_ratio: key = "worker", stores (hits, total)
    pub cache_hits: RwLock<HashMap<String, AtomicU64>>,
    pub cache_total: RwLock<HashMap<String, AtomicU64>>,

    /// queue_depth: key = "worker", current gauge
    pub queue_depth: RwLock<HashMap<String, AtomicU64>>,

    /// tool_call_total: key = "tool_name:status"
    pub tool_call_total: RwLock<HashMap<String, AtomicU64>>,

    /// worker_health: key = "worker", 1=healthy, 0=unhealthy
    pub worker_health: RwLock<HashMap<String, AtomicU64>>,

    /// prompt_tokens_total / completion_tokens_total: key = "worker:model"
    pub prompt_tokens: RwLock<HashMap<String, AtomicU64>>,
    pub completion_tokens: RwLock<HashMap<String, AtomicU64>>,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_total:       RwLock::new(HashMap::new()),
            ttft_ms:              RwLock::new(HashMap::new()),
            tokens_per_sec_sum:   RwLock::new(HashMap::new()),
            tokens_per_sec_count: RwLock::new(HashMap::new()),
            cache_hits:           RwLock::new(HashMap::new()),
            cache_total:          RwLock::new(HashMap::new()),
            queue_depth:          RwLock::new(HashMap::new()),
            tool_call_total:      RwLock::new(HashMap::new()),
            worker_health:        RwLock::new(HashMap::new()),
            prompt_tokens:        RwLock::new(HashMap::new()),
            completion_tokens:    RwLock::new(HashMap::new()),
        }
    }

    // ── Atomic helper: get-or-create + add ───────────────────────────────────

    fn atom_add(map: &RwLock<HashMap<String, AtomicU64>>, key: &str, delta: u64) {
        // Fast path: key already exists — no write lock needed
        {
            let guard = map.read().unwrap();
            if let Some(a) = guard.get(key) {
                a.fetch_add(delta, Ordering::Relaxed);
                return;
            }
        }
        // Slow path: key is new — acquire write lock, insert with 0, then add
        let mut g = map.write().unwrap();
        g.entry(key.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(delta, Ordering::Relaxed);
    }

    fn atom_set(map: &RwLock<HashMap<String, AtomicU64>>, key: &str, value: u64) {
        let guard = map.read().unwrap();
        if let Some(a) = guard.get(key) {
            a.store(value, Ordering::Relaxed);
        } else {
            drop(guard);
            let mut g = map.write().unwrap();
            g.entry(key.to_string())
                .or_insert_with(|| AtomicU64::new(value));
            if let Some(a) = g.get(key) {
                a.store(value, Ordering::Relaxed);
            }
        }
    }

    fn atom_read(map: &RwLock<HashMap<String, AtomicU64>>, key: &str) -> u64 {
        map.read().unwrap()
            .get(key)
            .map(|a| a.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    fn atom_snapshot(map: &RwLock<HashMap<String, AtomicU64>>) -> HashMap<String, u64> {
        map.read().unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect()
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Record a request completion. `status` is "200", "429", "503", etc.
    pub fn record_request(&self, worker_id: &str, model: &str, status: u16) {
        let key = format!("{}:{}:{}", worker_id, model, status);
        Self::atom_add(&self.requests_total, &key, 1);
    }

    /// Record time-to-first-token in milliseconds. Adds to all applicable buckets.
    pub fn record_ttft(&self, worker_id: &str, model: &str, ttft_ms: u64) {
        let buckets = [50, 100, 250, 500, 1000, 2500, 5000, 10000];
        for &le in &buckets {
            if ttft_ms <= le {
                let key = format!("{}:{}:le:{}", worker_id, model, le);
                Self::atom_add(&self.ttft_ms, &key, 1);
            }
        }
        // +Inf bucket
        let inf_key = format!("{}:{}:le:+Inf", worker_id, model);
        Self::atom_add(&self.ttft_ms, &inf_key, 1);
    }

    /// Record tokens-per-second measurement.
    pub fn record_tps(&self, worker_id: &str, model: &str, tps: f32) {
        let key = format!("{}:{}", worker_id, model);
        Self::atom_add(&self.tokens_per_sec_sum, &key, (tps * 1000.0) as u64);
        Self::atom_add(&self.tokens_per_sec_count, &key, 1);
    }

    /// Record cache hit ratio data point.
    pub fn record_cache_ratio(&self, worker_id: &str, hits: u64, total: u64) {
        let key = worker_id.to_string();
        Self::atom_add(&self.cache_hits, &key, hits);
        Self::atom_add(&self.cache_total, &key, total);
    }

    /// Set current queue depth for a worker.
    pub fn set_queue_depth(&self, worker_id: &str, depth: u64) {
        Self::atom_set(&self.queue_depth, worker_id, depth);
    }

    /// Record a tool call outcome.
    pub fn record_tool_call(&self, tool_name: &str, status: &str) {
        let key = format!("{}:{}", tool_name, status);
        Self::atom_add(&self.tool_call_total, &key, 1);
    }

    /// Set worker health gauge.
    pub fn set_worker_health(&self, worker_id: &str, healthy: bool) {
        Self::atom_set(&self.worker_health, worker_id, if healthy { 1 } else { 0 });
    }

    /// Record token counts.
    pub fn record_tokens(&self, worker_id: &str, model: &str, prompt: u64, completion: u64) {
        let key = format!("{}:{}", worker_id, model);
        Self::atom_add(&self.prompt_tokens, &key, prompt);
        Self::atom_add(&self.completion_tokens, &key, completion);
    }

    // ── Snapshot for /metrics endpoint ──────────────────────────────────────

    /// Build a Prometheus-compatible text representation of all metrics.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(4096);

        // --- HELP/TYPE headers ---
        out.push_str("# HELP brain_requests_total Total requests by worker, model, status\n");
        out.push_str("# TYPE brain_requests_total counter\n");
        for (k, v) in Self::atom_snapshot(&self.requests_total) {
            let parts: Vec<&str> = k.splitn(3, ':').collect();
            if parts.len() == 3 {
                out.push_str(&format!(
                    "brain_requests_total{{worker=\"{}\",model=\"{}\",status=\"{}\"}} {}\n",
                    parts[0], parts[1], parts[2], v
                ));
            }
        }

        out.push_str("# HELP brain_ttft_ms_bucket Time-to-first-token histogram\n");
        out.push_str("# TYPE brain_ttft_ms_bucket histogram\n");
        for (k, v) in Self::atom_snapshot(&self.ttft_ms) {
            // key format: "worker:model:le:<bucket>"
            let parts: Vec<&str> = k.split(':').collect();
            if parts.len() >= 4 {
                let worker = parts[0];
                let model  = parts[1];
                let le_str = if parts.len() == 5 {
                    format!("{}:{}", parts[3], parts[4])
                } else {
                    parts[3].to_string()
                };
                out.push_str(&format!(
                    "brain_ttft_ms_bucket{{worker=\"{}\",model=\"{}\",le=\"{}\"}} {}\n",
                    worker, model, le_str, v
                ));
            }
        }

        out.push_str("# HELP brain_tokens_per_sec Tokens-per-second by worker, model\n");
        out.push_str("# TYPE brain_tokens_per_sec gauge\n");
        for (k, total) in Self::atom_snapshot(&self.tokens_per_sec_sum) {
            let count = Self::atom_read(&self.tokens_per_sec_count, &k);
            let avg = if count > 0 {
                (total as f64 / count as f64) / 1000.0
            } else {
                0.0
            };
            let parts: Vec<&str> = k.splitn(2, ':').collect();
            if parts.len() == 2 {
                out.push_str(&format!(
                    "brain_tokens_per_sec{{worker=\"{}\",model=\"{}\"}} {:.2}\n",
                    parts[0], parts[1], avg
                ));
            }
        }

        out.push_str("# HELP brain_cache_hit_ratio Cache hit ratio by worker\n");
        out.push_str("# TYPE brain_cache_hit_ratio gauge\n");
        for (k, hits) in Self::atom_snapshot(&self.cache_hits) {
            let total = Self::atom_read(&self.cache_total, &k);
            let ratio = if total > 0 { hits as f64 / total as f64 } else { 0.0 };
            out.push_str(&format!(
                "brain_cache_hit_ratio{{worker=\"{}\"}} {:.4}\n", k, ratio
            ));
        }

        out.push_str("# HELP brain_queue_depth Current queue depth by worker\n");
        out.push_str("# TYPE brain_queue_depth gauge\n");
        for (k, v) in Self::atom_snapshot(&self.queue_depth) {
            out.push_str(&format!(
                "brain_queue_depth{{worker=\"{}\"}} {}\n", k, v
            ));
        }

        out.push_str("# HELP brain_tool_call_total Tool call outcomes\n");
        out.push_str("# TYPE brain_tool_call_total counter\n");
        for (k, v) in Self::atom_snapshot(&self.tool_call_total) {
            let parts: Vec<&str> = k.splitn(2, ':').collect();
            if parts.len() == 2 {
                out.push_str(&format!(
                    "brain_tool_call_total{{tool_name=\"{}\",status=\"{}\"}} {}\n",
                    parts[0], parts[1], v
                ));
            }
        }

        out.push_str("# HELP brain_worker_health Worker health gauge (1=healthy)\n");
        out.push_str("# TYPE brain_worker_health gauge\n");
        for (k, v) in Self::atom_snapshot(&self.worker_health) {
            out.push_str(&format!(
                "brain_worker_health{{worker=\"{}\"}} {}\n", k, v
            ));
        }

        out.push_str("# HELP brain_prompt_tokens_total Total prompt tokens\n");
        out.push_str("# TYPE brain_prompt_tokens_total counter\n");
        for (k, v) in Self::atom_snapshot(&self.prompt_tokens) {
            let parts: Vec<&str> = k.splitn(2, ':').collect();
            if parts.len() == 2 {
                out.push_str(&format!(
                    "brain_prompt_tokens_total{{worker=\"{}\",model=\"{}\"}} {}\n",
                    parts[0], parts[1], v
                ));
            }
        }

        out.push_str("# HELP brain_completion_tokens_total Total completion tokens\n");
        out.push_str("# TYPE brain_completion_tokens_total counter\n");
        for (k, v) in Self::atom_snapshot(&self.completion_tokens) {
            let parts: Vec<&str> = k.splitn(2, ':').collect();
            if parts.len() == 2 {
                out.push_str(&format!(
                    "brain_completion_tokens_total{{worker=\"{}\",model=\"{}\"}} {}\n",
                    parts[0], parts[1], v
                ));
            }
        }

        out
    }
}

impl Default for Metrics {
    fn default() -> Self { Self::new() }
}

// ── Turn-level telemetry struct (§9.3) ──────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct TurnMetrics {
    pub ttft_ms: u32,
    pub tokens_per_sec: f32,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cache_hit_tokens: u32,
    pub worker_id: String,
}
