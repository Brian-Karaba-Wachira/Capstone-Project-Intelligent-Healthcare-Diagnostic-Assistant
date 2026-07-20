use std::collections::HashMap;
use std::env;


pub const WORK_DIR: &str = "/etc/brain";

/// Governs which hosts `web_fetch` is allowed to reach. Deliberately
/// defaults to `DenyAll` (see `from_env`) — web_fetch does nothing until
/// this is explicitly configured, rather than silently being wide open.
#[derive(Debug, Clone, PartialEq)]
pub enum EgressPolicy {
    AllowAll,
    DenyAll,
    /// Only these domains (and subdomains) may be fetched.
    Allowlist(Vec<String>),
    /// Everything EXCEPT these domains (and subdomains) may be fetched.
    Denylist(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct Config {
    pub psk:              String,
    pub db_path:          String,
    pub addr:             String,
    pub downloads_dir:    String,
    pub jwt_secret:       String,

    /// How long to wait for a worker to respond before returning 504 (seconds)
    pub worker_timeout_s: u64,

    /// How long to wait for a free worker before returning 503 (seconds)
    pub queue_timeout_s:  u64,

    /// Model alias map: client sends "gpt-4" or "qwen" → we route to the real
    /// model name registered by the worker.
    /// Format: BRAIN_MODEL_ALIASES="gpt-4=qwen2.5-coder,gpt-4o=deepseek-coder"
    pub model_aliases:    HashMap<String, String>,

    /// If true, accept BRAIN_PSK directly as a bearer token in addition to
    /// DB-issued API keys (useful for Claude Code / simple setups)
    pub allow_psk_as_token: bool,

    // ── Telemetry & Observability (§9.3 + §12) ──────────────────────────────
    /// Enable /metrics endpoint for Prometheus scraping
    pub metrics_enabled: bool,

    /// Whether to capture turn-level telemetry (ttft, tps, cache hits, etc.)
    pub telemetry_enabled: bool,

    // ── Idempotency (§11.1) ─────────────────────────────────────────────────
    /// TTL for idempotency key cache in seconds (default 300s = 5 min)
    pub idempotency_ttl_s: u64,

    // ── Timeout decomposition (§11.2) ────────────────────────────────────────
    pub queue_wait_budget_s:    u64,
    pub prefill_budget_s:       u64,
    pub decode_budget_s:        u64,
    pub tool_dispatch_budget_s: u64,

    /// Default context window size (tokens) for workers that don't report it
    pub default_context_window: u32,

    // ── Pricing (for usage/cost accounting) ─────────────────────────────────
    /// Per-model USD cost per 1M prompt/completion tokens:
    /// BRAIN_MODEL_PRICING="qwen2.5-coder=0.00:0.00,gpt-4=5.00:15.00"
    /// Format per entry: "model=prompt_per_1m:completion_per_1m". Models with
    /// no entry default to 0.0 (i.e. free — the common case for local workers).
    pub model_pricing: HashMap<String, (f64, f64)>,

    // ── Hosted tools (web_search / web_fetch) ───────────────────────────────
    /// Base URL of a SearxNG instance to execute `web_search` against, e.g.
    /// "http://127.0.0.1:8080". `web_search`/`web_fetch` are only advertised
    /// to the worker (and only executed) when this is set — clients that
    /// request them get an explicit error otherwise, rather than a silent
    /// no-op. Requires SearxNG's `search.formats` to include `json` in
    /// settings.yml (disabled by default in most SearxNG installs).
    pub searxng_url: Option<String>,

    /// Max internal request/response rounds spent resolving web_search/
    /// web_fetch calls before giving up and surfacing an error — mirrors
    /// Claude Code's own turn cap, applied to this brain-side loop instead.
    pub hosted_tool_max_rounds: usize,

    /// Which hosts `web_fetch` may reach. Set via BRAIN_EGRESS_MODE
    /// (allow_all | deny_all | allowlist | denylist) + BRAIN_EGRESS_DOMAINS
    /// (comma-separated bare hostnames, no scheme/path).
    pub egress_policy: EgressPolicy,

    /// How many worker chunks a per-request stream channel buffers before the
    /// worker back-pressures. Bounded so a client that disconnects mid-stream
    /// can't make brain accumulate an entire response in memory; sized
    /// generously (default 1024, was a hardcoded 256) so a fast worker
    /// bursting tokens doesn't stall on a briefly slow client socket.
    pub stream_buffer_chunks: usize,
}

impl Config {
    pub fn from_env() -> Self {
        let psk = option_env!("BRAIN_PSK")
            .map(|s| s.to_string())
            .or_else(|| env::var("BRAIN_PSK").ok())
            .expect(
                "BRAIN_PSK is neither baked into the binary (compile-time) \
                 nor set as a runtime environment variable.",
            );

        let jwt_secret = env::var("BRAIN_JWT_SECRET")
            .unwrap_or_else(|_| psk.clone()); // fallback to PSK if not set

        // Parse "alias1=model1,alias2=model2" into a HashMap
        let model_aliases = env::var("BRAIN_MODEL_ALIASES")
            .unwrap_or_default()
            .split(',')
            .filter_map(|pair| {
                let mut parts = pair.splitn(2, '=');
                let alias = parts.next()?.trim().to_string();
                let model = parts.next()?.trim().to_string();
                if alias.is_empty() || model.is_empty() { None } else { Some((alias, model)) }
            })
            .collect::<HashMap<_, _>>();

        // Parse egress policy from environment
        let egress_domains: Vec<String> = env::var("BRAIN_EGRESS_DOMAINS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();

        let egress_policy = match env::var("BRAIN_EGRESS_MODE")
            .unwrap_or_else(|_| "deny_all".to_string())
            .as_str()
        {
            "allow_all" => EgressPolicy::AllowAll,
            "allowlist" => EgressPolicy::Allowlist(egress_domains),
            "denylist"  => EgressPolicy::Denylist(egress_domains),
            _           => EgressPolicy::DenyAll,
        };

        Self {
            psk,
            jwt_secret,
            db_path: env::var("BRAIN_DB")
                .unwrap_or_else(|_| format!("{}/brain.db", WORK_DIR)),
            addr: env::var("BRAIN_ADDR")
                .unwrap_or_else(|_| "127.0.0.1:9999".to_string()),
            downloads_dir: env::var("BRAIN_DOWNLOADS_DIR")
                .unwrap_or_else(|_| format!("{}/downloads", WORK_DIR)),
            worker_timeout_s: env::var("BRAIN_WORKER_TIMEOUT")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(120),
            queue_timeout_s: env::var("BRAIN_QUEUE_TIMEOUT")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(10),
            model_aliases,
            allow_psk_as_token: env::var("BRAIN_ALLOW_PSK_TOKEN")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true),

            // Telemetry
            metrics_enabled: env::var("BRAIN_METRICS_ENABLED")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true),
            telemetry_enabled: env::var("BRAIN_TELEMETRY_ENABLED")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(true),

            // Idempotency
            idempotency_ttl_s: env::var("BRAIN_IDEMPOTENCY_TTL")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(300),

            // Timeout budgets
            queue_wait_budget_s: env::var("BRAIN_QUEUE_WAIT_BUDGET")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(10),
            prefill_budget_s: env::var("BRAIN_PREFILL_BUDGET")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(30),
            decode_budget_s: env::var("BRAIN_DECODE_BUDGET")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(60),
            tool_dispatch_budget_s: env::var("BRAIN_TOOL_DISPATCH_BUDGET")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(15),
            default_context_window: env::var("BRAIN_DEFAULT_CTX")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(32768),

            // Pricing: "model=prompt_per_1m:completion_per_1m,model2=..."
            model_pricing: env::var("BRAIN_MODEL_PRICING")
                .unwrap_or_default()
                .split(',')
                .filter_map(|entry| {
                    let mut parts = entry.splitn(2, '=');
                    let model = parts.next()?.trim().to_string();
                    let rates = parts.next()?.trim();
                    let mut r = rates.splitn(2, ':');
                    let prompt_rate: f64 = r.next()?.trim().parse().ok()?;
                    let comp_rate: f64 = r.next()?.trim().parse().ok()?;
                    if model.is_empty() { None } else { Some((model, (prompt_rate, comp_rate))) }
                })
                .collect::<HashMap<_, _>>(),

            // Hosted tools
            searxng_url: env::var("BRAIN_SEARXNG_URL").ok().filter(|s| !s.trim().is_empty()),
            hosted_tool_max_rounds: env::var("BRAIN_HOSTED_TOOL_MAX_ROUNDS")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(4),

            egress_policy,

            stream_buffer_chunks: env::var("BRAIN_STREAM_BUFFER_CHUNKS")
                .ok().and_then(|v| v.parse().ok()).filter(|v: &usize| *v > 0).unwrap_or(1024),
        }
    }

    /// USD cost for a completed turn, given the resolved model name and
    /// worker-reported token counts. Models with no pricing entry cost $0
    /// (the default for self-hosted local workers).
    pub fn calc_cost_usd(&self, model: &str, prompt_tokens: u32, comp_tokens: u32) -> f64 {
        let (prompt_rate, comp_rate) = self.model_pricing.get(model).copied().unwrap_or((0.0, 0.0));
        (prompt_tokens as f64 / 1_000_000.0) * prompt_rate
            + (comp_tokens as f64 / 1_000_000.0) * comp_rate
    }

    /// Resolve a client-supplied model name to the real worker model name.
    /// Falls back to the original name if no alias matches (lenient resolution).
    /// Note: The worker registry also does a secondary lenient fallback —
    /// even if no alias is configured, any connected worker will serve the request.
    pub fn resolve_model<'a>(&'a self, client_model: &'a str) -> &'a str {
        let resolved = self.model_aliases
            .get(client_model)
            .map(|s| s.as_str())
            .unwrap_or(client_model);

        if resolved != client_model {
            log::info!("model alias: {} → {}", client_model, resolved);
        }

        resolved
    }
}

