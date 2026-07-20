// ── Hosted tools: web_search / web_fetch ────────────────────────────────────
//
// In the real Anthropic API, `web_search`/`web_fetch` are *server-side*
// tools: Claude decides to call one, Anthropic's own infrastructure runs it,
// and the result is folded back into the same turn before the client ever
// sees a response. Claude Code never round-trips for these.
//
// A local worker model has no such capability — from its point of view every
// tool call is "ask the caller to run this and hand me the result." So this
// module makes `brain` play the role Anthropic's servers play for the real
// API: it advertises `web_search`/`web_fetch` to the worker as ordinary
// function tools, and when the worker calls one, brain executes it itself
// (search via a self-hosted SearxNG instance, fetch via a plain HTTP GET +
// a light HTML-to-text pass), appends the result to the conversation, and
// asks the worker to continue — invisibly to Claude Code, which only ever
// sees the final turn.
//
// Limitation: if a single round mixes a hosted tool call (web_search) with
// a non-hosted one (e.g. Bash) in the same turn, we can't resolve the
// hosted one internally without also handing the non-hosted one back to
// Claude Code for execution — so a mixed round is treated as final and
// passed through as-is, same as a plain non-tool-call round, with whatever
// hosted-tool call sits unresolved among the tool_use blocks Claude Code
// receives. In practice models overwhelmingly call one tool at a time under
// tool_choice "auto", so this is an edge case, not the common path — but it
// is a real gap worth knowing about if you see a hosted tool call reach
// Claude Code unresolved.

use std::sync::Arc;
use std::net::{IpAddr, ToSocketAddrs};
use serde_json::Value;

use crate::core::config::{Config, EgressPolicy};
use crate::core::protocol::{
    Message, MessageContent, Tool, FunctionDef, ToolCall, ToolChoice, TaskBundle, FinishReason,
    WorkerMessage, ChunkMessage, DoneMessage,
};
use crate::api::router::Router;

use super::convert::accumulate_delta;

pub const WEB_SEARCH_TOOL_NAME: &str = "web_search";
pub const WEB_FETCH_TOOL_NAME:  &str = "web_fetch";

const MAX_SEARCH_RESULTS: usize = 5;
const MAX_FETCH_CHARS:    usize = 8_000; // characters, not bytes -- keeps one fetch from blowing the context window

fn is_hosted_tool_name(name: &str) -> bool {
    name == WEB_SEARCH_TOOL_NAME || name == WEB_FETCH_TOOL_NAME
}

/// Function-tool definitions handed to the worker in place of the real
/// Anthropic hosted-tool entries `ai-provider-converter` stripped out.
/// Anthropic's own request shape for hosted tools carries no input schema
/// (its servers already know it) — the worker needs one to call them as
/// ordinary function tools, so we supply one here.
pub fn hosted_tool_defs() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: WEB_SEARCH_TOOL_NAME.into(),
                description: Some(
                    "Search the web. Returns a numbered list of results, each with a title, URL, and short snippet.".into(),
                ),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "The search query" }
                    },
                    "required": ["query"]
                })),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: WEB_FETCH_TOOL_NAME.into(),
                description: Some(
                    "Fetch a URL and return its readable text content.".into(),
                ),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "The URL to fetch" }
                    },
                    "required": ["url"]
                })),
            },
        },
    ]
}

/// The round that finally didn't resolve entirely to hosted-tool calls —
/// either a clean end_turn, a length/content-filter stop, or a round mixing
/// in a non-hosted tool call (see module-level limitation note). Ready to be
/// handed to the normal streaming/blocking delivery path as if it had come
/// straight from a single worker turn.
pub struct HostedLoopOutcome {
    pub full_text:     String,
    pub tool_calls:    Vec<ToolCall>,
    pub prompt_tokens: u32,
    pub comp_tokens:   u32,
    pub finish_reason: FinishReason,
}

/// Parameters needed to issue one internal worker round. Grouped into a
/// struct rather than threaded through as loose arguments since every round
/// re-sends the same fixed request shape with only `messages` changing.
pub struct HostedLoopRequest {
    pub model:       String,
    pub max_tokens:  u32,
    pub temperature: f32,
    pub top_p:       Option<f32>,
    pub top_k:       Option<u32>,
    pub stop:        Option<Vec<String>>,
    pub tools:       Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub session_id:  String,
}

/// Runs the internal search-and-continue loop. Returns the final round's
/// outcome once the worker stops calling hosted tools (or the round mixes in
/// a non-hosted call), or an error if `cfg.hosted_tool_max_rounds` is
/// exhausted first, or the worker channel drops mid-loop.
pub async fn run_hosted_tool_loop(
    mut messages: Vec<Message>,
    req: HostedLoopRequest,
    worker_tx: flume::Sender<WorkerMessage>,
    worker_id: String,
    router: Arc<Router>,
    cfg: Arc<Config>,
    db: Arc<crate::db::memory::Database>,
) -> Result<HostedLoopOutcome, String> {
    let mut prompt_tokens_total = 0u32;
    let mut comp_tokens_total = 0u32;

    for round in 0..cfg.hosted_tool_max_rounds {
        let request_id = uuid::Uuid::new_v4().to_string();
        let task = TaskBundle {
            request_id: request_id.clone(),
            session_id: req.session_id.clone(),
            model: req.model.clone(),
            messages: messages.clone(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            stream: false,
            tools: req.tools.clone(),
            tool_choice: req.tool_choice.clone(),
            parallel_tool_calls: Some(true),
            stop: req.stop.clone(),
        };

        let (chunk_tx, chunk_rx) = flume::bounded::<ChunkMessage>(256);
        let (done_tx, done_rx) = flume::bounded::<DoneMessage>(1);
        router.register_pending(request_id.clone(), chunk_tx, done_tx, worker_tx.clone(), worker_id.clone());

        if worker_tx.send(WorkerMessage::Task(task)).is_err() {
            router.finish(&request_id);
            return Err("worker channel closed during hosted-tool round".into());
        }

        let round_result = collect_one_round(&chunk_rx, &done_rx, cfg.worker_timeout_s).await;
        router.finish(&request_id);
        let (full_text, tool_calls, prompt_tokens, comp_tokens, finish_reason) = round_result?;

        prompt_tokens_total += prompt_tokens;
        comp_tokens_total += comp_tokens;

        let all_hosted = !tool_calls.is_empty()
            && tool_calls.iter().all(|tc| is_hosted_tool_name(&tc.function.name));

        if !all_hosted {
            return Ok(HostedLoopOutcome {
                full_text,
                tool_calls,
                prompt_tokens: prompt_tokens_total,
                comp_tokens: comp_tokens_total,
                finish_reason,
            });
        }

        log::info!(
            "hosted tool round {round}: resolving {} call(s) ({:?})",
            tool_calls.len(),
            tool_calls.iter().map(|tc| tc.function.name.as_str()).collect::<Vec<_>>()
        );

        // Every call this round is one we can resolve ourselves. Append the
        // assistant's tool-call message, then one tool-result message per
        // call (matched by tool_call_id, same as any OpenAI-shaped
        // multi-tool turn), and go around again.
        messages.push(Message {
            role: "assistant".into(),
            content: MessageContent::Text(full_text),
            tool_calls: Some(tool_calls.clone()),
            tool_call_id: None,
        });

        for tc in &tool_calls {
            let result_text = execute_hosted_tool(tc, &cfg, &db).await;
            messages.push(Message {
                role: "tool".into(),
                content: MessageContent::Text(result_text),
                tool_calls: None,
                tool_call_id: Some(tc.id.clone()),
            });
        }
    }

    Err(format!(
        "hosted tool loop did not resolve within {} round(s)",
        cfg.hosted_tool_max_rounds
    ))
}

/// Drains one internal, non-streamed worker round to completion. Reuses
/// `accumulate_delta` — the same single-source-of-truth parser the real
/// streaming/blocking handlers use — so a tool call built up here is
/// identical to one built up on the normal client-facing path.
async fn collect_one_round(
    chunk_rx: &flume::Receiver<ChunkMessage>,
    done_rx: &flume::Receiver<DoneMessage>,
    timeout_s: u64,
) -> Result<(String, Vec<ToolCall>, u32, u32, FinishReason), String> {
    let timeout = std::time::Duration::from_secs(timeout_s);
    let mut last_activity = std::time::Instant::now();
    let mut full_text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut current_tool_call: Option<ToolCall> = None;
    let mut current_tool_index: Option<u64> = None;

    loop {
        match chunk_rx.try_recv() {
            Ok(chunk) => {
                last_activity = std::time::Instant::now();
                let _ = accumulate_delta(
                    &chunk.delta, &mut full_text, &mut tool_calls,
                    &mut current_tool_call, &mut current_tool_index,
                );
                continue;
            }
            Err(flume::TryRecvError::Disconnected) => {
                return Err("worker chunk channel disconnected mid hosted-tool round".into());
            }
            Err(flume::TryRecvError::Empty) => {}
        }

        if let Ok(done) = done_rx.try_recv() {
            if let Some(last) = current_tool_call.take() {
                tool_calls.push(last);
            }
            return Ok((full_text, tool_calls, done.prompt_tokens, done.comp_tokens, done.finish_reason));
        }

        if std::time::Instant::now().duration_since(last_activity) >= timeout {
            return Err("worker produced no output for too long during a hosted-tool round".into());
        }

        monoio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

async fn execute_hosted_tool(tc: &ToolCall, cfg: &Config, db: &crate::db::memory::Database) -> String {
    let args: Value = serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
    match tc.function.name.as_str() {
        WEB_SEARCH_TOOL_NAME => {
            let query = args.get("query").and_then(Value::as_str).unwrap_or_default();
            searxng_search(cfg, db, query)
        }
        WEB_FETCH_TOOL_NAME => {
            let url = args.get("url").and_then(Value::as_str).unwrap_or_default();
            fetch_url_text(url, cfg)
        }
        other => format!("error: \"{other}\" is not a hosted tool brain can execute"),
    }
}

/// NOTE: `ureq` is a blocking HTTP client, called inline the same way
/// brain's existing Google tokeninfo check already does in `api/auth.rs` —
/// it blocks the single monoio reactor thread for the round trip. That's an
/// existing tradeoff in this codebase, not a new one introduced here; if
/// hosted-tool traffic becomes significant, move this onto a dedicated
/// thread (`std::thread::spawn` + a channel back into the async task) rather
/// than calling it inline like this.
fn searxng_search(_cfg: &Config, db: &crate::db::memory::Database, query: &str) -> String {
    if query.trim().is_empty() {
        return "error: empty search query".into();
    }
    let url_setting = db.get_setting("searxng_url");
    let Some(base) = url_setting.as_deref().filter(|s| !s.trim().is_empty()) else {
        return "error: web_search is not configured on this brain instance (searxng_url is unset in db)".into();
    };
    let base = base.trim_end_matches('/');
    let url = format!("{base}/search?q={}&format=json", urlencoding::encode(query));

    let resp = match ureq::get(&url).call() {
        Ok(r) => r,
        Err(e) => return format!("error: search request to SearxNG failed: {e}"),
    };
    let body_text = match resp.into_body().read_to_string() {
        Ok(s) => s,
        Err(e) => return format!("error: could not read SearxNG response body: {e}"),
    };
    let body: Value = match serde_json::from_str(&body_text) {
        Ok(v) => v,
        Err(e) => return format!("error: SearxNG response was not valid JSON (is `search.formats: [json]` enabled in its settings.yml?): {e}"),
    };
    let results = body.get("results").and_then(Value::as_array).cloned().unwrap_or_default();
    if results.is_empty() {
        return "No results found.".into();
    }

    let mut out = String::new();
    for (i, r) in results.iter().take(MAX_SEARCH_RESULTS).enumerate() {
        let title = r.get("title").and_then(Value::as_str).unwrap_or("(no title)");
        let url = r.get("url").and_then(Value::as_str).unwrap_or("");
        let content = r.get("content").and_then(Value::as_str).unwrap_or("");
        out.push_str(&format!("{}. {title}\n{url}\n{content}\n\n", i + 1));
    }
    out
}

/// True if `host` is exactly `domain` or a subdomain of it.
/// "raw.githubusercontent.com" matches domain "githubusercontent.com" but
/// NOT domain "github.com" — list what you actually want fetchable.
fn host_matches_domain(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

fn domain_list_allows(host: &str, list: &[String]) -> bool {
    list.iter().any(|d| host_matches_domain(host, d))
}

/// Blocks the classic SSRF targets regardless of what the allow/deny list
/// says — metadata endpoints, loopback, link-local, and RFC1918 space have
/// no legitimate reason to be reachable from a "fetch this webpage" tool.
fn is_forbidden_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()          // covers 169.254.169.254 (cloud metadata)
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // unique local (fc00::/7)
                || (v6.segments()[0] & 0xffc0) == 0xfe80 // link-local
        }
    }
}

/// Full validation: scheme, domain policy, and a DNS-rebinding check (the
/// domain policy check happens on the *hostname string*, but an attacker
/// controlling DNS for an allowed domain could still point it at an internal
/// IP at request time — so we resolve and re-check the actual IP too).
fn validate_fetch_url(url_str: &str, cfg: &Config) -> Result<(), String> {
    let parsed = url::Url::parse(url_str).map_err(|e| format!("invalid URL: {e}"))?;

    match parsed.scheme() {
        "http" | "https" => {}
        other => return Err(format!("scheme \"{other}\" is not allowed (only http/https)")),
    }

    let host = parsed.host_str().ok_or("URL has no host")?.to_lowercase();

    // If someone passes an IP literal directly, check it before DNS even
    // enters the picture.
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_forbidden_ip(&ip) {
            return Err("target IP is in a private/loopback/link-local range".into());
        }
    }

    match &cfg.egress_policy {
        EgressPolicy::AllowAll => {}
        EgressPolicy::DenyAll => return Err("web_fetch egress is disabled on this brain instance (BRAIN_EGRESS_MODE=deny_all)".into()),
        EgressPolicy::Allowlist(list) => {
            if !domain_list_allows(&host, list) {
                return Err(format!("\"{host}\" is not on the egress allowlist"));
            }
        }
        EgressPolicy::Denylist(list) => {
            if domain_list_allows(&host, list) {
                return Err(format!("\"{host}\" is on the egress denylist"));
            }
        }
    }

    // DNS-rebinding guard: resolve and check the IPs actually behind the
    // hostname, not just the string itself.
    let port = parsed.port_or_known_default().unwrap_or(443);
    let addrs = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| format!("DNS resolution failed for \"{host}\": {e}"))?;

    let mut resolved_any = false;
    for addr in addrs {
        resolved_any = true;
        if is_forbidden_ip(&addr.ip()) {
            return Err(format!("\"{host}\" resolves to a private/internal IP ({}) — refusing to fetch", addr.ip()));
        }
    }
    if !resolved_any {
        return Err(format!("\"{host}\" did not resolve to any address"));
    }

    Ok(())
}

fn fetch_url_text(url: &str, cfg: &Config) -> String {
    if url.trim().is_empty() {
        return "error: empty url".into();
    }
    if let Err(e) = validate_fetch_url(url, cfg) {
        return format!("error: refused to fetch {url}: {e}");
    }
    let resp = match ureq::get(url).call() {
        Ok(r) => r,
        Err(e) => return format!("error: fetch of {url} failed: {e}"),
    };
    let body = match resp.into_body().read_to_string() {
        Ok(s) => s,
        Err(e) => return format!("error: could not read response body from {url}: {e}"),
    };

    let text = html_to_text(&body);
    if text.chars().count() > MAX_FETCH_CHARS {
        let truncated: String = text.chars().take(MAX_FETCH_CHARS).collect();
        format!("{truncated}\n\n[truncated — page was longer than {MAX_FETCH_CHARS} characters]")
    } else {
        text
    }
}

/// Deliberately minimal HTML→text reduction: strips tags and the contents of
/// `<script>`/`<style>` blocks, collapses whitespace. Not a real readability
/// pass — good enough to keep raw markup out of the model's context, not
/// good enough to compete with a proper article extractor. If fetched pages
/// come back too noisy in practice, this is the spot to swap in something
/// heavier (e.g. shelling out to a readability tool).
fn html_to_text(html: &str) -> String {
    fn starts_with_ci(haystack: &str, needle: &str) -> bool {
        let hb = haystack.as_bytes();
        let nb = needle.as_bytes();
        hb.len() >= nb.len() && hb[..nb.len()].eq_ignore_ascii_case(nb)
    }

    let mut out = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    let mut skip_depth: usize = 0;
    let mut i = 0usize;
    let bytes_len = html.len();

    while i < bytes_len {
        let rest = &html[i..];
        let ch = rest.chars().next().unwrap_or('\0');

        if ch == '<' {
            in_tag = true;
            if starts_with_ci(rest, "<script") || starts_with_ci(rest, "<style") {
                skip_depth += 1;
            } else if starts_with_ci(rest, "</script") || starts_with_ci(rest, "</style") {
                skip_depth = skip_depth.saturating_sub(1);
            }
            i += ch.len_utf8();
            continue;
        }
        if ch == '>' {
            in_tag = false;
            i += ch.len_utf8();
            continue;
        }
        if !in_tag && skip_depth == 0 {
            out.push(ch);
        }
        i += ch.len_utf8();
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── Final delivery ───────────────────────────────────────────────────────────
//
// By the time `run_hosted_tool_loop` returns, the whole final turn is already
// known in full — it came back from an internal, non-streamed worker round,
// not a live one we can just keep forwarding from. So these two functions
// synthesize the client-facing response from that already-known text/tool
// list, using the exact same SSE builders (streaming) and response structs
// (blocking) as the normal path in `streaming.rs`/`blocking.rs`, rather than
// hand-rolling a second response shape. From Claude Code's side this is
// indistinguishable from an ordinary turn — which is the point: the search
// loop is supposed to be invisible, same as it is on the real Anthropic API.

pub async fn deliver_streaming_result(
    mut stream: monoio::net::TcpStream,
    request_id: String,
    model: String,
    session_id: String,
    outcome: HostedLoopOutcome,
    router: Arc<Router>,
    db: Arc<crate::db::memory::Database>,
    turn_start: std::time::Instant,
) {
    use monoio::io::AsyncWriteRentExt;
    use serde_json::json;
    use ai_provider_converter::{
        message_start, content_block_start, content_block_delta, content_block_stop,
        message_delta, message_stop, AnthropicUsage,
    };

    let headers = "HTTP/1.1 200 OK\r\n\
                   Content-Type: text/event-stream\r\n\
                   Cache-Control: no-cache\r\n\
                   X-Accel-Buffering: no\r\n\
                   Connection: keep-alive\r\n\
                   \r\n";
    if stream.write_all(headers.as_bytes()).await.0.is_err() {
        router.cancel(&request_id);
        return;
    }

    if stream.write_all(message_start(&format!("msg_{request_id}"), &model).to_vec()).await.0.is_err() {
        router.cancel(&request_id);
        return;
    }

    let stop_reason = if !outcome.tool_calls.is_empty() {
        "tool_use".to_string()
    } else {
        ai_provider_converter::openai_finish_reason_to_anthropic_stop_reason(outcome.finish_reason.as_str())
    };

    // Text block first (even if empty — some models return a bare tool call
    // with no preceding text, in which case this is an empty block, same as
    // the normal path does for a tool-call-only turn).
    let _ = stream.write_all(content_block_start(0, json!({"type": "text", "text": ""})).to_vec()).await;
    if !outcome.full_text.is_empty() {
        let _ = stream.write_all(
            content_block_delta(0, json!({"type": "text_delta", "text": outcome.full_text})).to_vec()
        ).await;
    }
    let _ = stream.write_all(content_block_stop(0).to_vec()).await;

    for (i, tc) in outcome.tool_calls.iter().enumerate() {
        let block = i + 1;
        let _ = stream.write_all(content_block_start(
            block,
            json!({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": {}}),
        ).to_vec()).await;
        let _ = stream.write_all(content_block_delta(
            block,
            json!({"type": "input_json_delta", "partial_json": tc.function.arguments}),
        ).to_vec()).await;
        let _ = stream.write_all(content_block_stop(block).to_vec()).await;
    }

    let usage = AnthropicUsage {
        input_tokens: outcome.prompt_tokens as u64,
        output_tokens: outcome.comp_tokens as u64,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
    };
    let _ = stream.write_all(message_delta(Some(&stop_reason), usage).to_vec()).await;
    let _ = stream.write_all(message_stop().to_vec()).await;

    let elapsed = turn_start.elapsed();
    let _ = db.record_turn_metrics(
        &session_id, &request_id, "brain", None, &model,
        elapsed.as_millis() as u32,
        if outcome.comp_tokens > 0 { outcome.comp_tokens as f32 / elapsed.as_secs_f32().max(0.001) } else { 0.0 },
        outcome.prompt_tokens, outcome.comp_tokens, 0, 0.0,
    );

    router.finish(&request_id);
}

pub async fn deliver_blocking_result(
    mut stream: monoio::net::TcpStream,
    request_id: String,
    model: String,
    session_id: String,
    outcome: HostedLoopOutcome,
    db: Arc<crate::db::memory::Database>,
    turn_start: std::time::Instant,
) {
    use monoio::io::AsyncWriteRentExt;
    use serde_json::json;
    use super::types::{AnthropicResponse, AnthropicContentBlock, AnthropicUsage};

    let (content, stop_reason) = if !outcome.tool_calls.is_empty() {
        let mut blocks = Vec::new();
        if !outcome.full_text.is_empty() {
            blocks.push(AnthropicContentBlock {
                kind: "text".into(), text: Some(outcome.full_text.clone()), id: None, name: None, input: None,
            });
        }
        blocks.extend(outcome.tool_calls.clone().into_iter().map(|tc| {
            let input = serde_json::from_str::<Value>(&tc.function.arguments).unwrap_or(json!({}));
            AnthropicContentBlock {
                kind: "tool_use".into(), text: None, id: Some(tc.id), name: Some(tc.function.name), input: Some(input),
            }
        }));
        (blocks, Some("tool_use".to_string()))
    } else {
        let mapped = ai_provider_converter::openai_finish_reason_to_anthropic_stop_reason(outcome.finish_reason.as_str());
        (
            vec![AnthropicContentBlock {
                kind: "text".into(), text: Some(outcome.full_text.clone()), id: None, name: None, input: None,
            }],
            Some(mapped),
        )
    };

    let response = AnthropicResponse {
        id: format!("msg_{request_id}"),
        kind: "message".into(),
        role: "assistant".into(),
        content,
        model: model.clone(),
        stop_reason,
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: outcome.prompt_tokens as u64,
            output_tokens: outcome.comp_tokens as u64,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        },
    };

    let body = serde_json::to_vec(&response).unwrap_or_default();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nx-request-id: {request_id}\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), String::from_utf8_lossy(&body)
    );
    let _ = stream.write_all(resp.into_bytes()).await;

    let elapsed = turn_start.elapsed();
    let _ = db.record_turn_metrics(
        &session_id, &request_id, "brain", None, &model,
        elapsed.as_millis() as u32,
        if outcome.comp_tokens > 0 { outcome.comp_tokens as f32 / elapsed.as_secs_f32().max(0.001) } else { 0.0 },
        outcome.prompt_tokens, outcome.comp_tokens, 0, 0.0,
    );
}