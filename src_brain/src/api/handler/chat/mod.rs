// ── OpenAI-compatible /v1/chat/completions handler ──────────────────────────
// Split out of a single 770-line chat.rs into this module folder, mirroring
// the anthropic/ split:
//   types.rs      — request/response struct definitions
//   convert.rs     — inbound normalization + outbound tool-call parsing
//   streaming.rs   — SSE streaming response handler
//   blocking.rs    — non-streaming response handler
// `handle_chat` below is the only symbol re-exported to the rest of the
// crate. `ModelObject`/`ModelsResponse` are re-exported flat (chat::X) since
// api/mod.rs already imports them at that path.

mod types;
mod convert;
mod streaming;
mod blocking;

pub use types::{ModelObject, ModelsResponse};

use std::sync::Arc;
use std::time::Instant;
use bytes::BytesMut;
use monoio::net::TcpStream;
use uuid::Uuid;

use crate::core::config::Config;
use crate::core::idempotency::IdempotencyStore;
use crate::core::protocol::TaskBundle;
use crate::db::memory::Database;
use crate::worker::registry::WorkerRegistry;
use crate::api::router::Router;
use crate::metrics::Metrics;

use super::common::{read_body, send_json_error};
use convert::{legacy_functions_to_tools, legacy_function_call_to_tool_choice};
use streaming::handle_streaming;
use blocking::handle_blocking;
use types::ChatRequest;
use crate::core::protocol::{message_from_openai_value, openai_tool_json_to_protocol_tool, openai_tool_choice_json_to_protocol, stop_sequences_from_value};

pub async fn handle_chat(
    mut stream:     TcpStream,
    body_buf:       BytesMut,
    content_length: usize,
    session_id:     Option<String>,
    user_id:        Option<i64>,
    idempotency_key: Option<String>,
    registry:       Arc<WorkerRegistry>,
    router:         Arc<Router>,
    db:             Arc<Database>,
    cfg:            Arc<Config>,
    metrics:        Arc<Metrics>,
    idempotency:    Arc<IdempotencyStore>,


) {
    // ── Read + parse body ──────────────────────────────────────────────────
    let body = match read_body(&mut stream, body_buf, content_length).await {
        Ok(b) => b,
        Err(_) => {
            send_json_error(&mut stream, 400, "bad_request", "Failed to read request body").await;
            return;
        }
    };

    let mut req: ChatRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            send_json_error(&mut stream, 400, "invalid_json",
                            &format!("Could not parse request body: {}", e)).await;
            return;
        }
    };

    if req.messages.is_empty() {
        send_json_error(&mut stream, 400, "invalid_request", "messages must not be empty").await;
        return;
    }

    // ── Normalize legacy functions → tools (Value-level, same shape the
    // Anthropic path's tools end up in before the shared adapters run) ──────
    if req.tools.is_none() {
        if let Some(fns) = req.extra.get("functions") {
            req.tools = legacy_functions_to_tools(fns);
        }
    }
    if req.tool_choice.is_none() {
        if let Some(fc) = req.extra.get("function_call") {
            req.tool_choice = Some(legacy_function_call_to_tool_choice(fc));
        }
    }

    // ── Model routing ───────────────────────────────────────────────────────
    let resolved_model = cfg.resolve_model(&req.model).to_string();

    // ── Convert inbound messages/tools/tool_choice → protocol types ────────
    // Uses the exact same adapters the Anthropic handler uses on its
    // (library-)translated output, so a request ends up identically shaped
    // for the worker regardless of which API dialect the client spoke.
    let messages: Vec<crate::core::protocol::Message> =
        req.messages.iter().map(message_from_openai_value).collect();

    let tools = req.tools.as_ref().map(|tools| {
        tools.iter().filter_map(openai_tool_json_to_protocol_tool).collect::<Vec<_>>()
    }).filter(|t| !t.is_empty());

    let tool_choice = req.tool_choice.as_ref().map(openai_tool_choice_json_to_protocol);

    // ── Pick worker FIRST ────────────────────────────────────────────────────
    // FIX (was BUG): this used to truncate the message list BEFORE a worker
    // was even chosen, against a hardcoded `32000` constant that had nothing
    // to do with any real worker's context window. It also didn't compile
    // against the current registry.rs, which now returns the worker's
    // self-reported `max_context` (3rd tuple element) — the old 2-element
    // destructure here was a straight arity mismatch.
    //
    // We now pick the worker first, read back what IT says its context
    // window is (from `RegisterMsg.max_context`, i.e. whatever the worker
    // was actually launched with — 32k, 256k, whatever), and truncate
    // against that real number. `cfg.default_context_window` is now only a
    // fallback for a worker that reports 0.
    let (worker_tx, worker_id, actual_model, worker_max_context) = match registry.pick_best_lenient(&resolved_model) {
        Some(result) => result,
        None => {
            send_json_error(&mut stream, 503, "no_worker_available",
                            "No workers are currently connected. Start a Colab worker.").await;
            return;
        }
    };

    let routed_model = if actual_model != resolved_model {
        log::info!("lenient routing: requested={resolved_model} → actual={actual_model}");
        actual_model.clone()
    } else {
        resolved_model.clone()
    };

    let effective_context = if worker_max_context > 0 {
        worker_max_context
    } else {
        log::warn!(
            "worker for model={} reported max_context=0 — falling back to cfg.default_context_window={}",
            routed_model, cfg.default_context_window
        );
        cfg.default_context_window
    };

    // ── Context-window pre-flight ─────────────────────────────────────────
    // See matching comment in anthropic/mod.rs. Summary: the combined
    // prompt+max_tokens check stays a warning-only (max_tokens is a client
    // ceiling, not a forecast), but the prompt-alone check is now enforced,
    // since a too-big prompt cannot succeed regardless of requested output
    // length and there's no false-positive risk in gating on it.
    let requested_max_tokens = req.max_tokens.unwrap_or(4096);
    let estimated_prompt_tokens =
        crate::core::protocol::estimate_prompt_tokens(&messages, tools.as_ref());

    if estimated_prompt_tokens > effective_context {
        log::warn!(
            "rejecting request: estimated ~{estimated_prompt_tokens} prompt tokens alone exceed \
             worker '{actual_model}'s context window ({effective_context} tokens) — refusing \
             to forward, this cannot succeed regardless of requested output length"
        );
        send_json_error(
            &mut stream,
            413,
            "context_length_exceeded",
            &format!(
                "This request's prompt is approximately {estimated_prompt_tokens} tokens, which \
                 exceeds the {effective_context}-token context window of worker '{actual_model}'. \
                 Start a new session, compact/summarize the conversation, or restart the worker \
                 with a larger --ctx-size."
            ),
        ).await;
        return;
    } else if estimated_prompt_tokens + requested_max_tokens > effective_context {
        log::warn!(
            "estimated ~{estimated_prompt_tokens} prompt + {requested_max_tokens} requested output \
             tokens may exceed worker '{actual_model}'s context window ({effective_context} tokens) \
             — forwarding anyway (output-ceiling gating disabled)"
        );
    }

    let session_id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());


    let request_id  = Uuid::new_v4().to_string();

    // ── Telemetry: record request start (§9.3) ───────────────────────────────
    let turn_start = Instant::now();
    if cfg.telemetry_enabled {
        metrics.record_request("brain", &resolved_model, 200);
    }

    // ── Build TaskBundle ─────────────────────────────────────────────────────
    let task = TaskBundle {
        request_id:          request_id.clone(),
        session_id:          session_id.clone(),
        model:               routed_model.clone(),
        messages:            messages.clone(),
        max_tokens:          req.max_tokens.unwrap_or(4096),
        temperature:         req.temperature.unwrap_or(0.7) as f32,
        top_p:               req.top_p.map(|v| v as f32),
        top_k:               None,
        stream:              req.stream,
        tools,
        tool_choice,
        parallel_tool_calls: req.parallel_tool_calls,
        stop:                req.stop.as_ref().and_then(stop_sequences_from_value),
    };

    // Register this request in the router so chunks can find it.
    // Bounded channel (MEM-1 fix): if the client disconnects and we stop
    // draining, the worker will back-pressure after the buffer fills rather
    // than accumulating the entire response in memory. Capacity is
    // configurable via BRAIN_STREAM_BUFFER_CHUNKS (default 1024).
    let (chunk_tx, chunk_rx) = flume::bounded::<crate::core::protocol::ChunkMessage>(cfg.stream_buffer_chunks);
    let (done_tx,  done_rx)  = flume::bounded::<crate::core::protocol::DoneMessage>(1);
    router.register_pending(request_id.clone(), chunk_tx, done_tx, worker_tx.clone(), worker_id.clone());

    if worker_tx.send(crate::core::protocol::WorkerMessage::Task(task)).is_err() {
        router.finish(&request_id);
        send_json_error(&mut stream, 503, "worker_unavailable", "Worker channel closed — it may have disconnected").await;
        return;
    }

    // ── Stream or collect response ────────────────────────────────────────────
    let last_user_msg = messages.last().cloned();
    if req.stream {
        handle_streaming(
            stream, request_id, routed_model,
            chunk_rx, done_rx, router, db, session_id.clone(), user_id,
            last_user_msg, cfg, metrics, turn_start,
        ).await;
    } else {
        handle_blocking(
            stream, request_id, routed_model,
            chunk_rx, done_rx, router, db, session_id, user_id,
            last_user_msg, cfg, metrics, turn_start,
            idempotency_key, idempotency
        ).await;
    }
}
