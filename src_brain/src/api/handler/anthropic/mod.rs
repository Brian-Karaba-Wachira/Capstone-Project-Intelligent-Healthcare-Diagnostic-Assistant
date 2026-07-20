// ── Anthropic Messages API handler (§Dual API support) ─────────────────────
// Auto-detected via `anthropic-version` header or `x-api-key`.
// Converts Anthropic Messages API ↔ internal OpenAI-compatible protocol
// so lmodel/llama.cpp workers see a unified TaskBundle regardless.
//
// Anthropic API spec: https://docs.anthropic.com/en/api/messages
//
// Split out of a single 846-line file into this module folder:
//   types.rs      — request/response/SSE struct definitions
//   convert.rs     — Anthropic <-> OpenAI message/tool conversion
//   streaming.rs   — SSE streaming response handler
//   blocking.rs    — non-streaming response handler
// `handle_anthropic` below is the only symbol re-exported to the rest of
// the crate — everything else is module-private.

mod types;
mod convert;
mod streaming;
mod blocking;
mod hosted_tools;

use std::sync::Arc;
use std::time::Instant;
use bytes::BytesMut;
use monoio::net::TcpStream;
use uuid::Uuid;

use crate::core::config::Config;
use crate::core::idempotency::IdempotencyStore;
use crate::db::memory::Database;
use crate::worker::registry::WorkerRegistry;
use crate::api::router::Router;
use crate::metrics::Metrics;

use super::common::{read_body, send_json_error};
use convert::{translate_anthropic_request, anthropic_models_response};
use streaming::handle_anthropic_streaming;
use blocking::handle_anthropic_blocking;
use types::{AnthropicRequest, required_max_tokens, extract_stop_sequences, extract_top_k};

// ── Main handler ────────────────────────────────────────────────────────────

pub async fn handle_anthropic(
    mut stream:     TcpStream,
    path:           String,
    body_buf:       BytesMut,
    content_length: usize,
    session_id:     Option<String>,
    registry:       Arc<WorkerRegistry>,
    router:         Arc<Router>,
    db:             Arc<Database>,
    cfg:            Arc<Config>,
    metrics:        Arc<Metrics>,
    idempotency:    Arc<IdempotencyStore>,


    idempotency_key: Option<String>,
    user_id:        Option<i64>,
) {
    // ── GET /v1/models (Anthropic format) ──────────────────────────────────
    if path == "/v1/models" || path == "/v1/models?limit=100" {
        let models = anthropic_models_response(registry.active_models());
        let body = serde_json::to_vec(&models).unwrap_or_default();
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(), String::from_utf8_lossy(&body)
        );
        use monoio::io::AsyncWriteRentExt;
        let _ = stream.write_all(resp.into_bytes()).await;
        return;
    }

    // ── POST /messages ─────────────────────────────────────────────────
    if !path.contains("messages") {
        send_json_error(&mut stream, 404, "not_found", "Anthropic endpoint not found").await;
        return;
    }

    let body = match read_body(&mut stream, body_buf, content_length).await {
        Ok(b) => b,
        Err(_) => { send_json_error(&mut stream, 400, "bad_request", "bad body").await; return; }
    };

    let req: AnthropicRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            send_json_error(&mut stream, 400, "invalid_json",
                            &format!("Could not parse Anthropic request: {}", e)).await;
            return;
        }
    };

    let resolved_model = cfg.resolve_model(&req.model).to_string();

    // Convert Anthropic → OpenAI-compatible shape via ai-provider-converter,
    // then adapt that into brain's internal Message/Tool/ToolChoice types.
    // This is the single translation call for this handler now — no more
    // hand-rolled message/tool/tool_choice conversion living in convert.rs.
    let translated = match translate_anthropic_request(&req, &resolved_model) {
        Ok(t) => t,
        Err(e) => {
            send_json_error(&mut stream, 400, "invalid_request",
                            &format!("Could not translate Anthropic request: {}", e)).await;
            return;
        }
    };

    let wants_hosted_tools = !translated.hosted_tools.is_empty();
    if wants_hosted_tools {
        let names: Vec<&str> = translated.hosted_tools.iter().map(|t| t.name.as_str()).collect();
        log::info!(
            "client requested hosted tool(s) {:?} for model={} — resolving via brain's own \
             web_search/web_fetch execution instead of the worker",
            names, resolved_model
        );
    }

    let messages = translated.messages;
    let mut tools = translated.tools;
    let tool_choice = translated.tool_choice;

    if wants_hosted_tools {
        let mut defs = hosted_tools::hosted_tool_defs();
        defs.extend(tools.take().unwrap_or_default());
        tools = Some(defs);
    }

    // ── Pick worker FIRST ────────────────────────────────────────────────────
    // FIX (was BUG): The old code used pick_best() which returns (Sender, max_context)
    // — it did NOT capture the actual model name used when lenient fallback occurred.
    // If no worker matched the requested model, the task bundle still had
    // `model: resolved_model` (client's name) but the worker was running something
    // different, causing the worker to potentially reject or misroute the task.
    //
    // Fix: use pick_best_lenient() which returns (Sender, actual_model_name, max_context)
    // — the same 3-tuple that the OpenAI chat handler already uses. This ensures the
    // task bundle carries the model name the worker actually registered with.
    //
    // cfg.default_context_window is only a fallback for a worker reporting 0.
    let (worker_tx, worker_id, actual_model, worker_max_context) = match registry.pick_best_lenient(&resolved_model) {
        Some(result) => result,
        None => {
            send_json_error(&mut stream, 503, "no_worker_available",
                            "No workers are currently connected.").await;
            return;
        }
    };

    let routed_model = if actual_model != resolved_model {
        log::info!("anthropic lenient routing: requested={resolved_model} → actual={actual_model}");
        actual_model.clone()
    } else {
        resolved_model.clone()
    };

    let effective_context = if worker_max_context > 0 {
        worker_max_context
    } else {
        log::warn!(
            "worker for model={} reported max_context=0 — falling back to cfg.default_context_window={}",
            resolved_model, cfg.default_context_window
        );
        cfg.default_context_window
    };

    // ── Context-window pre-flight ─────────────────────────────────────────
    // Two separate checks, on purpose:
    //
    // 1. `estimated_prompt_tokens + requested_max_tokens > effective_context`
    //    is NOT enforced. Claude Code's `max_tokens` is a ceiling, not a
    //    real usage forecast (it routinely sends 32000 regardless of how
    //    much output a turn actually needs), so summing it straight against
    //    the prompt produced false-positive rejections on perfectly fine
    //    requests. Left as a warning only.
    //
    // 2. `estimated_prompt_tokens` ALONE exceeding `effective_context` IS
    //    enforced (hard reject). Unlike (1), this has no dependency on the
    //    client's optimistic `max_tokens` ceiling — if the prompt itself
    //    doesn't fit, the request cannot possibly succeed no matter what
    //    output length is requested, so there's no false-positive risk.
    //    Previously this case was left to llama.cpp to reject after a full
    //    round trip through the WS tunnel instead of failing fast here with
    //    a message that actually tells the client what happened.
    let requested_max_tokens = required_max_tokens(&req);
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
            "request_id pending: estimated ~{estimated_prompt_tokens} prompt + {requested_max_tokens} \
             requested output tokens may exceed worker '{actual_model}'s context window \
             ({effective_context} tokens) — forwarding anyway (output-ceiling gating disabled), \
             llama.cpp will reject it directly if it's actually too big"
        );
    }

    let session_id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());


    let request_id = Uuid::new_v4().to_string();
    let turn_start = Instant::now();

    if cfg.telemetry_enabled {
        metrics.record_request("brain", &resolved_model, 200);
    }

    // Keep the last inbound message around for history persistence without
    // re-storing the entire context on every turn.
    let last_user_msg = messages.last().cloned();

    if wants_hosted_tools {
        let hosted_req = hosted_tools::HostedLoopRequest {
            model: routed_model.clone(),
            max_tokens: required_max_tokens(&req),
            temperature: req.temperature.unwrap_or(1.0) as f32,
            top_p: req.top_p.map(|v| v as f32),
            top_k: extract_top_k(&req),
            stop: extract_stop_sequences(&req),
            tools,
            tool_choice,
            session_id: session_id.clone(),
        };

        let outcome = match hosted_tools::run_hosted_tool_loop(
            messages, hosted_req, worker_tx.clone(), worker_id.clone(), router.clone(), cfg.clone(), db.clone(),
        ).await {
            Ok(o) => o,
            Err(e) => {
                log::error!("hosted tool loop failed for request_id={request_id}: {e}");
                send_json_error(&mut stream, 502, "hosted_tool_error", &e).await;
                return;
            }
        };

        if req.wants_stream() {
            hosted_tools::deliver_streaming_result(
                stream, request_id, routed_model, session_id, outcome, router.clone(), db, turn_start,
            ).await;
        } else {
            hosted_tools::deliver_blocking_result(
                stream, request_id, routed_model, session_id, outcome, db, turn_start,
            ).await;
        }
        return;
    }

    let task = crate::core::protocol::TaskBundle {
        request_id: request_id.clone(),
        session_id: session_id.clone(),
        model:      routed_model.clone(),
        messages,
        max_tokens: required_max_tokens(&req),
        temperature: req.temperature.unwrap_or(1.0) as f32,
        top_p:       req.top_p.map(|v| v as f32),
        top_k:       extract_top_k(&req),
        stream:      req.wants_stream(),
        tools,
        tool_choice,
        parallel_tool_calls: Some(true),
        // FIX (was BUG): this was hardcoded `None`, so `stop_sequences` sent
        // by Claude Code never reached the worker. Without a stop sequence,
        // a local model with no clean turn-boundary token can run past the
        // end of its assistant turn — hallucinating a fake next user message
        // and "answering" it, which is indistinguishable from "the model is
        // out of context." `stop_sequences` has no OpenAI equivalent, so the
        // library's translation leaves it in `req.extra` — pulled back out
        // via `extract_stop_sequences` rather than being lost.
        stop: extract_stop_sequences(&req),
    };

    // Bounded channel: prevents memory accumulation for dropped clients.
    // Capacity is configurable via BRAIN_STREAM_BUFFER_CHUNKS (default 1024).
    let (chunk_tx, chunk_rx) = flume::bounded::<crate::core::protocol::ChunkMessage>(cfg.stream_buffer_chunks);
    let (done_tx,  done_rx)  = flume::bounded::<crate::core::protocol::DoneMessage>(1);
    router.register_pending(request_id.clone(), chunk_tx, done_tx, worker_tx.clone(), worker_id.clone());

    if worker_tx.send(crate::core::protocol::WorkerMessage::Task(task)).is_err() {
        router.finish(&request_id);
        send_json_error(&mut stream, 503, "worker_unavailable", "Worker channel closed").await;
        return;
    }

    if req.wants_stream() {
        handle_anthropic_streaming(
            stream, request_id, routed_model, session_id,
            chunk_rx, done_rx, router.clone(), cfg, metrics, turn_start,
            user_id, last_user_msg, db,
        ).await;
    } else {
        handle_anthropic_blocking(
            stream, request_id, routed_model, session_id,
            chunk_rx, done_rx, router.clone(), cfg, metrics, turn_start,
            user_id, last_user_msg, idempotency_key, idempotency, db,
        ).await;
    }
}
