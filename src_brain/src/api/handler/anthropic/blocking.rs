use std::sync::Arc;
use std::time::Instant;
use monoio::io::AsyncWriteRentExt;
use monoio::net::TcpStream;
use serde_json::json;

use crate::core::config::Config;
use crate::core::idempotency::IdempotencyStore;
use crate::core::protocol::{FinishReason, Message, ToolCall};
use crate::db::memory::Database;
use crate::api::router::Router;
use crate::metrics::Metrics;

use super::convert::{accumulate_delta, parse_tool_calls_from_text};
use super::types::{AnthropicResponse, AnthropicContentBlock, AnthropicUsage};
use super::super::common::{now_secs, send_json_error};

// ── Anthropic blocking handler ──────────────────────────────────────────────

pub async fn handle_anthropic_blocking(
    mut stream:  TcpStream,
    request_id:  String,
    model:       String,
    session_id:  String,
    chunk_rx:    flume::Receiver<crate::core::protocol::ChunkMessage>,
    done_rx:     flume::Receiver<crate::core::protocol::DoneMessage>,
    router:      Arc<Router>,
    cfg:         Arc<Config>,
    metrics:     Arc<Metrics>,
    turn_start:  Instant,
    _user_id:     Option<i64>,
    _last_user_msg: Option<Message>,
    idempotency_key: Option<String>,
    idempotency: Arc<IdempotencyStore>,
    db:          Arc<Database>,
) {
    let timeout = std::time::Duration::from_secs(cfg.worker_timeout_s);
    let mut last_activity = std::time::Instant::now();

    let mut full_text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut current_tool_call: Option<ToolCall> = None;
    let mut current_tool_index: Option<u64> = None;
    let mut prompt_tokens = 0u32;
    let mut comp_tokens   = 0u32;
    let mut finish_reason = FinishReason::Stop;

    loop {
        if let Ok(done) = done_rx.try_recv() {
            prompt_tokens = done.prompt_tokens;
            comp_tokens   = done.comp_tokens;
            finish_reason = done.finish_reason;

            if cfg.telemetry_enabled {
                let elapsed_ms = turn_start.elapsed().as_millis() as u64;
                metrics.record_ttft("brain", &model, elapsed_ms);
                metrics.record_tokens("brain", &model, prompt_tokens as u64, comp_tokens as u64);
            }

            if let Some(last) = current_tool_call.take() {
                tool_calls.push(last);
            }
            break;
        }

        if std::time::Instant::now().duration_since(last_activity) >= timeout {
            send_json_error(&mut stream, 504, "gateway_timeout", "Worker produced no output for too long").await;
            router.cancel(&request_id);
            return;
        }

        monoio::time::sleep(std::time::Duration::from_millis(1)).await;

        match chunk_rx.try_recv() {
            Ok(chunk) => {
                last_activity = std::time::Instant::now();
                
                let (_text, _tool_events) = accumulate_delta(&chunk.delta, &mut full_text, &mut tool_calls, &mut current_tool_call, &mut current_tool_index);
            }
            Err(flume::TryRecvError::Empty)      => continue,
            Err(flume::TryRecvError::Disconnected) => {
                router.cancel(&request_id);
                break;
            }
        }
    }

    // Only fall back to text-pattern parsing if the worker didn't already
    // give us structured tool_calls.
    if tool_calls.is_empty() {
        if let Some(parsed) = parse_tool_calls_from_text(&full_text) {
            tool_calls = parsed;
        }
    }


    let _ = db.record_turn_metrics(
        &session_id, &request_id, "brain",
        None, // user_id
        &model,
        turn_start.elapsed().as_millis() as u32,
        if comp_tokens > 0 { comp_tokens as f32 / turn_start.elapsed().as_secs_f32().max(0.001) } else { 0.0 },
        prompt_tokens, comp_tokens,
        0, // cache_hit_tokens
        0.0, // cost_usd
    );

    // Build Anthropic response
    let (content, stop_reason) = if !tool_calls.is_empty() {
        let blocks: Vec<AnthropicContentBlock> = tool_calls.into_iter().map(|tc| {
            let input = serde_json::from_str::<serde_json::Value>(&tc.function.arguments).unwrap_or(json!({}));
            AnthropicContentBlock {
                kind: "tool_use".into(),
                text: None,
                id: Some(tc.id),
                name: Some(tc.function.name),
                input: Some(input),
            }
        }).collect();
        (blocks, Some("tool_use".to_string()))
    } else {
        // Map the worker's actual finish_reason (stop/length/content_filter/...)
        // through the shared library mapper instead of always claiming
        // "end_turn" — a response cut short by max_tokens or stopped by a
        // content filter should say so, not look like a clean completion.
        let mapped = ai_provider_converter::openai_finish_reason_to_anthropic_stop_reason(finish_reason.as_str());
        (
            vec![AnthropicContentBlock {
                kind: "text".into(),
                text: Some(full_text),
                id: None,
                name: None,
                input: None,
            }],
            Some(mapped),
        )
    };

    let response = AnthropicResponse {
        id:      format!("msg_{}", request_id),
        kind:    "message".into(),
        role:    "assistant".into(),
        content,
        model,
        stop_reason,
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens:  prompt_tokens as u64,
            output_tokens: comp_tokens as u64,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        },
    };

    let body = serde_json::to_vec(&response).unwrap_or_default();

    // Save to idempotency store if a key was provided
    if let Some(key) = idempotency_key {
        let idem_resp = crate::core::idempotency::IdempotentResponse {
            status: 200,
            content_type: "application/json".into(),
            body: body.clone(),
            created_at: now_secs() as i64,
        };
        idempotency.put(&key, idem_resp);
        let _ = db.put_idempotent_response(&key, 200, &body, "application/json");
    }

    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nx-request-id: {}\r\nContent-Length: {}\r\n\r\n{}",
        request_id, body.len(), String::from_utf8_lossy(&body)
    );
    let _ = stream.write_all(resp.into_bytes()).await;
}
