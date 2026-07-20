use std::sync::Arc;
use std::time::Instant;
use monoio::io::AsyncWriteRentExt;
use monoio::net::TcpStream;
use uuid::Uuid;

use crate::core::config::Config;
use crate::core::idempotency::IdempotencyStore;
use crate::core::protocol::{FinishReason, ToolCall, ToolCallFunction};
use crate::db::memory::Database;
use crate::api::router::Router;
use crate::metrics::Metrics;

use super::super::common::{build_error_response, now_secs, send_raw_json};
use super::convert::parse_tool_calls_from_response;
use super::types::{AssistantMessage, ChatResponse, Choice, Usage};

// ─────────────────────────────────────────────────────────────────────────────
// Blocking (non-streaming) response — collect chunks, build envelope
// ─────────────────────────────────────────────────────────────────────────────

pub async fn handle_blocking(
    mut stream:  TcpStream,
    request_id:  String,
    model:       String,
    chunk_rx:    flume::Receiver<crate::core::protocol::ChunkMessage>,
    done_rx:     flume::Receiver<crate::core::protocol::DoneMessage>,
    router:      Arc<Router>,
    db:          Arc<Database>,
    session_id:  String,
    _user_id:     Option<i64>,
    _last_user_msg: Option<crate::core::protocol::Message>,
    cfg:          Arc<Config>,
    metrics:      Arc<Metrics>,
    turn_start:   Instant,
    idempotency_key: Option<String>,
    idempotency:  Arc<IdempotencyStore>,
) {
    let timeout  = std::time::Duration::from_secs(cfg.worker_timeout_s);
    let mut last_activity = std::time::Instant::now();

    let mut full_response = String::new();
    let mut finish_reason = FinishReason::Stop;
    let mut prompt_tokens = 0u32;
    let mut comp_tokens   = 0u32;
    let mut accumulated_tool_calls: Vec<ToolCall> = Vec::new();
    let mut current_tool_call: Option<ToolCall> = None;

    // TTFT tracking for blocking responses
    let mut first_token = true;

    loop {
        if let Ok(done) = done_rx.try_recv() {
            finish_reason = done.finish_reason;
            prompt_tokens = done.prompt_tokens;
            comp_tokens   = done.comp_tokens;

            // Telemetry (§9.3)
            if cfg.telemetry_enabled {
                let elapsed_ms = turn_start.elapsed().as_millis() as u64;
                if first_token { metrics.record_ttft("brain", &model, elapsed_ms); }
                if comp_tokens > 0 {
                    let elapsed_s = turn_start.elapsed().as_secs_f32().max(0.001);
                    let tps = comp_tokens as f32 / elapsed_s;
                    metrics.record_tps("brain", &model, tps);
                }
                metrics.record_tokens("brain", &model, prompt_tokens as u64, comp_tokens as u64);
                // Persist telemetry to DB
                let _ = db.record_turn_metrics(
                    &session_id, &request_id, "brain",
                    None, // user_id
                    &model,
                    elapsed_ms as u32,
                    if comp_tokens > 0 { comp_tokens as f32 / turn_start.elapsed().as_secs_f32().max(0.001) } else { 0.0 },
                    prompt_tokens, comp_tokens,
                    0, // cache_hit_tokens
                    0.0, // cost_usd
                );
            }
            break;
        }

        if std::time::Instant::now().duration_since(last_activity) >= timeout {
            if cfg.telemetry_enabled {
                metrics.record_request("brain", &model, 504);
            }
            router.cancel(&request_id);
            let timeout_resp = build_error_response(504, "gateway_timeout", "Worker produced no output for too long");
            send_raw_json(&mut stream, 504, &timeout_resp).await;
            return;
        }

        // Yield to the io_uring event loop for 1ms, then poll for a chunk.
        monoio::time::sleep(std::time::Duration::from_millis(1)).await;

        match chunk_rx.try_recv() {
            Ok(chunk) => {
                last_activity = std::time::Instant::now();
                first_token = false;
                let json_str = chunk.delta.strip_prefix("data: ").unwrap_or(&chunk.delta).trim();
                if json_str.is_empty() || json_str == "[DONE]" {
                    // nothing to parse
                } else if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(choices) = val.get("choices").and_then(|c| c.as_array()) {
                        if let Some(first) = choices.first() {
                            if let Some(delta) = first.get("delta") {
                                // Extract content
                                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                    if !content.is_empty() {
                                        full_response.push_str(content);
                                    }
                                }

                                // Extract tool calls
                                if let Some(tc_array) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                                    for tc in tc_array {
                                        if let Some(function) = tc.get("function") {
                                            if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                                                // A new tool call name signals the start of a new call.
                                                // Flush the previous one into the accumulator first.
                                                if let Some(prev) = current_tool_call.take() {
                                                    accumulated_tool_calls.push(prev);
                                                }
                                                // Assign the new call directly — no push/pop dance.
                                                current_tool_call = Some(ToolCall {
                                                    id: format!("call_{}", Uuid::new_v4().to_string().replace('-', "")[..8].to_string()),
                                                    call_type: "function".into(),
                                                    function: ToolCallFunction {
                                                        name: name.to_string(),
                                                        arguments: String::new(),
                                                    }
                                                });
                                            }
                                            if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                                                if let Some(mut current) = current_tool_call.take() {
                                                    current.function.arguments.push_str(args);
                                                    current_tool_call = Some(current);
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                log::warn!(
                                    "[unhandled-translation] request={} worker chunk missing choices[0].delta: {}",
                                    request_id, super::super::common::truncate_for_log(json_str),
                                );
                            }
                        }
                    } else {
                        log::warn!(
                            "[unhandled-translation] request={} worker chunk missing choices array: {}",
                            request_id, super::super::common::truncate_for_log(json_str),
                        );
                    }
                } else {
                    log::warn!(
                        "[unhandled-translation] request={} worker chunk isn't valid JSON, dropping: {}",
                        request_id, super::super::common::truncate_for_log(json_str),
                    );
                }
            }
            Err(flume::TryRecvError::Empty)        => continue,
            Err(flume::TryRecvError::Disconnected) => {
                router.cancel(&request_id);
                break;
            }
        }
    }

    if let Some(last_tc) = current_tool_call.take() {
        accumulated_tool_calls.push(last_tc);
    }

    // ── Detect tool calls in the assembled response ───────────────────────────
    let (content, mut tool_calls) = parse_tool_calls_from_response(&full_response);

    if !accumulated_tool_calls.is_empty() {
        // If the worker natively sent tool_calls, use them and ignore any parsed from content
        // to avoid duplicating the same calls if the worker echoed them in text.
        tool_calls = Some(accumulated_tool_calls);
    }

    let actual_finish = if tool_calls.is_some() {
        FinishReason::ToolCalls
    } else {
        finish_reason
    };

    // ── Persist only the NEW user message + assistant response to DB ─────────────

    // ── Build OpenAI response envelope ───────────────────────────────────────
    let response = ChatResponse {
        id:      format!("chatcmpl-{}", request_id),
        object:  "chat.completion",
        created: now_secs(),
        model,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role:       "assistant",
                content,
                tool_calls,
            },
            finish_reason: actual_finish.as_str().to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: comp_tokens,
            total_tokens: prompt_tokens + comp_tokens,
        },
    };

    let body_bytes = serde_json::to_vec(&response).unwrap_or_default();

    // Save to idempotency if a key was provided (BUG-1 + BUG-2 fixes)
    if let Some(key) = idempotency_key {
        let idem_resp = crate::core::idempotency::IdempotentResponse {
            status: 200,
            content_type: "application/json".into(),
            body: body_bytes.clone(),
            created_at: now_secs() as i64,
        };
        idempotency.put(&key, idem_resp);
        // Note: arg order is (key, status, body: &[u8], content_type)
        let _ = db.put_idempotent_response(&key, 200, &body_bytes, "application/json");
    }

    let resp_str = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body_bytes.len()
    );
    let _ = stream.write_all(resp_str.into_bytes()).await;
    let _ = stream.write_all(body_bytes).await;
}