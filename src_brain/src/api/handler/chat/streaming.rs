use std::sync::Arc;
use std::time::Instant;
use monoio::io::AsyncWriteRentExt;
use monoio::net::TcpStream;
use serde_json::json;

use crate::core::config::Config;
use crate::db::memory::Database;
use crate::api::router::Router;
use crate::metrics::Metrics;

use super::convert::extract_delta_content;

// ─────────────────────────────────────────────────────────────────────────────
// Streaming SSE response passthrough
// ─────────────────────────────────────────────────────────────────────────────

pub async fn handle_streaming(
    mut stream:  TcpStream,
    request_id:  String,
    model:       String,
    chunk_rx:    flume::Receiver<crate::core::protocol::ChunkMessage>,
    done_rx:     flume::Receiver<crate::core::protocol::DoneMessage>,
    router:      Arc<Router>,
    _db:          Arc<Database>,
    _session_id:  String,
    _user_id:     Option<i64>,
    _last_user_msg: Option<crate::core::protocol::Message>,
    cfg:         Arc<Config>,
    metrics:     Arc<Metrics>,
    turn_start:  Instant,
) {
    // Send SSE headers
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

    let timeout = std::time::Duration::from_secs(cfg.worker_timeout_s);
    let mut last_activity = std::time::Instant::now();
    let mut last_ping = std::time::Instant::now();
    let mut first_token = true;
    let mut full_response = String::new(); // accumulate for DB persistence
    let mut generated_tool_calls: Vec<crate::core::protocol::ToolCall> = Vec::new();
    let mut current_tool_call: Option<crate::core::protocol::ToolCall> = None;
    let mut current_tool_index: Option<u64> = None;

    loop {
        // Check done channel first (non-blocking)
        if let Ok(done) = done_rx.try_recv() {
            // Telemetry: record TTFT and tokens (§9.3)
            if cfg.telemetry_enabled {
                let elapsed_ms = turn_start.elapsed().as_millis() as u64;
                if first_token {
                    metrics.record_ttft("brain", &model, elapsed_ms);
                }
                if done.comp_tokens > 0 {
                    let elapsed_s = turn_start.elapsed().as_secs_f32().max(0.001);
                    let tps = done.comp_tokens as f32 / elapsed_s;
                    metrics.record_tps("brain", &model, tps);
                }
                metrics.record_tokens("brain", &model, done.prompt_tokens as u64, done.comp_tokens as u64);
            }

            if let Some(cur) = current_tool_call.take() {
                generated_tool_calls.push(cur);
            }


            // Send final chunk with finish_reason
            let final_chunk = format!(
                "data: {}\n\n",
                json!({
                    "id": format!("chatcmpl-{}", request_id),
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": done.finish_reason.as_str()}]
                })
            );
            let _ = stream.write_all(final_chunk.into_bytes()).await;
            let _ = stream.write_all(b"data: [DONE]\n\n").await;
            // router.finish() already called by tunnel when worker sends Done
            return;
        }

        // Check timeout
        let idle = std::time::Instant::now().duration_since(last_activity);
        if idle >= timeout {
            // Telemetry: timeout metric
            if cfg.telemetry_enabled {
                // Only record TTFT on timeout if we never saw a real first token —
                // if first_token is already false, we already recorded TTFT when
                // the first chunk arrived and recording again would inflate the bucket.
                if first_token {
                    let elapsed_ms = turn_start.elapsed().as_millis() as u64;
                    metrics.record_ttft("brain", &model, elapsed_ms);
                }
                metrics.record_request("brain", &model, 504);
            }

            // finish_reason "length" (not "stop") — this was cut off, not a
            // clean natural end, and callers checking finish_reason need to
            // be able to tell the difference.
            let timeout_chunk = format!(
                "data: {}\n\n",
                json!({
                    "id": format!("chatcmpl-{}", request_id),
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                    "error": "worker_idle_timeout"
                })
            );
            let _ = stream.write_all(timeout_chunk.into_bytes()).await;
            let _ = stream.write_all(b"data: [DONE]\n\n").await;
            // Clean up router entry on timeout — worker may eventually send Done but
            // neither receiver is listening; this prevents the HashMap from leaking.
            router.finish(&request_id);
            return;
        }

        // Emit a keep-alive ping every 15 seconds to prevent client socket timeout
        if last_ping.elapsed().as_secs() >= 15 {
            let _ = stream.write_all(b": ping\n\n").await;
            last_ping = std::time::Instant::now();
        }

        // Yield to the io_uring event loop for 5ms, then poll for a chunk.
        // This is preferable to recv_timeout() which would block the entire thread.
        monoio::time::sleep(std::time::Duration::from_millis(5)).await;

        match chunk_rx.try_recv() {
            Ok(chunk) => {
                last_activity = std::time::Instant::now();
                if cfg.telemetry_enabled && first_token {
                    first_token = false;
                    let ttft_ms = turn_start.elapsed().as_millis() as u64;
                    metrics.record_ttft("brain", &model, ttft_ms);
                }

                let text = extract_delta_content(&chunk.delta);
                full_response.push_str(&text); // accumulate for history

                let chunk_delta;
                let json_str = chunk.delta.strip_prefix("data: ").unwrap_or(&chunk.delta).trim();
                if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(json_str) {
                    let mut modified = false;
                    if let Some(choices) = val.get_mut("choices").and_then(|c| c.as_array_mut()) {
                        if let Some(first) = choices.first_mut() {
                            if let Some(delta) = first.get_mut("delta") {
                                if let Some(tc_array) = delta.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
                                    for tc in tc_array.iter_mut() {
                                        let index = tc.get("index").and_then(|i| i.as_u64());
                                        let has_name = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).is_some();

                                        // Is this a NEW tool call, or more of the one we're
                                        // already building? `current_tool_index != index` used
                                        // to be the whole check, but plenty of OpenAI-compatible
                                        // local servers (this deployment's llama.cpp included)
                                        // omit `index` entirely — `None != None` is `false`, so
                                        // that comparison silently never fired and the very
                                        // first tool call's name/id/arguments were dropped on
                                        // the floor for the rest of the turn. Mirror the same
                                        // heuristic anthropic/convert.rs::accumulate_delta uses:
                                        //  - nothing tracked yet -> definitely new.
                                        //  - both sides have a usable index -> trust it.
                                        //  - no usable index -> only "new" if this delta actually
                                        //    announces a name AND the call we're building already
                                        //    has some arguments (a genuinely fresh call announces
                                        //    its name before any arguments arrive).
                                        let is_new_call = match current_tool_call.as_ref() {
                                            None => true,
                                            Some(cur) => match (current_tool_index, index) {
                                                (Some(ci), Some(ni)) => ci != ni,
                                                _ => has_name && !cur.function.arguments.is_empty(),
                                            },
                                        };

                                        if is_new_call {
                                            // New tool call at a new index
                                            if let Some(cur) = current_tool_call.take() {
                                                generated_tool_calls.push(cur);
                                            }
                                            current_tool_index = index;

                                            let new_id = tc.get("id").and_then(|i| i.as_str()).map(|s| s.to_string())
                                                .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().to_string().replace('-', "")[..8].to_string()));

                                            if tc.get("id").is_none() {
                                                tc["id"] = serde_json::Value::String(new_id.clone());
                                                tc["type"] = serde_json::Value::String("function".to_string());
                                                modified = true;
                                            }

                                            current_tool_call = Some(crate::core::protocol::ToolCall {
                                                id: new_id,
                                                call_type: "function".to_string(),
                                                function: crate::core::protocol::ToolCallFunction {
                                                    name: tc.get("function").and_then(|f| f.get("name")).and_then(|s| s.as_str()).unwrap_or_default().to_string(),
                                                    arguments: String::new(),
                                                }
                                            });
                                        }
                                        if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")).and_then(|a| a.as_str()) {
                                            if let Some(cur) = current_tool_call.as_mut() {
                                                cur.function.arguments.push_str(args);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if modified {
                        chunk_delta = serde_json::to_string(&val).unwrap_or(chunk.delta.clone());
                    } else {
                        chunk_delta = json_str.to_string();
                    }
                } else {
                    log::warn!(
                        "[unhandled-translation] request={} worker delta chunk isn't valid JSON, forwarding raw: {}",
                        request_id, super::super::common::truncate_for_log(json_str),
                    );
                    chunk_delta = json_str.to_string();
                }

                let line = format!("data: {}\n\n", chunk_delta);
                if stream.write_all(line.into_bytes()).await.0.is_err() {
                    // Client disconnected — clean up router entry now (BUG-8 fix)
                    router.cancel(&request_id);
                    return;
                }
            }
            Err(flume::TryRecvError::Empty)        => continue,
            Err(flume::TryRecvError::Disconnected) => {
                router.cancel(&request_id);
                return;
            }
        }
    }
}