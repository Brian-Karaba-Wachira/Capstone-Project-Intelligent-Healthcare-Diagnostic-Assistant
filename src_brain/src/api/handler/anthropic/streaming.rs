use std::sync::Arc;
use std::time::Instant;
use monoio::io::AsyncWriteRentExt;
use monoio::net::TcpStream;
use serde_json::json;

use ai_provider_converter::{
    message_start, content_block_start, content_block_delta, content_block_stop,
    message_delta, message_stop, claude_ping, AnthropicUsage,
};

use crate::core::config::Config;
use crate::core::protocol::{Message, ToolCall};
use crate::db::memory::Database;
use crate::api::router::Router;
use crate::metrics::Metrics;

use super::convert::{accumulate_delta, parse_tool_calls_from_text, ToolStreamEvent};

// ── Anthropic streaming handler ─────────────────────────────────────────────
//
// Frame-level SSE events (message_start, content_block_*, message_delta,
// message_stop, ping) are now built via ai-provider-converter's stream
// builders instead of hand-rolled `json!`/`format!` strings, so the exact
// same event shapes the library already validated (and uses in its own
// `translate_chat_stream` reducer) are what this handler emits — one
// definition of "what an Anthropic SSE event looks like" instead of two.
// The tool-call accumulation below stays brain-specific: it's reading
// *worker* output (a local llama.cpp-style OpenAI-shaped stream) chunk by
// chunk off a flume channel, which is a different shape of problem than the
// library's `translate_chat_stream` (which owns a real `Stream` of upstream
// bytes end-to-end). It replays `ToolStreamEvent`s from `accumulate_delta`
// rather than re-parsing the raw chunk itself, so there is exactly one place
// that decides where tool-call block boundaries fall.
//
// Some local models aren't fine-tuned for native function calling and print
// a `<tool_call>{...}</tool_call>` tag as plain content instead of using the
// structured `tool_calls` delta field. Text is therefore held back by a
// small margin (`TOOL_TAG_HOLDBACK`) before being streamed as `text_delta`,
// so that if a `<tool_call>` tag is about to appear we can suppress it
// entirely instead of showing the raw tag to the client and *then* also
// executing it as a real tool call once the fallback parser catches it at
// end-of-stream.
const TOOL_TAG_HOLDBACK: usize = 10; // len("<tool_call>") - 1

/// Largest byte index `<= index` that lands on a UTF-8 character boundary
/// in `s`. `str::floor_char_boundary` is nightly-only, so this is the
/// stable equivalent — used to make byte-offset arithmetic on streamed
/// model text safe to slice/drain regardless of where multi-byte
/// characters happen to fall.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut idx = index;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

pub async fn handle_anthropic_streaming(
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
    db:          Arc<Database>,
) {
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

    // Anthropic streaming starts with message_start event
    if stream.write_all(message_start(&format!("msg_{}", request_id), &model).to_vec()).await.0.is_err() {
        router.cancel(&request_id);
        return;
    }

    // Content block start
    let _ = stream.write_all(
        content_block_start(0, json!({"type": "text", "text": ""})).to_vec()
    ).await;

    let timeout = std::time::Duration::from_secs(cfg.worker_timeout_s);
    let mut last_activity = std::time::Instant::now();
    let mut last_ping = std::time::Instant::now();
    let mut first_token = true;
    let mut full_text = String::new();

    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut current_tool_call: Option<ToolCall> = None;
    let mut current_tool_index: Option<u64> = None;
    let prompt_tokens;
    let comp_tokens;

    // Text held back so we can detect a leading `<tool_call>` tag before
    // showing any of it to the client (see TOOL_TAG_HOLDBACK above).
    let mut pending_text = String::new();
    let mut tool_call_leak_detected = false;

    // Which Anthropic content block is currently open. `None` while text
    // block 0 is open and hasn't been closed yet; `Some(0)` once it's been
    // explicitly tracked as closed; `Some(n)` for the currently open tool
    // block. Tracked explicitly (rather than re-derived from `tool_calls`)
    // so end-of-stream/timeout close exactly one block, exactly once.
    let mut text_block_closed = false;
    let mut current_tool_block_index: Option<usize> = None;
    let mut next_tool_block_index: usize = 1;

    loop {
        if let Ok(done) = done_rx.try_recv() {
            prompt_tokens = done.prompt_tokens;
            comp_tokens = done.comp_tokens;

            if cfg.telemetry_enabled {
                let elapsed_ms = turn_start.elapsed().as_millis() as u64;
                if first_token { metrics.record_ttft("brain", &model, elapsed_ms); }
                if comp_tokens > 0 {
                    metrics.record_tps("brain", &model, comp_tokens as f32 / turn_start.elapsed().as_secs_f32().max(0.001));
                }
                metrics.record_tokens("brain", &model, prompt_tokens as u64, comp_tokens as u64);
            }

            // Persist conversation history (parity with the OpenAI-format
            // path — this was previously never called for Anthropic
            // requests at all, so Claude Code sessions were invisible in
            // session
            let _ = db.record_turn_metrics(
                &session_id, &request_id, "brain",
                None, // user_id
                &model,
                if first_token { 0 } else { turn_start.elapsed().as_millis() as u32 },
                if comp_tokens > 0 { comp_tokens as f32 / turn_start.elapsed().as_secs_f32().max(0.001) } else { 0.0 },
                prompt_tokens, comp_tokens,
                0, // cache_hit_tokens
                0.0, // cost_usd
            );

            if let Some(last_tc) = current_tool_call.take() {
                tool_calls.push(last_tc);
            }

            // Flush whatever text we were holding back for tag detection.
            // If a `<tool_call>` tag was ever detected, this stays empty —
            // none of it was shown to the client, so there's nothing stale
            // to flush and no risk of showing the raw tag alongside the
            // synthesized tool_use block below.
            if !tool_call_leak_detected && !pending_text.is_empty() {
                let out_delta = content_block_delta(0, json!({"type": "text_delta", "text": pending_text}));
                let _ = stream.write_all(out_delta.to_vec()).await;
                pending_text.clear();
            }

            // Only fall back to text-pattern parsing if the worker didn't
            // already give us structured tool_calls — avoids double-counting
            // the same call twice if a chat template leaks a <tool_call> tag
            // into the content on top of the structured delta. Because we
            // suppressed streaming the raw tag text above, nothing has
            // leaked into the visible transcript by the time we synthesize
            // proper tool_use blocks here.
            let mut fallback_ran = false;
            if tool_calls.is_empty() {
                if let Some(parsed) = parse_tool_calls_from_text(&full_text) {
                    fallback_ran = true;
                    tool_calls = parsed;

                    if !text_block_closed {
                        let _ = stream.write_all(content_block_stop(0).to_vec()).await;
                        text_block_closed = true;
                    }

                    for tc in tool_calls.iter() {
                        let block = next_tool_block_index;
                        next_tool_block_index += 1;
                        let tool_start = content_block_start(
                            block,
                            json!({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": {}}),
                        );
                        let _ = stream.write_all(tool_start.to_vec()).await;

                        if let Ok(args) = serde_json::from_str::<serde_json::Value>(&tc.function.arguments) {
                            let delta = content_block_delta(
                                block,
                                json!({"type": "input_json_delta", "partial_json": serde_json::to_string(&args).unwrap_or_default()}),
                            );
                            let _ = stream.write_all(delta.to_vec()).await;
                        }

                        // Each fallback block is opened and closed here, in
                        // full, immediately — so it must not be closed again
                        // below.
                        let _ = stream.write_all(content_block_stop(block).to_vec()).await;
                    }
                }
            }

            // Close whichever single block is still open. The fallback loop
            // above already closes each block it creates, so skip this when
            // it ran. Otherwise: a tool block left open by live structured
            // streaming, or text block 0 if the response was plain text.
            if !fallback_ran {
                if let Some(block) = current_tool_block_index {
                    let _ = stream.write_all(content_block_stop(block).to_vec()).await;
                } else if !text_block_closed {
                    let _ = stream.write_all(content_block_stop(0).to_vec()).await;
                }
            }

            // "tool_use" if the model made tool calls (regardless of what the
            // worker's finish_reason literally said — some local chat
            // templates report "stop" even when they emitted a tool call as
            // text). Otherwise map the worker's real finish_reason through
            // the shared library mapper instead of hardcoding "end_turn", so
            // e.g. a length-limited completion is reported as "max_tokens".
            let stop_reason = if !tool_calls.is_empty() {
                "tool_use".to_string()
            } else {
                ai_provider_converter::openai_finish_reason_to_anthropic_stop_reason(done.finish_reason.as_str())
            };
            let usage = AnthropicUsage {
                input_tokens: prompt_tokens as u64,
                output_tokens: comp_tokens as u64,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            };
            let _ = stream.write_all(message_delta(Some(&stop_reason), usage).to_vec()).await;
            let _ = stream.write_all(message_stop().to_vec()).await;
            return;
        }

        let idle = std::time::Instant::now().duration_since(last_activity);
        if idle >= timeout {
            // Close out whatever block is currently open before signalling
            // stop — an Anthropic client's SSE parser expects every
            // content_block_start to be matched by a content_block_stop, and
            // a message_delta with a stop_reason before message_stop.
            // Skipping straight to message_stop (as this used to) is invalid
            // framing and gives the client nothing to recover from mid-turn.
            // Flush any held-back text first so it isn't silently lost.
            if !tool_call_leak_detected && !pending_text.is_empty() {
                let out_delta = content_block_delta(0, json!({"type": "text_delta", "text": std::mem::take(&mut pending_text)}));
                let _ = stream.write_all(out_delta.to_vec()).await;
            }
            if let Some(block) = current_tool_block_index {
                let _ = stream.write_all(content_block_stop(block).to_vec()).await;
            } else if !text_block_closed {
                let _ = stream.write_all(content_block_stop(0).to_vec()).await;
            }
            // "max_tokens" isn't literally true, but of the fixed Anthropic
            // stop_reason values it's the one that correctly tells the
            // client "this response is incomplete, don't treat it as a
            // clean end_turn" — closer to reality than any alternative.
            let usage = AnthropicUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            };
            let _ = stream.write_all(message_delta(Some("max_tokens"), usage).to_vec()).await;
            let _ = stream.write_all(message_stop().to_vec()).await;
            router.cancel(&request_id);
            return;
        }

        // Emit a keep-alive ping every 15 seconds to prevent client socket timeout
        if last_ping.elapsed().as_secs() >= 15 {
            let _ = stream.write_all(claude_ping().to_vec()).await;
            last_ping = std::time::Instant::now();
        }

        monoio::time::sleep(std::time::Duration::from_millis(5)).await;

        match chunk_rx.try_recv() {
            Ok(chunk) => {
                last_activity = std::time::Instant::now();
                if cfg.telemetry_enabled && first_token {
                    first_token = false;
                    metrics.record_ttft("brain", &model, turn_start.elapsed().as_millis() as u64);
                }

                // Single parse of this delta — updates full_text, tool_calls,
                // and current_tool_call, and reports back the block-boundary
                // decisions it made so we don't have to re-derive them here.
                let (text_res, tool_events) = accumulate_delta(&chunk.delta, &mut full_text, &mut tool_calls, &mut current_tool_call, &mut current_tool_index);

                // Stream text, but held back by TOOL_TAG_HOLDBACK bytes so a
                // `<tool_call>` tag can be caught before any of it reaches
                // the client. Once caught, we stop streaming text for the
                // rest of the turn — the fallback parser in the done_rx
                // branch will synthesize proper tool_use blocks from
                // full_text instead.
                if let Some(text) = text_res {
                    if !tool_call_leak_detected {
                        pending_text.push_str(&text);
                        if pending_text.contains("<tool_call>") {
                            tool_call_leak_detected = true;
                            pending_text.clear();
                        } else {
                            let holdback = TOOL_TAG_HOLDBACK.min(pending_text.len());
                            // `pending_text.len() - holdback` is a byte offset, but
                            // holdback is a fixed byte count with no awareness of
                            // UTF-8 character boundaries — if the model emits any
                            // multi-byte character (accents, CJK, emoji, box-drawing,
                            // etc.) near the tail of `pending_text`, this can land
                            // mid-character. `String::drain` then panics
                            // (`assertion failed: self.is_char_boundary(end)`), which
                            // is exactly what was crashing this task and silently
                            // killing the stream. Round down to the nearest char
                            // boundary instead.
                            let raw_len = pending_text.len() - holdback;
                            let safe_len = floor_char_boundary(&pending_text, raw_len);
                            if safe_len > 0 {
                                let to_emit: String = pending_text.drain(..safe_len).collect();
                                let out_delta = content_block_delta(0, json!({"type": "text_delta", "text": to_emit}));
                                if stream.write_all(out_delta.to_vec()).await.0.is_err() {
                                    router.cancel(&request_id);
                                    return;
                                }
                            }
                        }
                    }
                }

                // Replay the tool-call block-boundary decisions accumulate_delta
                // already made — no re-parsing, so these can never disagree
                // with what ended up in `tool_calls`.
                for event in tool_events {
                    match event {
                        ToolStreamEvent::Started { id, name } => {
                            if let Some(prev) = current_tool_block_index {
                                let _ = stream.write_all(content_block_stop(prev).to_vec()).await;
                            } else if !text_block_closed {
                                // Flush any text still sitting in the holdback
                                // buffer before closing block 0 — otherwise
                                // text right at a text-then-tool-call boundary
                                // gets silently dropped instead of shown.
                                if !tool_call_leak_detected && !pending_text.is_empty() {
                                    let out_delta = content_block_delta(0, json!({"type": "text_delta", "text": std::mem::take(&mut pending_text)}));
                                    let _ = stream.write_all(out_delta.to_vec()).await;
                                }
                                let _ = stream.write_all(content_block_stop(0).to_vec()).await;
                                text_block_closed = true;
                            }
                            let block = next_tool_block_index;
                            next_tool_block_index += 1;
                            current_tool_block_index = Some(block);
                            let tool_start = content_block_start(
                                block,
                                json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
                            );
                            let _ = stream.write_all(tool_start.to_vec()).await;
                        }
                        ToolStreamEvent::ArgsDelta(args) => {
                            if let Some(block) = current_tool_block_index {
                                let delta_json = content_block_delta(
                                    block,
                                    json!({"type": "input_json_delta", "partial_json": args}),
                                );
                                let _ = stream.write_all(delta_json.to_vec()).await;
                            }
                        }
                    }
                }
            }
            Err(flume::TryRecvError::Empty)      => continue,
            Err(flume::TryRecvError::Disconnected) => {
                let _ = stream.write_all(message_stop().to_vec()).await;
                router.cancel(&request_id);
                return;
            }
        }
    }
}
