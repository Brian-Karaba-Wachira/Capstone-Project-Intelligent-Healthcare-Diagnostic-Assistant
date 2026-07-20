// ═════════════════════════════════════════════════════════════════════════════
//  lmodel/src/bridge/mod.rs
//
//  Bridges a single task from the brain to the local llama.cpp server and
//  streams the SSE response back over the WS tunnel.
//
// ═════════════════════════════════════════════════════════════════════════════

use crate::protocol::{
    ChunkMessage, DoneMessage, ErrorMessage, FinishReason, Message, TaskMessage,
};
use bytes::{Buf, BytesMut};
use flume::Sender;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::TcpStream;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Entry point — called per task from router::dispatch.
/// Uses a `Sender<String>` to serialize all WS writes through the main loop,
/// avoiding concurrent `borrow_mut()` on the shared `RefCell<WsWriter>`.
pub async fn run_task(
    task: TaskMessage,
    write_tx: Sender<String>,
    llamacpp_port: u16,
    active_requests: Arc<AtomicU32>,
    cancel_rx: flume::Receiver<()>,
) {
    let request_id = task.request_id.clone();
    active_requests.fetch_add(1, Ordering::Relaxed);

    let result = run_task_inner(&task, &write_tx, llamacpp_port, cancel_rx).await;

    active_requests.fetch_sub(1, Ordering::Relaxed);

    if let Err(e) = result {
        log::error!("Bridge error for {}: {}", request_id, e);
        let err_msg = Message::Error(ErrorMessage {
            request_id,
            message: e,
            // 502: this is always an upstream (llama.cpp / tunnel) failure,
            // never something caused by the client's request shape.
            code: 502,
        });
        if let Ok(json) = serde_json::to_string(&err_msg) {
            let _ = write_tx.send_async(json).await;
        }
    }
}

async fn run_task_inner(
    task: &TaskMessage,
    write_tx: &Sender<String>,
    llamacpp_port: u16,
    cancel_rx: flume::Receiver<()>,
) -> Result<(), String> {
    // ── Connect to llama.cpp ──────────────────────────────────────────────────
    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", llamacpp_port))
        .await
        .map_err(|e| format!("Connect to llama.cpp: {}", e))?;

    // ── Build request body — forward exactly what the brain sent ─────────────
    // No local defaults here: this worker is a transparent tunnel, and the
    // client CLI (Claude Code, etc.) is the one driving generation-control
    // decisions for a session — brain forwards whatever it received
    // (see extract_stop_sequences on the brain side). Previously this
    // substituted a hardcoded ChatML `["<|im_end|>", "<|im_start|>"]` stop
    // list whenever `task.stop` was None, which is wrong for any non-ChatML
    // model this same worker binary might be pointed at, and unnecessary:
    // llama-server already stops at the model's own EOS token independent
    // of the `stop` field, which only adds *extra* string cutoffs on top of
    // that. If `task.stop` is None, we now omit `stop` entirely instead of
    // guessing.
    let mut body = json!({
        "model":       task.model,
        "messages":    task.messages,
        "max_tokens":  task.max_tokens,
        "temperature": task.temperature,
        "stream":      true,
    });

    if let Some(stop) = &task.stop {
        body["stop"] = json!(stop);
    }

    if let Some(top_p) = task.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(top_k) = task.top_k {
        body["top_k"] = json!(top_k);
    }
    if let Some(tools) = &task.tools {
        body["tools"] = json!(tools);
    }
    if let Some(tool_choice) = &task.tool_choice {
        body["tool_choice"] = json!(tool_choice);
    }
    if let Some(parallel) = task.parallel_tool_calls {
        body["parallel_tool_calls"] = json!(parallel);
    }

    let body_str = serde_json::to_string(&body).map_err(|e| e.to_string())?;

    let http_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: 127.0.0.1\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        body_str.len(),
        body_str
    );

    stream
        .write_all(http_req.into_bytes())
        .await
        .0
        .map_err(|e| format!("Write to llama.cpp: {}", e))?;

    // ── Read & parse SSE response ─────────────────────────────────────────────
    // `buf` holds raw, still-possibly-chunk-framed bytes straight off the
    // socket; `body_buf` holds the decoded SSE body once chunk framing (if
    // any) has been stripped out of it.
    let mut buf = BytesMut::with_capacity(8192);
    let mut body_buf = BytesMut::with_capacity(8192);
    let mut headers_done = false;
    let mut is_chunked = false;
    let mut chunked_finished = false;

    // llama-server usually only attaches usage/finish_reason to the final
    // chunk (sometimes not at all) — accumulate as we go, default to
    // Stop/0/0 so the eventual DoneMessage is always fully populated.
    let mut prompt_tokens = 0u32;
    let mut comp_tokens = 0u32;
    let mut finish_reason = FinishReason::Stop;

    loop {
        let tmp = vec![0u8; 4096];
        
        monoio::select! {
            read_res = stream.read(tmp) => {
                let (res, tmp) = read_res;
                let n = res.map_err(|e| format!("Read from llama.cpp: {}", e))?;
                if n == 0 {
                    break; // EOF — llama.cpp closed the connection
                }
                buf.extend_from_slice(&tmp[..n]);
            },
            _ = cancel_rx.recv_async() => {
                log::info!("Task {} cancelled by client disconnect", task.request_id);
                // Return early; `stream` goes out of scope and drops the TCP connection,
                // which signals llama.cpp to abort generation instantly.
                return Ok(());
            }
        }

        // Skip HTTP headers, and check whether the body is chunked-transfer
        // encoded. llama-server streams SSE over chunked transfer encoding,
        // but this used to be ignored entirely: the raw chunk framing (a
        // hex chunk-size line like "12c\r\n" before every chunk, plus a
        // trailing "\r\n" after each one) fell straight into the line
        // splitter below and got logged as a bogus "non-SSE line" — one
        // warning per streamed token. Harmless-looking most of the time
        // because chunk boundaries usually line up with SSE line
        // boundaries, but not guaranteed, so a chunk split landing mid-line
        // could silently corrupt or drop real data instead of just
        // spamming the log.
        if !headers_done {
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let header_bytes = buf.split_to(pos + 4);
                let header_str = String::from_utf8_lossy(&header_bytes);
                is_chunked = header_str.lines().any(|l| {
                    let l = l.to_ascii_lowercase();
                    l.starts_with("transfer-encoding:") && l.contains("chunked")
                });
                headers_done = true;
            } else {
                continue;
            }
        }

        if is_chunked {
            if !chunked_finished {
                chunked_finished = dechunk(&mut buf, &mut body_buf)
                    .map_err(|e| format!("Malformed chunked response from llama.cpp: {}", e))?;
            }
        } else {
            // Not chunked (e.g. a plain error body) — pass bytes through as-is.
            body_buf.extend_from_slice(&buf);
            buf.clear();
        }

        // Process complete lines
        while let Some(nl) = body_buf.iter().position(|&b| b == b'\n') {
            let line_bytes = body_buf.split_to(nl + 1);
            let line = String::from_utf8_lossy(&line_bytes).trim().to_string();

            if line.is_empty() {
                continue;
            }

            if line == "data: [DONE]" {
                return send_done(write_tx, &task.request_id, finish_reason, prompt_tokens, comp_tokens).await;
            }

            // Anything llama.cpp sends on this stream that isn't a "data: "
            // SSE line (a bare HTTP/1.0-style error body, a stray log line
            // leaking onto the socket, etc.) used to vanish here with no
            // trace at all — the client would just see the turn go quiet.
            // Log it so a hung request can be traced back to whatever the
            // worker actually sent instead of just "nothing happened".
            let Some(json_str) = line.strip_prefix("data: ") else {
                log::warn!(
                    "[unhandled-translation] task={} non-SSE line from llama.cpp (not \"data: \"-prefixed), dropping: {}",
                    task.request_id,
                    truncate_for_log(&line),
                );
                continue;
            };

            let val = match serde_json::from_str::<Value>(json_str) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!(
                        "[unhandled-translation] task={} failed to parse SSE payload as JSON ({}), dropping: {}",
                        task.request_id, e, truncate_for_log(json_str),
                    );
                    continue;
                }
            };

            // Usage can show up on any chunk depending on llama-server build.
            if let Some(usage) = val.get("usage") {
                if let Some(p) = usage.get("prompt_tokens").and_then(Value::as_u64) {
                    prompt_tokens = p as u32;
                }
                if let Some(c) = usage.get("completion_tokens").and_then(Value::as_u64) {
                    comp_tokens = c as u32;
                }
            }

            // A well-formed OpenAI-style chunk always has choices[0].delta —
            // if it's missing, this chunk isn't shaped the way the rest of
            // this function assumes and we're about to silently forward
            // useless "delta": null content to the client. Surface that
            // instead of guessing.
            if val.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("delta")).is_none() {
                log::warn!(
                    "[unhandled-translation] task={} SSE payload missing choices[0].delta — worker may be sending a shape this bridge doesn't recognize: {}",
                    task.request_id, truncate_for_log(json_str),
                );
            }

            let delta = &val["choices"][0]["delta"];
            let is_tool_call = delta.get("tool_calls").map_or(false, |v| !v.is_null());

            // Forward the raw SSE line verbatim — the brain pipes `delta`
            // straight through to the client unmodified, so this must be
            // the *whole* line (still prefixed "data: "), not just the
            // extracted text. Sending bare text here (the old behavior)
            // produced invalid SSE on the client side.
            send_chunk(write_tx, &task.request_id, &line, is_tool_call).await?;

            if let Some(fr) = val["choices"][0]["finish_reason"].as_str() {
                finish_reason = map_finish_reason(fr);
                // Some llama.cpp builds close the socket right after the
                // final chunk without ever sending a literal [DONE] line —
                // treat a non-null finish_reason as authoritative completion.
                return send_done(write_tx, &task.request_id, finish_reason, prompt_tokens, comp_tokens).await;
            }
        }

        if is_chunked && chunked_finished {
            break; // terminating chunk consumed — no need to wait for EOF too
        }
    }

    // ── Leftover, never-line-terminated body (was BUG) ───────────────────────
    // llama-server's error responses (e.g. "request (N tokens) exceeds the
    // available context size") come back as a single, non-chunked JSON body
    // with NO trailing newline — e.g. a context-overflow rejection. The line
    // splitter above only fires on '\n', so that whole body sat in `body_buf`
    // untouched until the socket hit EOF, and execution fell straight through
    // to `send_done` with the *default* Stop/0/0 values below — turning a
    // real upstream error into what looked to the client like a normal,
    // silently-empty completed turn (the task just "stops" with no error).
    //
    // Before manufacturing a fake Done, check whatever's left in `body_buf`
    // for an actual error payload and propagate it as a real error instead.
    if !body_buf.is_empty() {
        if let Some(msg) = extract_error_message(&body_buf) {
            return Err(format!("llama.cpp: {}", msg));
        }
        // Not JSON / not an {"error": ...} shape — still worth a log line
        // rather than silently discarding it, since we're about to treat
        // this as a normal completion.
        log::warn!(
            "[unhandled-translation] task={} {} unconsumed byte(s) left in body_buf at EOF, \
             not recognized as an error payload, dropping: {}",
            task.request_id,
            body_buf.len(),
            truncate_for_log(&String::from_utf8_lossy(&body_buf)),
        );
    }

    // EOF without [DONE] or finish_reason — send Done anyway so the brain
    // doesn't hang waiting on a request that will never resolve.
    send_done(write_tx, &task.request_id, finish_reason, prompt_tokens, comp_tokens).await
}

/// Try to pull a human-readable message out of a plain (non-SSE) llama.cpp
/// error body. llama-server's own shape is `{"error": {"message": "...", ...}}`,
/// but we also fall back to a couple of looser shapes in case the build in
/// use differs, rather than only recognizing one exact structure and staying
/// silent on anything else.
fn extract_error_message(body: &[u8]) -> Option<String> {
    let val: Value = serde_json::from_slice(body).ok()?;

    if let Some(msg) = val.get("error").and_then(|e| e.get("message")).and_then(Value::as_str) {
        return Some(msg.to_string());
    }
    if let Some(msg) = val.get("error").and_then(Value::as_str) {
        return Some(msg.to_string());
    }
    if let Some(msg) = val.get("message").and_then(Value::as_str) {
        return Some(msg.to_string());
    }
    None
}

/// Strip HTTP/1.1 chunked-transfer-encoding framing (RFC 7230 §4.1) out of
/// `buf`, appending each fully-received chunk's decoded body onto `out`.
/// Any not-yet-complete chunk-size line or chunk body is left untouched in
/// `buf` so a later call (once more bytes have arrived off the socket) can
/// pick up where this one left off. Returns `Ok(true)` once the
/// terminating zero-length chunk (and any trailer headers) has been fully
/// consumed, meaning the body is complete.
fn dechunk(buf: &mut BytesMut, out: &mut BytesMut) -> Result<bool, String> {
    loop {
        let Some(line_end) = buf.windows(2).position(|w| w == b"\r\n") else {
            return Ok(false); // chunk-size line not fully buffered yet
        };

        let size_str = std::str::from_utf8(&buf[..line_end])
            .map_err(|_| "chunk size line is not valid UTF-8".to_string())?
            .split(';') // ignore chunk-extensions, if any
            .next()
            .unwrap_or("")
            .trim();
        let size = usize::from_str_radix(size_str, 16)
            .map_err(|_| format!("invalid chunk size {:?}", size_str))?;

        let data_start = line_end + 2;

        if size == 0 {
            // Terminating chunk. Usually just a bare CRLF follows; but
            // trailer headers (each their own CRLF-terminated line) are
            // technically legal before the final blank-line CRLF, so
            // handle both.
            let rest = &buf[data_start..];
            if rest.starts_with(b"\r\n") {
                buf.advance(data_start + 2);
                return Ok(true);
            }
            return match rest.windows(4).position(|w| w == b"\r\n\r\n") {
                Some(p) => {
                    buf.advance(data_start + p + 4);
                    Ok(true)
                }
                None => Ok(false), // trailers not fully buffered yet
            };
        }

        let needed = data_start + size + 2; // chunk data + its trailing CRLF
        if buf.len() < needed {
            return Ok(false); // wait for the rest of this chunk
        }

        out.extend_from_slice(&buf[data_start..data_start + size]);
        buf.advance(needed);
        // Don't return yet — another full chunk may already be buffered.
    }
}

fn map_finish_reason(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

/// Keep unhandled-translation log lines from blowing up the log with an
/// entire multi-KB SSE payload — enough context to recognize the shape,
/// not the whole thing.
fn truncate_for_log(s: &str) -> String {
    const MAX_CHARS: usize = 300;
    if s.chars().count() <= MAX_CHARS {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(MAX_CHARS).collect();
        format!("{}… ({} bytes total)", truncated, s.len())
    }
}

async fn send_chunk(
    write_tx: &Sender<String>,
    request_id: &str,
    raw_sse_line: &str,
    is_tool_call: bool,
) -> Result<(), String> {
    let chunk = Message::Chunk(ChunkMessage {
        request_id: request_id.to_string(),
        delta: raw_sse_line.to_string(),
        is_tool_call,
    });
    match serde_json::to_string(&chunk) {
        Ok(json) => {
            if write_tx.send_async(json).await.is_err() {
                return Err("Write channel closed".into());
            }
        }
        Err(e) => log::error!("Chunk serialize error: {}", e),
    }
    Ok(())
}

async fn send_done(
    write_tx: &Sender<String>,
    request_id: &str,
    finish_reason: FinishReason,
    prompt_tokens: u32,
    comp_tokens: u32,
) -> Result<(), String> {
    let done = Message::Done(DoneMessage {
        request_id: request_id.to_string(),
        finish_reason,
        prompt_tokens,
        comp_tokens,
    });
    if let Ok(json) = serde_json::to_string(&done) {
        let _ = write_tx.send_async(json).await;
    }
    Ok(())
}
