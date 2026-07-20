use crate::error::Result;
use crate::stream::openai::builders::{chat_completion_chunk, done, sse_event, usage_chunk};
use crate::utils::finish_reason::anthropic_stop_reason_to_openai_finish_reason;
use async_stream::try_stream;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::pin::Pin;

pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;

pub fn translate_anthropic_to_openai_stream(
    upstream: ByteStream,
    model: String,
) -> ByteStream {
    Box::pin(try_stream! {
        let mut reducer = AnthropicStreamReducer::new(model);
        let mut parser = SseParser::default();
        futures_util::pin_mut!(upstream);

        while let Some(chunk) = upstream.next().await {
            for event in parser.push(&chunk?) {
                for bytes in reducer.process_event(&event) {
                    yield bytes;
                }
            }
        }
        for event in parser.finish() {
            for bytes in reducer.process_event(&event) {
                yield bytes;
            }
        }
        // Streams that die before `message_stop` (worker crash, connection
        // cut) must still terminate the OpenAI side properly — a client
        // waiting on `[DONE]` that never comes hangs until its own timeout.
        for bytes in reducer.finalize() {
            yield bytes;
        }
    })
}

struct AnthropicSseEvent {
    event_type: String,
    data: Value,
}

#[derive(Debug, Default)]
struct SseParser {
    buffer: String,
}

impl SseParser {
    fn push(&mut self, chunk: &[u8]) -> Vec<AnthropicSseEvent> {
        // Strip CR on ingest so `\r\n\r\n` event separators (perfectly legal
        // SSE framing) parse the same as `\n\n` — including a CR/LF pair
        // split across two chunks.
        let text = String::from_utf8_lossy(chunk);
        self.buffer.reserve(text.len());
        for ch in text.chars() {
            if ch != '\r' {
                self.buffer.push(ch);
            }
        }
        let mut events = Vec::new();
        while let Some(idx) = self.buffer.find("\n\n") {
            let raw = self.buffer[..idx].to_string();
            self.buffer.drain(..idx + 2);
            if let Some(event) = parse_sse_event(&raw) {
                events.push(event);
            }
        }
        events
    }

    fn finish(&mut self) -> Vec<AnthropicSseEvent> {
        if self.buffer.trim().is_empty() {
            return Vec::new();
        }
        let raw = std::mem::take(&mut self.buffer);
        parse_sse_event(&raw).into_iter().collect()
    }
}

fn parse_sse_event(raw: &str) -> Option<AnthropicSseEvent> {
    let mut event_type = "message".to_string();
    let mut data_lines = Vec::new();

    for line in raw.lines() {
        if let Some(e) = line.strip_prefix("event:") {
            event_type = e.trim().to_string();
        } else if let Some(d) = line.strip_prefix("data:") {
            data_lines.push(d.trim());
        }
    }

    let data_str = data_lines.join("\n");
    if data_str.is_empty() { return None; }
    let data: Value = serde_json::from_str(&data_str).ok()?;
    Some(AnthropicSseEvent { event_type, data })
}

struct AnthropicStreamReducer {
    id: String,
    model: String,
    input_tokens: u64,
    output_tokens: u64,
    /// Anthropic content-block index → OpenAI tool_calls index. Anthropic
    /// indexes count *all* content blocks (a leading text block pushes the
    /// first tool_use to index 1+), while OpenAI tool_calls indexes count
    /// tool calls only, starting at 0 — so they must be remapped, not
    /// passed through.
    tool_indices: BTreeMap<u64, u64>,
    next_tool_index: u64,
    finish_sent: bool,
    done_sent: bool,
}

impl AnthropicStreamReducer {
    fn new(model: String) -> Self {
        Self {
            id: "chatcmpl-000".to_string(), // Will be overwritten by message_start
            model,
            input_tokens: 0,
            output_tokens: 0,
            tool_indices: BTreeMap::new(),
            next_tool_index: 0,
            finish_sent: false,
            done_sent: false,
        }
    }

    fn openai_tool_index(&mut self, anthropic_block_index: u64) -> u64 {
        if let Some(index) = self.tool_indices.get(&anthropic_block_index) {
            return *index;
        }
        let index = self.next_tool_index;
        self.next_tool_index += 1;
        self.tool_indices.insert(anthropic_block_index, index);
        index
    }

    fn process_event(&mut self, event: &AnthropicSseEvent) -> Vec<Bytes> {
        let mut out = Vec::new();
        match event.event_type.as_str() {
            "message_start" => {
                if let Some(msg_id) = event.data.pointer("/message/id").and_then(Value::as_str) {
                    self.id = msg_id.to_string();
                }
                if let Some(usage) = event.data.pointer("/message/usage") {
                    self.input_tokens = usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
                    self.output_tokens = usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
                }
                out.push(chat_completion_chunk(
                    &self.id,
                    &self.model,
                    json!({ "role": "assistant", "content": "" }),
                    None
                ));
            }
            "content_block_start" => {
                let block_type = event.data.pointer("/content_block/type").and_then(Value::as_str);
                if block_type == Some("tool_use") {
                    let tool_name = event.data.pointer("/content_block/name").and_then(Value::as_str).unwrap_or("tool");
                    let tool_id = event.data.pointer("/content_block/id").and_then(Value::as_str).unwrap_or("tool_call");
                    let index = event.data.get("index").and_then(Value::as_u64).unwrap_or(0);
                    let tool_index = self.openai_tool_index(index);

                    out.push(chat_completion_chunk(
                        &self.id,
                        &self.model,
                        json!({
                            "tool_calls": [{
                                "index": tool_index,
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": ""
                                }
                            }]
                        }),
                        None
                    ));
                }
            }
            "content_block_delta" => {
                let delta_type = event.data.pointer("/delta/type").and_then(Value::as_str);
                let index = event.data.get("index").and_then(Value::as_u64).unwrap_or(0);

                if delta_type == Some("text_delta") {
                    if let Some(text) = event.data.pointer("/delta/text").and_then(Value::as_str) {
                        out.push(chat_completion_chunk(
                            &self.id,
                            &self.model,
                            json!({ "content": text }),
                            None
                        ));
                    }
                } else if delta_type == Some("input_json_delta") {
                    if let Some(partial) = event.data.pointer("/delta/partial_json").and_then(Value::as_str) {
                        let tool_index = self.openai_tool_index(index);
                        out.push(chat_completion_chunk(
                            &self.id,
                            &self.model,
                            json!({
                                "tool_calls": [{
                                    "index": tool_index,
                                    "function": {
                                        "arguments": partial
                                    }
                                }]
                            }),
                            None
                        ));
                    }
                }
                // thinking_delta / signature_delta have no OpenAI equivalent
                // and are intentionally not surfaced as visible content.
            }
            "message_delta" => {
                if let Some(usage) = event.data.get("usage") {
                    if let Some(out_tokens) = usage.get("output_tokens").and_then(Value::as_u64) {
                        self.output_tokens = out_tokens;
                    }
                    if let Some(in_tokens) = usage.get("input_tokens").and_then(Value::as_u64) {
                        self.input_tokens = in_tokens;
                    }
                }
                if let Some(stop_reason) = event.data.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    // Single source of truth for the mapping — a private
                    // copy here previously disagreed with it (it sent
                    // stop_sequence → "content_filter", falsely claiming
                    // moderation kicked in on a clean stop).
                    let mapped_reason = anthropic_stop_reason_to_openai_finish_reason(stop_reason);
                    self.finish_sent = true;
                    out.push(chat_completion_chunk(
                        &self.id,
                        &self.model,
                        json!({}),
                        Some(mapped_reason.as_str())
                    ));
                }
            }
            "message_stop" => {
                out.extend(self.finalize());
            }
            "ping" => {
                // Forward keep-alives as an SSE comment — spec-legal, ignored
                // by OpenAI clients, and keeps idle proxies/load-balancers
                // from cutting the connection during long generations.
                out.push(Bytes::from(": ping\n\n"));
            }
            "error" => {
                // Surface upstream errors in the de-facto OpenAI stream error
                // shape instead of swallowing them.
                let error = event.data.get("error").cloned().unwrap_or_else(|| json!({
                    "type": "api_error",
                    "message": "upstream error"
                }));
                out.push(sse_event(json!({ "error": error })));
            }
            _ => {}
        }
        out
    }

    fn finalize(&mut self) -> Vec<Bytes> {
        let mut out = Vec::new();
        if self.done_sent {
            return out;
        }
        self.done_sent = true;
        if !self.finish_sent {
            self.finish_sent = true;
            out.push(chat_completion_chunk(&self.id, &self.model, json!({}), Some("stop")));
        }
        if self.input_tokens > 0 || self.output_tokens > 0 {
            out.push(usage_chunk(
                &self.id,
                &self.model,
                self.input_tokens,
                self.output_tokens,
            ));
        }
        out.push(done());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    fn run(chunks: Vec<&str>) -> String {
        let upstream: ByteStream = Box::pin(stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok(Bytes::from(c.to_string())))
                .collect::<Vec<Result<Bytes>>>(),
        ));
        let translated = translate_anthropic_to_openai_stream(upstream, "test-model".into());
        let out = futures::executor::block_on(async {
            let mut collected = Vec::new();
            futures_util::pin_mut!(translated);
            while let Some(item) = translated.next().await {
                collected.extend_from_slice(&item.unwrap());
            }
            collected
        });
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn tool_call_indices_are_remapped_to_start_at_zero() {
        // Text block at Anthropic index 0, tool_use at Anthropic index 1 —
        // the OpenAI tool_calls index must be 0, not 1.
        let out = run(vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"read\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]);
        // serde_json sorts object keys alphabetically, so "id" precedes "index"
        assert!(out.contains("\"id\":\"toolu_1\",\"index\":0"), "tool index must be remapped to 0: {out}");
        assert!(!out.contains("\"id\":\"toolu_1\",\"index\":1"));
        assert!(out.contains("\"finish_reason\":\"tool_calls\""));
        assert!(out.contains("data: [DONE]"));
    }

    #[test]
    fn stop_sequence_maps_to_stop_not_content_filter() {
        let out = run(vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"stop_sequence\"},\"usage\":{\"output_tokens\":1}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]);
        assert!(out.contains("\"finish_reason\":\"stop\""));
        assert!(!out.contains("content_filter"));
    }

    #[test]
    fn truncated_stream_still_emits_done() {
        let out = run(vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        ]);
        assert!(out.contains("\"finish_reason\":\"stop\""));
        assert!(out.ends_with("data: [DONE]\n\n"));
    }
}
