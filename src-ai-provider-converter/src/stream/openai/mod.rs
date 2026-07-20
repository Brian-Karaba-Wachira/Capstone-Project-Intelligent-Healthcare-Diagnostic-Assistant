pub mod builders;
use crate::error::{ConverterError, Result};
use crate::response::anthropic::{AnthropicResponse, AnthropicUsage, AnthropicContentBlock};
use crate::stream::anthropic as anthropic_stream;
use async_stream::try_stream;
use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::pin::Pin;
use uuid::Uuid;

pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;

pub fn translate_chat_stream(
    upstream: ByteStream,
    model: String,
    request_id: Option<String>,
) -> ByteStream {
    Box::pin(try_stream! {
        let message_id = format!("msg_{}", Uuid::new_v4().simple());
        let mut reducer = ChatStreamReducer::new();
        yield anthropic_stream::message_start(&message_id, &model);
        let mut parser = SseParser::default();
        futures_util::pin_mut!(upstream);
        while let Some(chunk) = upstream.next().await {
            for event in parser.push(&chunk?, request_id.as_deref()) {
                for bytes in reducer.process_event(&event) {
                    yield bytes;
                }
            }
        }
        for event in parser.finish(request_id.as_deref()) {
            for bytes in reducer.process_event(&event) {
                yield bytes;
            }
        }
        for bytes in reducer.finish_events() {
            yield bytes;
        }
    })
}

pub async fn accumulate_chat_response(
    upstream: ByteStream,
    model: String,
) -> Result<AnthropicResponse> {
    let value = read_json_body(upstream).await?;
    chat_response_from_value(value, model)
}

async fn read_json_body(upstream: ByteStream) -> Result<Value> {
    futures_util::pin_mut!(upstream);
    let mut bytes = BytesMut::new();
    while let Some(chunk) = upstream.next().await {
        bytes.extend_from_slice(&chunk?);
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn chat_response_from_value(value: Value, model: String) -> Result<AnthropicResponse> {
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| ConverterError::InvalidRequest("custom OpenAI chat response had no choices".into()))?;
    let message = choice.get("message").ok_or_else(|| {
        ConverterError::InvalidRequest("custom OpenAI chat response had no message".into())
    })?;
    let mut content = Vec::new();
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            content.push(AnthropicContentBlock {
                kind: "text".into(),
                text: Some(text.to_string()),
                id: None,
                name: None,
                input: None,
            });
        }
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            let function = tool_call.get("function").unwrap_or(&Value::Null);
            content.push(AnthropicContentBlock {
                kind: "tool_use".into(),
                text: None,
                id: Some(
                    tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| format!("toolu_{}", Uuid::new_v4().simple())),
                ),
                name: Some(
                    function
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("tool")
                        .to_string(),
                ),
                input: Some(parse_tool_arguments(
                    function
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}"),
                )),
            });
        }
    }
    let usage = usage_from_value(value.get("usage"));
    let stop_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .map(map_chat_finish_reason);
    Ok(AnthropicResponse {
        id: format!("msg_{}", Uuid::new_v4().simple()),
        kind: "message".into(),
        role: "assistant".into(),
        model,
        content,
        stop_reason,
        stop_sequence: None,
        usage,
    })
}

#[derive(Debug, Default)]
struct SseParser {
    buffer: String,
}

impl SseParser {
    fn push(&mut self, chunk: &[u8], request_id: Option<&str>) -> Vec<Value> {
        // Strip CR on ingest so `\r\n\r\n` event separators (legal SSE
        // framing some OpenAI-compatible servers emit) parse the same as
        // `\n\n` — including a CR/LF pair split across two chunks.
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
            if let Some(value) = parse_sse_event(&raw, request_id) {
                events.push(value);
            }
        }
        events
    }

    fn finish(&mut self, request_id: Option<&str>) -> Vec<Value> {
        if self.buffer.trim().is_empty() {
            return Vec::new();
        }
        let raw = std::mem::take(&mut self.buffer);
        parse_sse_event(&raw, request_id).into_iter().collect()
    }
}

fn parse_sse_event(raw: &str, _request_id: Option<&str>) -> Option<Value> {
    let data_lines = raw
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect::<Vec<_>>();
    let data = data_lines.join("\n");
    if data.is_empty() || data == "[DONE]" {
        return None;
    }
    serde_json::from_str(&data).ok()
}

/// Which Anthropic content block is currently open. Anthropic's stream
/// framing is strictly sequential — every `content_block_start` must be
/// preceded by a `content_block_stop` for the previous block — so exactly
/// one block can be open at a time. (The previous reducer left the text
/// block open across tool blocks and appended post-tool-call text to it,
/// which is invalid framing on the Anthropic side.)
#[derive(Debug, Clone, Copy)]
enum OpenBlock {
    Text(usize),
    Tool { upstream: usize, block: usize },
}

struct ChatStreamReducer {
    next_block_index: usize,
    open_block: Option<OpenBlock>,
    /// Upstream tool_calls index → whether we've already started a block for
    /// it, so a re-announced name mid-call doesn't open a duplicate block.
    seen_tools: BTreeMap<usize, usize>,
    usage: AnthropicUsage,
    stop_reason: Option<String>,
}

impl ChatStreamReducer {
    fn new() -> Self {
        Self {
            next_block_index: 0,
            open_block: None,
            seen_tools: BTreeMap::new(),
            usage: AnthropicUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
            stop_reason: None,
        }
    }

    fn process_event(&mut self, event: &Value) -> Vec<Bytes> {
        let mut out = Vec::new();
        // OpenAI's include_usage mode sends `"usage": null` on every chunk
        // and the real numbers only on the last one — only accept objects so
        // the nulls don't zero out totals we already captured.
        if let Some(usage) = event.get("usage") {
            if usage.is_object() {
                self.usage = usage_from_value(Some(usage));
            }
        }
        // Surface worker/upstream error frames ({"error": {...}}) as a real
        // Anthropic error event instead of silently dropping them.
        if let Some(error) = event.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("upstream error");
            out.push(anthropic_stream::error("api_error", message));
            return out;
        }
        let Some(choice) = event
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return out;
        };
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.stop_reason = Some(map_chat_finish_reason(reason));
        }
        let Some(delta) = choice.get("delta") else {
            return out;
        };
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                let index = self.ensure_text_block(&mut out);
                out.push(anthropic_stream::content_block_delta(
                    index,
                    json!({ "type": "text_delta", "text": text }),
                ));
            }
        }
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in tool_calls {
                self.process_tool_call_delta(call, &mut out);
            }
        }
        out
    }

    fn close_open_block(&mut self, out: &mut Vec<Bytes>) {
        if let Some(block) = self.open_block.take() {
            let index = match block {
                OpenBlock::Text(index) => index,
                OpenBlock::Tool { block, .. } => block,
            };
            out.push(anthropic_stream::content_block_stop(index));
        }
    }

    fn ensure_text_block(&mut self, out: &mut Vec<Bytes>) -> usize {
        if let Some(OpenBlock::Text(index)) = self.open_block {
            return index;
        }
        // A tool block is open (or nothing is) — close it and start a fresh
        // text block, so text arriving after a tool call gets its own block
        // instead of being appended to one that conceptually ended.
        self.close_open_block(out);
        let index = self.next_block_index;
        self.next_block_index += 1;
        self.open_block = Some(OpenBlock::Text(index));
        out.push(anthropic_stream::content_block_start(
            index,
            json!({ "type": "text", "text": "" }),
        ));
        index
    }

    fn process_tool_call_delta(&mut self, call: &Value, out: &mut Vec<Bytes>) {
        let upstream_index = call
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        let function = call.get("function").unwrap_or(&Value::Null);
        if !self.seen_tools.contains_key(&upstream_index) {
            self.close_open_block(out);
            let block_index = self.next_block_index;
            self.next_block_index += 1;
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("toolu_{}", Uuid::new_v4().simple()));
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("tool")
                .to_string();
            self.seen_tools.insert(upstream_index, block_index);
            self.open_block = Some(OpenBlock::Tool { upstream: upstream_index, block: block_index });
            out.push(anthropic_stream::content_block_start(
                block_index,
                json!({ "type": "tool_use", "id": id, "name": name, "input": {} }),
            ));
        }
        if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
            if !arguments.is_empty() {
                // Argument deltas are only valid on the currently open tool
                // block — OpenAI streams each call's arguments contiguously,
                // so anything else is out-of-order framing we must not
                // replay into an already-stopped block.
                if let Some(OpenBlock::Tool { upstream, block }) = self.open_block {
                    if upstream == upstream_index {
                        out.push(anthropic_stream::content_block_delta(
                            block,
                            json!({ "type": "input_json_delta", "partial_json": arguments }),
                        ));
                    }
                }
            }
        }
    }

    fn finish_events(mut self) -> Vec<Bytes> {
        let mut out = Vec::new();
        self.close_open_block(&mut out);
        // A stream that ended without a finish_reason (worker died, [DONE]
        // arrived early) still needs a concrete stop_reason — Anthropic
        // clients treat null as malformed. "end_turn" is the least-wrong
        // default for a stream that terminated without saying why.
        let stop_reason = self
            .stop_reason
            .clone()
            .unwrap_or_else(|| "end_turn".to_string());
        out.push(anthropic_stream::message_delta(
            Some(&stop_reason),
            self.usage,
        ));
        out.push(anthropic_stream::message_stop());
        out
    }
}

fn usage_from_value(value: Option<&Value>) -> AnthropicUsage {
    let input_tokens = value
        .and_then(|value| value.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = value
        .and_then(|value| value.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    AnthropicUsage {
        input_tokens,
        output_tokens,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: value
            .and_then(|value| value.pointer("/prompt_tokens_details/cached_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
    }
}

fn map_chat_finish_reason(reason: &str) -> String {
    crate::utils::finish_reason::openai_finish_reason_to_anthropic_stop_reason(reason)
}

fn parse_tool_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
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
        let translated = translate_chat_stream(upstream, "test-model".into(), None);
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
    fn text_then_tool_call_produces_sequential_closed_blocks() {
        let out = run(vec![
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Let me check.\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"p\\\":1}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        ]);
        // The text block (index 0) must be STOPPED before the tool block
        // (index 1) starts — Anthropic framing is strictly sequential.
        // (serde_json sorts object keys, so "index" precedes "type".)
        let text_stop = out.find("event: content_block_stop\ndata: {\"index\":0").expect("text block stop");
        let tool_start = out.find("\"type\":\"tool_use\"").expect("tool block start");
        assert!(text_stop < tool_start, "text block must close before tool block opens");
        assert!(out.contains("\"partial_json\":\"{\\\"p\\\":1}\""));
        assert!(out.contains("\"stop_reason\":\"tool_use\""));
        assert!(out.contains("message_stop"));
    }

    #[test]
    fn crlf_framing_and_missing_finish_reason_still_terminate_cleanly() {
        let out = run(vec![
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\r\n\r\n",
        ]);
        assert!(out.contains("\"text\":\"hi\""), "CRLF-framed event must still parse: {out}");
        // No finish_reason ever arrived — stop_reason must default, not be null.
        assert!(out.contains("\"stop_reason\":\"end_turn\""));
        assert!(out.contains("message_stop"));
    }

    #[test]
    fn upstream_error_frame_becomes_anthropic_error_event() {
        let out = run(vec![
            "data: {\"error\":{\"message\":\"model exploded\",\"type\":\"server_error\"}}\n\n",
        ]);
        assert!(out.contains("event: error"));
        assert!(out.contains("model exploded"));
    }
}
