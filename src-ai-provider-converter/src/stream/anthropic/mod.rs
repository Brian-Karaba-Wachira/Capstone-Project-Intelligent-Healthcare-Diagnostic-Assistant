pub mod parser;
use crate::response::anthropic::{AnthropicResponse, AnthropicUsage};
use bytes::Bytes;
use serde_json::{json, Value};

pub fn sse_event(event: &str, data: Value) -> Bytes {
    let mut out = String::with_capacity(128);
    out.push_str("event: ");
    out.push_str(event);
    out.push('\n');
    out.push_str("data: ");
    out.push_str(&data.to_string());
    out.push_str("\n\n");
    Bytes::from(out)
}

pub fn message_start(message_id: &str, model: &str) -> Bytes {
    sse_event(
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0
                }
            }
        }),
    )
}

pub fn content_block_start(index: usize, block: Value) -> Bytes {
    sse_event(
        "content_block_start",
        json!({
            "type": "content_block_start",
            "index": index,
            "content_block": block
        }),
    )
}

pub fn content_block_delta(index: usize, delta: Value) -> Bytes {
    sse_event(
        "content_block_delta",
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": delta
        }),
    )
}

pub fn content_block_stop(index: usize) -> Bytes {
    sse_event(
        "content_block_stop",
        json!({
            "type": "content_block_stop",
            "index": index
        }),
    )
}

pub fn message_delta(stop_reason: Option<&str>, usage: AnthropicUsage) -> Bytes {
    sse_event(
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": null
            },
            "usage": usage
        }),
    )
}

pub fn message_stop() -> Bytes {
    sse_event("message_stop", json!({ "type": "message_stop" }))
}

pub fn ping() -> Bytes {
    sse_event("ping", json!({ "type": "ping" }))
}

pub fn error(error_type: &str, message: &str) -> Bytes {
    sse_event(
        "error",
        json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        }),
    )
}

pub fn response_json(response: AnthropicResponse) -> Value {
    serde_json::to_value(response).unwrap_or_else(|_| json!({}))
}
