use bytes::Bytes;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn sse_event(data: Value) -> Bytes {
    let mut out = String::with_capacity(128);
    out.push_str("data: ");
    out.push_str(&data.to_string());
    out.push_str("\n\n");
    Bytes::from(out)
}

pub fn chat_completion_chunk(
    id: &str,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
) -> Bytes {
    sse_event(json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ]
    }))
}

pub fn done() -> Bytes {
    Bytes::from("data: [DONE]\n\n")
}

pub fn usage_chunk(
    id: &str,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Bytes {
    sse_event(json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }))
}
