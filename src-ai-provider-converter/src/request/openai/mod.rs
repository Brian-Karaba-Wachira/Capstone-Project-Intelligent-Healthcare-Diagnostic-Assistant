use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The canonical OpenAI-shaped chat request — used both as the *output* of
/// Anthropic→OpenAI translation and, now, as the *input* type for any
/// OpenAI-compatible client (Cursor, Claude Code in OpenAI mode, etc.)
/// talking directly to an OpenAI-shaped endpoint.
///
/// Using this one type for both directions is the whole point: a client
/// hitting the native OpenAI endpoint ends up with the exact same struct
/// shape a translated Anthropic request would produce, so both paths can
/// share identical downstream handling instead of drifting apart.
///
/// `messages`/`tools`/`tool_choice`/`stop` are kept as loose `Value` rather
/// than strict typed enums deliberately: real-world OpenAI-compatible
/// clients vary in exactly what they put in these fields, and a strict
/// untagged enum fails the *entire* request the moment one client sends a
/// shape that doesn't match a known variant. `extra` catches everything
/// else (legacy `functions`/`function_call`, `n`, `user`, vendor-specific
/// fields, ...) so an unrecognized field never causes a hard parse failure.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// String or array-of-strings per OpenAI's spec — left untyped for the
    /// same reason as `messages`/`tools` above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    /// Legacy `functions`/`function_call`, `n`, `user`, and any other field
    /// a real client sends land here instead of failing deserialization.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}
