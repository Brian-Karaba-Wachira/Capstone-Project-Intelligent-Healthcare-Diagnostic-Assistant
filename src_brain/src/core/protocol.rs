use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Shared message type (brain ↔ worker) ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum WorkerMessage {
    Register(RegisterMsg),
    Heartbeat(HeartbeatMsg),
    Ping,
    Task(TaskBundle),
    Chunk(ChunkMessage),
    Done(DoneMessage),
    Error(ErrorMessage),
    Cancel(CancelMessage),
}

// ── Brain → Worker ────────────────────────────────────────────────────────────

/// Full task sent to worker — carries everything llama-server needs.
/// Previously missing: tools, tool_choice, parallel_tool_calls, model, stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBundle {
    pub request_id:           String,
    pub session_id:           String,
    pub model:                String,          // resolved model name for llama-server
    pub messages:             Vec<Message>,
    pub max_tokens:           u32,
    pub temperature:          f32,
    pub top_p:                Option<f32>,
    pub top_k:                Option<u32>,
    pub stream:               bool,

    // ── Tool calling — the critical fields most proxies drop ─────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools:                Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice:          Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls:  Option<bool>,

    // ── Stop sequences ───────────────────────────────────────────────────────
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop:                 Option<Vec<String>>,
}

// ── Worker → Brain ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterMsg {
    pub worker_id:        String,
    pub model:            String,
    pub gpu:              String,
    pub vram_free_mb:     u32,
    pub max_context:      u32,
    pub active_requests:  u32,
    /// Which wire format llama-server (or whatever sits behind lmodel) speaks
    /// natively — almost always "openai" today, but lets a future worker
    /// self-report "anthropic" (or something else) so the gateway knows
    /// whether a translation pass is even needed for that worker, instead
    /// of assuming every worker is OpenAI-shaped.
    #[serde(default = "default_worker_api_type")]
    pub api_type:         String,
}

fn default_worker_api_type() -> String {
    "openai".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMsg {
    pub worker_id:       String,
    pub model:           String,   // model name echoed on every heartbeat
    pub vram_free_mb:    u32,
    pub active_requests: u32,
}

/// A chunk from the worker. Can be either a text delta or a tool call delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMessage {
    pub request_id:   String,
    pub delta:        String,     // raw SSE "data: ..." line from llama-server
    pub is_tool_call: bool,       // hint: did worker detect a tool_call in this chunk?
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoneMessage {
    pub request_id:    String,
    pub finish_reason: FinishReason,
    pub prompt_tokens: u32,
    pub comp_tokens:   u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    pub request_id: String,
    pub message:    String,
    pub code:       u16,    // HTTP status to return to client
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelMessage {
    pub request_id: String,
}

// ── Shared primitives ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role:       String,
    pub content:    MessageContent,

    /// Present when role == "assistant" and model made tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Present when role == "tool" (tool result being sent back)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Content can be a plain string or a list of typed content parts (vision etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn as_str(&self) -> &str {
        match self {
            MessageContent::Text(s) => s.as_str(),
            MessageContent::Parts(_) => "",
        }
    }

    /// Like `as_str`, but for `Parts` returns the concatenated text of every
    /// text-typed part instead of throwing it away. `as_str` can't do this
    /// itself since it has to return a borrowed `&str`; callers that persist
    /// or forward message content (history, memory, logging) should prefer
    /// this over `as_str` so multipart (vision/mixed) messages don't get
    /// silently recorded as blank.
    pub fn to_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter(|p| p.part_type == "text")
                .filter_map(|p| p.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            MessageContent::Text(s) => s.len(),
            MessageContent::Parts(parts) => parts.iter().map(|p| p.text_len()).sum(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text:      Option<String>,
}

impl ContentPart {
    fn text_len(&self) -> usize {
        self.text.as_deref().map(|t| t.len()).unwrap_or(0)
    }
}

// ── Tool calling types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,   // always "function"
    pub function:  FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name:        String,
    pub description: Option<String>,
    pub parameters:  Option<Value>,  // JSON Schema object
}

/// tool_choice can be "auto", "required", "none", or {"type":"function","function":{"name":"..."}}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),                      // "auto" | "required" | "none"
    Specific(SpecificToolChoice),      // {"type":"function","function":{"name":"fn_name"}}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecificToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function:    ToolChoiceFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id:       String,
    #[serde(rename = "type")]
    pub call_type: String,   // "function"
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name:      String,
    pub arguments: String,   // JSON string
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Error,
}

impl FinishReason {
    pub fn as_str(&self) -> &str {
        match self {
            FinishReason::Stop          => "stop",
            FinishReason::ToolCalls     => "tool_calls",
            FinishReason::Length        => "length",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Error         => "stop",
        }
    }
}

// Memory, skills, and session-bridge types have been intentionally removed.
// Agentic CLI clients (Claude Code and friends) own conversation history,
// memory, and skill injection on their own machine — brain is a stateless
// translate-and-route proxy and must not carry any of that logic or state.

// ── OpenAI-shaped JSON → internal protocol bridge ───────────────────────────
//
// Both the OpenAI chat handler and the Anthropic handler (via
// ai-provider-converter's Anthropic→OpenAI translation) end up with a plain
// OpenAI chat-message JSON list. This is the single shared place that turns
// one of those `serde_json::Value` messages into our internal `Message`, so
// both call paths converge on identical parsing instead of drifting apart.
pub fn message_from_openai_value(v: &Value) -> Message {
    let role = v.get("role").and_then(|r| r.as_str()).unwrap_or("user").to_string();

    let content = match v.get("content") {
        Some(Value::String(s)) => MessageContent::Text(s.clone()),
        Some(Value::Array(parts)) => {
            let text = parts.iter()
                .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            MessageContent::Text(text)
        }
        _ => MessageContent::Text(String::new()),
    };

    let tool_calls = v.get("tool_calls")
        .and_then(|tc| serde_json::from_value::<Vec<ToolCall>>(tc.clone()).ok())
        .filter(|v| !v.is_empty());

    let tool_call_id = v.get("tool_call_id")
        .and_then(|id| id.as_str())
        .map(|s| s.to_string());

    Message { role, content, tool_calls, tool_call_id }
}

/// Turn one OpenAI-shaped tool JSON value (`{"type":"function","function":{...}}`)
/// into our internal `Tool`. Shared by both the Anthropic handler (fed by
/// ai-provider-converter's translation output) and the native OpenAI/chat
/// handler (fed directly by the client), so a tool looks identical to the
/// worker regardless of which API dialect it arrived through.
pub fn openai_tool_json_to_protocol_tool(value: &Value) -> Option<Tool> {
    let function = value.get("function")?;
    Some(Tool {
        tool_type: "function".into(),
        function: FunctionDef {
            name:        function.get("name")?.as_str()?.to_string(),
            description: function.get("description").and_then(Value::as_str).map(String::from),
            parameters:  function.get("parameters").cloned(),
        },
    })
}

/// Same sharing rationale as above, for `tool_choice`.
pub fn openai_tool_choice_json_to_protocol(value: &Value) -> ToolChoice {
    match value {
        Value::String(mode) => ToolChoice::Mode(mode.clone()),
        Value::Object(_) => {
            if let Some(name) = value.pointer("/function/name").and_then(Value::as_str) {
                ToolChoice::Specific(SpecificToolChoice {
                    choice_type: "function".into(),
                    function: ToolChoiceFunction { name: name.to_string() },
                })
            } else {
                ToolChoice::Mode("auto".into())
            }
        }
        _ => ToolChoice::Mode("auto".into()),
    }
}

/// Rough prompt-token estimate for a request, used ONLY to decide whether a
/// request should be rejected up front (before it ever reaches a worker) —
/// not as a billing-accurate count. We have no access to the target model's
/// real tokenizer here (workers can be any local GGUF), so this uses the
/// standard "~4 chars per token" heuristic on the combined message text
/// plus a flat per-message overhead for role/formatting tokens, and adds
/// tool definitions (names/descriptions/JSON-schema parameters) since those
/// count against the same context window.
///
/// Deliberately conservative (rounds UP): a false "too large" rejection is
/// cheap (client compacts and retries), but a false "fits" that gets
/// rejected by the worker instead surfaces as an opaque 500 several hops
/// downstream, or worse — silently, if the worker's own error framing
/// doesn't survive the trip back (see lmodel bridge fix).
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;
const PER_MESSAGE_OVERHEAD_TOKENS: u32 = 4;

pub fn estimate_prompt_tokens(messages: &[Message], tools: Option<&Vec<Tool>>) -> u32 {
    let mut chars: usize = 0;

    for m in messages {
        chars += m.content.len();
        chars += m.role.len();
        if let Some(calls) = &m.tool_calls {
            for c in calls {
                chars += c.function.name.len() + c.function.arguments.len();
            }
        }
    }

    let mut estimate = (chars.div_ceil(CHARS_PER_TOKEN_ESTIMATE)) as u32
        + PER_MESSAGE_OVERHEAD_TOKENS * messages.len() as u32;

    if let Some(tools) = tools {
        for t in tools {
            let mut tchars = t.function.name.len();
            if let Some(d) = &t.function.description {
                tchars += d.len();
            }
            if let Some(p) = &t.function.parameters {
                tchars += p.to_string().len();
            }
            estimate += (tchars.div_ceil(CHARS_PER_TOKEN_ESTIMATE)) as u32 + PER_MESSAGE_OVERHEAD_TOKENS;
        }
    }

    estimate
}

/// OpenAI's `stop` field can be a bare string or an array of strings.
/// Shared so both handlers normalize it identically.
pub fn stop_sequences_from_value(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(items) => {
            let strings: Vec<String> = items.iter().filter_map(|v| v.as_str().map(String::from)).collect();
            (!strings.is_empty()).then_some(strings)
        }
        _ => None,
    }
}