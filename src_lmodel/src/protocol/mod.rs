// ═════════════════════════════════════════════════════════════════════════════
//  lmodel/src/protocol/mod.rs
//
//  Wire protocol between lmodel (worker) and brain.
//
//  UPDATED: brought back into field-for-field parity with brain's
//  src/core/protocol.rs after it drifted out of sync. The drift was silently
//  breaking the tunnel (missing required fields on Chunk/Done caused brain's
//  NdjsonReader::recv() to error on every message, tearing down the whole
//  connection) and silently dropping tool-calling (Task/ChatMessage had no
//  tools/tool_choice/tool_calls fields, so they vanished before reaching
//  bridge.rs). See inline comments below for the specific fields added.
// ═════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Must mirror brain/src/core/protocol.rs exactly ───────────────────────────
// Brain's outer enum is named `WorkerMessage`; this one is kept as `Message`
// for historical reasons, but the wire shape (tag = "type", snake_case variant
// names) is identical, and every payload struct below now matches brain's
// field-for-field. Drift here is what was silently breaking the tunnel and
// dropping tool-calling — keep these two files in lockstep.

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Message {
    Register(RegisterMessage),
    Heartbeat(HeartbeatMessage),
    Ping,
    Task(TaskMessage),
    Chunk(ChunkMessage),
    Done(DoneMessage),
    Error(ErrorMessage),
    Cancel(CancelMessage),
}

// ── Brain → lmodel ────────────────────────────────────────────────────────────

/// Mirrors brain's `TaskBundle`. Previously missing: model, top_p, tools,
/// tool_choice, parallel_tool_calls, stop — all silently dropped on arrival.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMessage {
    pub request_id:  String,
    pub session_id:  String,
    pub model:       String,
    pub messages:    Vec<ChatMessage>,
    pub max_tokens:  u32,
    pub temperature: f32,
    pub top_p:       Option<f32>,
    pub top_k:       Option<u32>,
    pub stream:      bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools:               Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice:         Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop:                Option<Vec<String>>,
}

/// Mirrors brain's `Message` (chat message). Previously just {role, content}
/// — now carries tool_calls/tool_call_id so assistant tool-call history and
/// tool-result messages survive the brain → worker hop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role:    String,
    pub content: MessageContent,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Content can be a plain string or a list of typed content parts (vision etc).
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

    /// See brain/src/core/protocol.rs::MessageContent::to_text — kept in
    /// lockstep. Concatenates text-typed parts instead of discarding them.
    #[allow(dead_code)]
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
    pub text: Option<String>,
}

impl ContentPart {
    fn text_len(&self) -> usize {
        self.text.as_deref().map(|t| t.len()).unwrap_or(0)
    }
}

// ── Tool calling types — mirrors brain exactly ───────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // always "function"
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>, // JSON Schema object
}

/// "auto" | "required" | "none" or {"type":"function","function":{"name":"..."}}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Specific(SpecificToolChoice),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecificToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: ToolChoiceFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String, // "function"
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String, // JSON string
}

// ── lmodel → Brain ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterMessage {
    pub worker_id: String,
    pub model: String,
    pub gpu: String,
    pub vram_free_mb: u32,
    pub max_context: u32,
    pub active_requests: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMessage {
    pub worker_id:       String,
    pub model:           String,   // model name sent on every heartbeat
    pub vram_free_mb:    u32,
    pub active_requests: u32,
}

/// Mirrors brain's `ChunkMessage`. Previously missing `is_tool_call` —
/// brain's field has no `Option`/default, so its absence caused a hard
/// deserialize failure on *every* chunk, which `tunnel.rs` treats as a
/// fatal read error and tears down the whole connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMessage {
    pub request_id: String,
    pub delta: String, // raw "data: ..." SSE line, passed through verbatim
    pub is_tool_call: bool,
}

/// Mirrors brain's `DoneMessage`. Previously missing finish_reason,
/// prompt_tokens, comp_tokens — same fatal-deserialize problem as above,
/// just triggered at end-of-stream instead of on the first chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoneMessage {
    pub request_id: String,
    pub finish_reason: FinishReason,
    pub prompt_tokens: u32,
    pub comp_tokens: u32,
}

/// Mirrors brain's `ErrorMessage`. Previously missing `code`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    pub request_id: String,
    pub message: String,
    pub code: u16, // HTTP status the brain should return to the client
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelMessage {
    pub request_id: String,
}