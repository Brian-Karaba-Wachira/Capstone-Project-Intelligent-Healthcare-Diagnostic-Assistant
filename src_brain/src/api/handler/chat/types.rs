// ── OpenAI /v1/chat/completions request/response types ─────────────────────
//
// The inbound request type is now ai_provider_converter's ChatCompletionsRequest
// — the exact same struct produced when a request is translated from
// Anthropic. Using one shared type for both "already OpenAI-shaped" and
// "translated from Anthropic" means a client talking OpenAI natively (Cursor,
// Claude Code in OpenAI-compat mode, etc.) and the Anthropic handler converge
// on identical structure before anything touches the worker. It's also far
// more lenient than the old hand-rolled `InboundMessage`/`InboundContent`
// untagged-enum types — those failed the ENTIRE request the moment a real
// client sent a message/tool shape that didn't cleanly match a known variant,
// which is almost certainly what was crashing OpenAI-compat CLI clients here
// while the same clients worked fine against ai-provider-converter directly.

pub use ai_provider_converter::ChatCompletionsRequest as ChatRequest;

use serde::Serialize;
use crate::core::protocol::ToolCall;

// ── Outbound OpenAI response ──────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ChatResponse {
    pub id:      String,
    pub object:  &'static str,
    pub created: u64,
    pub model:   String,
    pub choices: Vec<Choice>,
    pub usage:   Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index:         u32,
    pub message:       AssistantMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct AssistantMessage {
    pub role:       &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content:    Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens:     u32,
    pub completion_tokens: u32,
    pub total_tokens:      u32,
}

// ── /v1/models response ───────────────────────────────────────────────────────
// Re-exported at the `chat::` root by mod.rs — api/mod.rs imports these as
// `chat::ModelObject` / `chat::ModelsResponse`, so the public path must not change.

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data:   Vec<ModelObject>,
}

#[derive(Serialize)]
pub struct ModelObject {
    pub id:       String,
    pub object:   &'static str,
    pub created:  u64,
    pub owned_by: &'static str,
}
