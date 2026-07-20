// ── Anthropic Messages API types ────────────────────────────────────────────

use serde::Serialize;
use serde_json::Value;

// ── Request-side: re-exported from the shared translation library ──────────

pub use ai_provider_converter::AnthropicRequest;

/// `AnthropicRequest::max_tokens` is `Option<u32>` in the library (not every
/// provider requires it up front), but Anthropic's real API spec requires
/// it. Centralizing the "what do we default to if a client somehow omits
/// it" decision here, rather than scattering `.unwrap_or(...)` calls.
pub fn required_max_tokens(req: &AnthropicRequest) -> u32 {
    req.max_tokens.unwrap_or(4096)
}

/// The library's `AnthropicRequest` catches unknown/provider-specific fields
/// (like Anthropic's `stop_sequences` and `top_k`, which have no OpenAI
/// equivalent) in `extra` via `#[serde(flatten)]` rather than dropping them.
/// These two fields matter enough for local-model behavior (see the FIX
/// notes in mod.rs about stop sequences) that we pull them back out here
/// instead of leaving them stranded in a generic JSON map.
pub fn extract_stop_sequences(req: &AnthropicRequest) -> Option<Vec<String>> {
    req.extra
        .get("stop_sequences")
        .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok())
}

pub fn extract_top_k(req: &AnthropicRequest) -> Option<u32> {
    req.extra.get("top_k").and_then(Value::as_u64).map(|v| v as u32)
}

// ── Response types: re-exported from the shared translation library ────────
//
// These used to be hand-duplicated local structs (`AnthropicResponse` /
// `AnthropicResponseBlock` / `AnthropicUsage`) that happened to serialize to
// the same JSON shape as the library's `response::anthropic` types, minus
// the `cache_creation_input_tokens`/`cache_read_input_tokens` usage fields
// (which real Anthropic responses do include). Re-exporting the library's
// types instead means the request AND response sides of this handler now
// come from the same single source of truth.
pub use ai_provider_converter::{AnthropicResponse, AnthropicUsage, AnthropicContentBlock};

// ── Models list (Anthropic format) ──────────────────────────────────────────
// Not a translation concern (no equivalent OpenAI-side type to map from/to),
// so this stays local rather than living in the shared library.

#[derive(Serialize)]
pub struct AnthropicModelsResponse {
    pub data:   Vec<AnthropicModelObject>,
    pub has_more: bool,
    pub first_id: Option<String>,
    pub last_id:  Option<String>,
}

#[derive(Serialize)]
pub struct AnthropicModelObject {
    pub id:           String,
    #[serde(rename = "type")]
    pub type_:        &'static str,
    pub display_name: String,
    pub created_at:   String,
}
