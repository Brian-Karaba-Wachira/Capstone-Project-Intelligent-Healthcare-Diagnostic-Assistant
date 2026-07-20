use serde_json::Value;
use uuid::Uuid;

use crate::core::protocol::{
    Message, Tool, ToolChoice, ToolCall, ToolCallFunction,
    message_from_openai_value, openai_tool_json_to_protocol_tool, openai_tool_choice_json_to_protocol,
};

use ai_provider_converter::{AnthropicTool, TranslatedChatRequest};

use super::super::common::truncate_for_log;
use super::types::AnthropicRequest;

// ─── Translation: Anthropic → internal (via ai-provider-converter) ─────────
//
// This is the one and only place that turns a parsed Anthropic request into
// what the rest of the handler needs. Instead of hand-rolling the
// message/tool/tool_choice conversion here, we call straight into
// ai-provider-converter's `translate_anthropic_to_openai_request_ex` and
// reshape *its* OpenAI-JSON output into our internal `Message`/`Tool`/
// `ToolChoice` types using the SAME adapters (`openai_tool_json_to_protocol_tool`,
// `openai_tool_choice_json_to_protocol`, `message_from_openai_value`) that
// the native OpenAI/chat handler uses on its own inbound requests — so a
// tool or message ends up identically shaped for the worker regardless of
// which API dialect the client spoke.

pub struct TranslatedAnthropicRequest {
    pub messages:     Vec<Message>,
    pub tools:        Option<Vec<Tool>>,
    pub tool_choice:  Option<ToolChoice>,
    /// Hosted/server-side tools (e.g. Anthropic's `web_search_*`) that the
    /// library stripped out of the outgoing tool list because an
    /// OpenAI-shaped worker can't execute them as function tools. Purely
    /// informational — brain does not execute these itself (yet); surfacing
    /// them is the hook point for wiring in a real web_search/web_fetch
    /// implementation later without touching this translation path again.
    pub hosted_tools: Vec<AnthropicTool>,
}

pub fn translate_anthropic_request(
    req: &AnthropicRequest,
    upstream_model: &str,
) -> Result<TranslatedAnthropicRequest, String> {
    let TranslatedChatRequest { request, hosted_tools } =
        ai_provider_converter::translate_anthropic_to_openai_request_ex(req, upstream_model)
            .map_err(|e| e.to_string())?;

    let messages = request.messages.iter().map(message_from_openai_value).collect();

    let tools = request.tools.map(|tools| {
        tools.iter().filter_map(openai_tool_json_to_protocol_tool).collect::<Vec<_>>()
    }).filter(|t: &Vec<Tool>| !t.is_empty());

    let tool_choice = request.tool_choice.as_ref().map(openai_tool_choice_json_to_protocol);

    Ok(TranslatedAnthropicRequest { messages, tools, tool_choice, hosted_tools })
}

pub fn anthropic_models_response(model_ids: Vec<String>) -> super::types::AnthropicModelsResponse {
    let objects: Vec<super::types::AnthropicModelObject> = model_ids.into_iter().map(|id| {
        super::types::AnthropicModelObject {
            id: id.clone(),
            type_: "model",
            display_name: id,
            created_at: "2024-01-01T00:00:00Z".into(),
        }
    }).collect();
    let first_id = objects.first().map(|o| o.id.clone());
    let last_id  = objects.last().map(|o| o.id.clone());
    super::types::AnthropicModelsResponse {
        data: objects,
        has_more: false,
        first_id,
        last_id,
    }
}

// ── Shared streaming/blocking delta accumulator ─────────────────────────────
// Parses worker output (OpenAI-shaped SSE deltas coming back from llama.cpp),
// which is not something ai-provider-converter covers today. This is now the
// ONLY place that parses a delta's `tool_calls` array and decides "is this a
// new tool call, or more of the one we're already building". `streaming.rs`
// used to re-derive that same decision independently from the raw JSON (to
// know when to open/close Anthropic content blocks), which meant two call
// sites could disagree about block boundaries whenever a worker's streaming
// shape deviated from OpenAI's. Now `accumulate_delta` reports the boundary
// decisions it made as `ToolStreamEvent`s, and `streaming.rs` just replays
// them — one parse, one state machine, one source of truth.

/// A tool-call-related event derived from a single worker delta, in the order
/// it should be replayed as Anthropic content-block SSE events.
#[derive(Debug, Clone)]
pub enum ToolStreamEvent {
    /// A new tool call began — close whatever block was previously open and
    /// start a new `tool_use` content block.
    Started { id: String, name: String },
    /// More of the current tool call's arguments arrived — emit an
    /// `input_json_delta` on the currently open tool block.
    ArgsDelta(String),
}

pub fn accumulate_delta(
    delta_line: &str,
    full_text: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    current_tool_call: &mut Option<ToolCall>,
    current_tool_index: &mut Option<u64>,
) -> (Option<String>, Vec<ToolStreamEvent>) {
    let trimmed = delta_line.strip_prefix("data: ").unwrap_or(delta_line).trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return (None, Vec::new());
    }

    let val = match serde_json::from_str::<Value>(trimmed) {
        Ok(v) => v,
        Err(e) => {
            log::warn!(
                "[unhandled-translation] worker delta failed to parse as JSON ({}), dropping: {}",
                e, truncate_for_log(trimmed),
            );
            return (None, Vec::new());
        }
    };
    let Some(delta) = val.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("delta")) else {
        log::warn!(
            "[unhandled-translation] worker delta missing choices[0].delta — worker may be sending a shape this bridge doesn't recognize: {}",
            truncate_for_log(trimmed),
        );
        return (None, Vec::new());
    };

    let mut new_text = None;
    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
        if !content.is_empty() {
            full_text.push_str(content);
            new_text = Some(content.to_string());
        }
    }

    let mut events = Vec::new();

    if let Some(tc_array) = delta.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tc_array {
            let index = tc.get("index").and_then(Value::as_u64);
            let Some(function) = tc.get("function") else { continue };

            if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                // Is this actually a NEW tool call, or just a duplicate/echoed
                // `name` on the call we're already building? We can't always
                // trust `index` — plenty of OpenAI-compatible local servers
                // omit it entirely — so:
                //  - nothing tracked yet -> definitely new.
                //  - both sides have an index -> trust it.
                //  - no usable index -> only treat it as new if the call
                //    we're currently building already has some arguments.
                //    A genuinely fresh call announces its name before any
                //    arguments arrive; a backend re-announcing the same
                //    call's name mid-stream has not.
                let is_new_call = match current_tool_call.as_ref() {
                    None => true,
                    Some(cur) => match (*current_tool_index, index) {
                        (Some(ci), Some(ni)) => ci != ni,
                        _ => !cur.function.arguments.is_empty(),
                    },
                };

                if is_new_call {
                    if let Some(prev) = current_tool_call.take() {
                        tool_calls.push(prev);
                    }
                    let id = format!(
                        "toolu_{}",
                        Uuid::new_v4().to_string().split('-').next().unwrap_or("0")
                    );
                    *current_tool_call = Some(ToolCall {
                        id: id.clone(),
                        call_type: "function".into(),
                        function: ToolCallFunction {
                            name: name.to_string(),
                            arguments: String::new(),
                        },
                    });
                    *current_tool_index = index;
                    events.push(ToolStreamEvent::Started { id, name: name.to_string() });
                }
                // Not new -> a duplicate announcement for the call already
                // open. Don't reset state or re-emit Started, or a
                // repeat-happy backend would fracture one tool call's
                // arguments across several bogus content blocks.
            }
            if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                if !args.is_empty() {
                    if let Some(cur) = current_tool_call.as_mut() {
                        cur.function.arguments.push_str(args);
                    }
                    events.push(ToolStreamEvent::ArgsDelta(args.to_string()));
                }
            }
        }
    }

    (new_text, events)
}

pub fn parse_tool_calls_from_text(text: &str) -> Option<Vec<ToolCall>> {
    if let Ok(val) = serde_json::from_str::<Value>(text) {
        if let Some(calls) = val.get("tool_calls") {
            return serde_json::from_value::<Vec<ToolCall>>(calls.clone()).ok();
        }
    }
    if text.contains("<tool_call>") {
        if let Some(start) = text.find("<tool_call>") {
            if let Some(end) = text.find("</tool_call>") {
                let json_str = &text[start + 11..end];
                if let Ok(val) = serde_json::from_str::<Value>(json_str) {
                    let name = val.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                    let args = val.get("arguments").or_else(|| val.get("parameters")).cloned().unwrap_or(serde_json::json!({}));
                    return Some(vec![ToolCall {
                        id: format!("toolu_{}", Uuid::new_v4().to_string().split('-').next().unwrap_or("0")),
                        call_type: "function".into(),
                        function: ToolCallFunction {
                            name: name.to_string(),
                            arguments: serde_json::to_string(&args).unwrap_or_default(),
                        },
                    }]);
                }
            }
        }
    }
    None
}