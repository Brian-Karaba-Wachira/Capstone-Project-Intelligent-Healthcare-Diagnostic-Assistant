use serde_json::{json, Value};
use uuid::Uuid;

use crate::core::protocol::{ToolCall, ToolCallFunction};

// ── Legacy function-calling → tools normalization ───────────────────────────
//
// Operates at the raw JSON level now (on `request.extra`, where
// ai_provider_converter::ChatCompletionsRequest parks unknown/legacy fields
// like `functions`/`function_call`) rather than on brain's typed protocol —
// so the output plugs directly into `request.tools`/`request.tool_choice`
// (both `Option<Vec<Value>>`/`Option<Value>`) before the shared
// `openai_tool_json_to_protocol_tool`/`openai_tool_choice_json_to_protocol`
// adapters in core::protocol run, identically to the Anthropic path.

pub fn legacy_functions_to_tools(functions: &Value) -> Option<Vec<Value>> {
    let functions = functions.as_array()?;
    let tools: Vec<Value> = functions.iter().map(|f| {
        json!({
            "type": "function",
            "function": {
                "name": f.get("name").and_then(Value::as_str).unwrap_or("tool"),
                "description": f.get("description"),
                "parameters": f.get("parameters"),
            }
        })
    }).collect();
    (!tools.is_empty()).then_some(tools)
}

pub fn legacy_function_call_to_tool_choice(function_call: &Value) -> Value {
    match function_call.as_str() {
        Some("none") => json!("none"),
        Some("auto") => json!("auto"),
        _ => {
            if let Some(name) = function_call.get("name").and_then(Value::as_str) {
                json!({ "type": "function", "function": { "name": name } })
            } else {
                json!("auto")
            }
        }
    }
}

// ── Tool call detection — parse assembled response text ─────────────────────
// Unchanged: this parses worker OUTPUT, not client input, so it's unrelated
// to the inbound-shape fragility this refactor is fixing.

pub fn parse_tool_calls_from_response(response: &str) -> (Option<String>, Option<Vec<ToolCall>>) {
    if let Ok(val) = serde_json::from_str::<Value>(response) {
        if let Some(calls) = val.get("tool_calls") {
            if let Ok(tool_calls) = serde_json::from_value::<Vec<ToolCall>>(calls.clone()) {
                if !tool_calls.is_empty() {
                    return (None, Some(tool_calls));
                }
            }
        }
    }

    if response.contains("<tool_call>") {
        if let Some(start) = response.find("<tool_call>") {
            if let Some(end) = response.find("</tool_call>") {
                let json_str = &response[start + 11..end];
                if let Ok(val) = serde_json::from_str::<Value>(json_str) {
                    let name = val.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                    let args = val.get("arguments")
                        .or_else(|| val.get("parameters"))
                        .cloned()
                        .unwrap_or(json!({}));
                    let call = ToolCall {
                        id:        format!("call_{}", Uuid::new_v4().to_string().replace('-', "")[..8].to_string()),
                        call_type: "function".into(),
                        function:  ToolCallFunction {
                            name:      name.to_string(),
                            arguments: serde_json::to_string(&args).unwrap_or_default(),
                        },
                    };
                    return (None, Some(vec![call]));
                }
            }
        }
    }

    (Some(response.to_string()), None)
}

/// Extract the `content` field from a SSE delta line.
pub fn extract_delta_content(sse_line: &str) -> String {
    let json_str = sse_line
        .strip_prefix("data: ")
        .unwrap_or(sse_line)
        .trim();

    if json_str == "[DONE]" || json_str.is_empty() {
        return String::new();
    }

    serde_json::from_str::<Value>(json_str)
        .ok()
        .and_then(|v| {
            v.get("choices")?
                .get(0)?
                .get("delta")?
                .get("content")?
                .as_str()
                .map(|s| s.to_string())
        })
        .unwrap_or_default()
}
