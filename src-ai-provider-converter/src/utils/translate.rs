use crate::error::{ConverterError, Result};
use crate::request::anthropic::{AnthropicRequest, AnthropicTool};
use crate::request::openai::ChatCompletionsRequest;
use crate::response::anthropic::AnthropicResponse;
use crate::response::openai::ChatCompletionsResponse;
use crate::utils::canonical::{canonicalize_anthropic_tools, is_hosted_web_search_tool};
use serde_json::{json, Value};

/// Result of translating an Anthropic request to the OpenAI-compatible shape,
/// plus any *hosted* tools (e.g. Anthropic's `web_search_*` server tool) that
/// were requested but stripped out of `request.tools` because an
/// OpenAI-shaped backend (e.g. a llama.cpp worker) has no way to execute
/// them as function tools.
///
/// This struct is purely informational — the converter does not execute,
/// simulate, or otherwise act on hosted tools. It only tells the caller
/// "these were asked for and removed," so a gateway sitting above this
/// library can decide whether/how to fulfill them (e.g. run a real web
/// search and inject the result back into the conversation) without the
/// converter itself knowing anything about HTTP, auth, or egress policy.
#[derive(Debug, Clone, Default)]
pub struct TranslatedChatRequest {
    pub request: ChatCompletionsRequest,
    pub hosted_tools: Vec<AnthropicTool>,
}

pub fn translate_anthropic_to_openai_request(
    request: &AnthropicRequest,
    upstream_model: &str,
) -> Result<ChatCompletionsRequest> {
    Ok(translate_anthropic_to_openai_request_ex(request, upstream_model)?.request)
}

/// Same translation as [`translate_anthropic_to_openai_request`], but also
/// surfaces any hosted/server-side tools that were stripped from the
/// outgoing tool list instead of silently discarding them.
pub fn translate_anthropic_to_openai_request_ex(
    request: &AnthropicRequest,
    upstream_model: &str,
) -> Result<TranslatedChatRequest> {
    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        let content = system_to_text(system);
        if !content.is_empty() {
            messages.push(json!({ "role": "system", "content": content }));
        }
    }
    for message in &request.messages {
        match message.role.as_str() {
            "user" => push_user_chat_messages(&mut messages, &message.content),
            "assistant" => push_assistant_chat_message(&mut messages, &message.content),
            "system" => {
                let content = content_to_text(&message.content);
                if !content.is_empty() {
                    messages.push(json!({ "role": "system", "content": content }));
                }
            }
            other => {
                return Err(ConverterError::InvalidRequest(format!(
                    "unsupported message role \"{other}\""
                )));
            }
        }
    }

    let mut hosted_tools = Vec::new();
    let tools = request
        .tools
        .as_ref()
        .map(|tools| translate_chat_tools(tools, &mut hosted_tools))
        .transpose()?
        .filter(|tools| !tools.is_empty());

    let tool_choice = if tools.is_some() {
        request.tool_choice.as_ref().map(translate_chat_tool_choice)
    } else {
        None
    };

    // Anthropic's `stop_sequences` has no dedicated field on AnthropicRequest
    // (it lands in `extra` via the flatten), but OpenAI's `stop` is its exact
    // equivalent — carry it across instead of dropping it.
    let stop = request.extra.get("stop_sequences").and_then(|value| match value {
        Value::Array(items) if !items.is_empty() => Some(value.clone()),
        Value::String(_) => Some(json!([value])),
        _ => None,
    });

    Ok(TranslatedChatRequest {
        request: ChatCompletionsRequest {
            model: upstream_model.to_string(),
            messages,
            stream: request.wants_stream(),
            tools,
            tool_choice,
            parallel_tool_calls: None,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            response_format: chat_response_format(request),
            stop,
            extra: Default::default(),
        },
        hosted_tools,
    })
}

fn push_user_chat_messages(messages: &mut Vec<Value>, content: &Value) {
    let mut parts = Vec::new();
    for block in normalized_blocks(content) {
        match block.get("type").and_then(Value::as_str).unwrap_or("text") {
            "text" => parts.push(json!({
                "type": "text",
                "text": block.get("text").and_then(Value::as_str).unwrap_or_default(),
            })),
            "image" => {
                if let Some(image) = block_to_chat_image(&block) {
                    parts.push(image);
                }
            }
            "tool_result" => {
                flush_user_parts(messages, &mut parts);
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id").and_then(Value::as_str).unwrap_or("tool_call"),
                    "content": tool_result_output(&block),
                }));
            }
            // Unknown block types (documents, future additions) must not leak
            // raw JSON into the prompt — keep any human-readable text they
            // carry and drop the rest.
            _ => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        parts.push(json!({ "type": "text", "text": text }));
                    }
                }
            }
        }
    }
    flush_user_parts(messages, &mut parts);
}

fn push_assistant_chat_message(messages: &mut Vec<Value>, content: &Value) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in normalized_blocks(content) {
        match block.get("type").and_then(Value::as_str).unwrap_or("text") {
            "text" => {
                if let Some(part) = block.get("text").and_then(Value::as_str) {
                    text.push_str(part);
                }
            }
            "tool_use" => tool_calls.push(json!({
                "id": block.get("id").and_then(Value::as_str).unwrap_or("tool_call"),
                "type": "function",
                "function": {
                    "name": block.get("name").and_then(Value::as_str).unwrap_or("tool"),
                    "arguments": block.get("input").filter(|input| input.is_object()).cloned().unwrap_or_else(|| json!({})).to_string(),
                }
            })),
            // Claude Code replays its own `thinking`/`redacted_thinking`
            // blocks on the next turn; an OpenAI-shaped backend has no
            // concept of them and they must never be re-fed as prompt text
            // (previously they were dumped into the prompt as raw JSON).
            "thinking" | "redacted_thinking" => {}
            _ => {
                if let Some(part) = block.get("text").and_then(Value::as_str) {
                    text.push_str(part);
                }
            }
        }
    }
    let mut message = serde_json::Map::new();
    message.insert("role".into(), json!("assistant"));
    if !text.is_empty() {
        message.insert("content".into(), Value::String(text));
    } else if tool_calls.is_empty() {
        // `content: null` with no tool_calls is rejected by most
        // OpenAI-compatible backends — an empty string is always accepted.
        message.insert("content".into(), Value::String(String::new()));
    } else {
        message.insert("content".into(), Value::Null);
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    messages.push(Value::Object(message));
}

fn flush_user_parts(messages: &mut Vec<Value>, parts: &mut Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    let content =
        if parts.len() == 1 && parts[0].get("type").and_then(Value::as_str) == Some("text") {
            parts[0]
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
                .into()
        } else {
            Value::Array(std::mem::take(parts))
        };
    messages.push(json!({ "role": "user", "content": content }));
    parts.clear();
}

fn normalized_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(text) => vec![json!({ "type": "text", "text": text })],
        Value::Array(items) => items.clone(),
        other => vec![json!({ "type": "text", "text": other.to_string() })],
    }
}

fn block_to_chat_image(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            let data = source.get("data").and_then(Value::as_str)?;
            Some(json!({
                "type": "image_url",
                "image_url": { "url": format!("data:{media};base64,{data}") }
            }))
        }
        Some("url") => source.get("url").and_then(Value::as_str).map(|url| {
            json!({
                "type": "image_url",
                "image_url": { "url": url }
            })
        }),
        _ => None,
    }
}

fn tool_result_output(block: &Value) -> String {
    let content = block.get("content").unwrap_or(&Value::Null);
    match content {
        // Tool results can carry typed blocks (text, image, ...). Extract the
        // text; represent images as a placeholder instead of leaking their
        // raw base64 JSON into the tool message.
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                if let Some(text) = item.as_str() {
                    return Some(text.to_string());
                }
                match item.get("type").and_then(Value::as_str) {
                    Some("image") => Some("[image omitted]".to_string()),
                    _ => item.get("text").and_then(Value::as_str).map(ToOwned::to_owned),
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => content_to_text(other),
    }
}

fn content_to_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(ToOwned::to_owned)
                    .or_else(|| {
                        item.get("text")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                    })
                    .unwrap_or_else(|| item.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn system_to_text(system: &Value) -> String {
    content_to_text(system)
}

fn translate_chat_tools(
    tools: &[AnthropicTool],
    hosted_tools: &mut Vec<AnthropicTool>,
) -> Result<Vec<Value>> {
    let tools = canonicalize_anthropic_tools(tools.to_vec());
    let mut out = Vec::with_capacity(tools.len());
    for tool in &tools {
        if is_hosted_web_search_tool(tool) {
            hosted_tools.push(tool.clone());
            continue;
        }
        let mut function = serde_json::Map::new();
        function.insert("name".into(), json!(tool.name.clone()));
        if let Some(desc) = &tool.description {
            function.insert("description".into(), json!(desc));
        }
        if let Some(schema) = &tool.input_schema {
            function.insert("parameters".into(), schema.clone());
        }
        out.push(json!({
            "type": "function",
            "function": function
        }));
    }
    Ok(out)
}

fn translate_chat_tool_choice(choice: &Value) -> Value {
    match choice.get("type").and_then(Value::as_str) {
        Some("auto") => json!("auto"),
        Some("any") => json!("required"),
        Some("tool") => {
            let name = choice.get("name").and_then(Value::as_str).unwrap_or("tool");
            json!({
                "type": "function",
                "function": { "name": name }
            })
        }
        _ => json!("auto"),
    }
}

fn chat_response_format(request: &AnthropicRequest) -> Option<Value> {
    let output_config = request.output_config.as_ref()?;
    let format = output_config.get("format")?;
    match format.get("type").and_then(Value::as_str) {
        Some("json") => Some(json!({ "type": "json_object" })),
        _ => None,
    }
}

/// Append blocks to the previous message when the roles match, otherwise
/// start a new message. Anthropic's Messages API expects alternating
/// user/assistant turns, and OpenAI conversations routinely contain runs of
/// same-role messages (several `tool` results in a row after a parallel tool
/// call, split user turns) that must fold into a single Anthropic turn.
fn push_anthropic_blocks(
    messages: &mut Vec<crate::request::anthropic::AnthropicMessage>,
    role: &str,
    mut blocks: Vec<Value>,
) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut() {
        if last.role == role {
            if let Value::Array(existing) = &mut last.content {
                existing.append(&mut blocks);
                return;
            }
        }
    }
    messages.push(crate::request::anthropic::AnthropicMessage {
        role: role.to_string(),
        content: Value::Array(blocks),
        extra: Default::default(),
    });
}

pub fn translate_openai_to_anthropic_request(
    request: &ChatCompletionsRequest,
    upstream_model: &str,
) -> Result<AnthropicRequest> {
    let mut anthropic_messages = Vec::new();
    let mut system_text = String::new();

    for msg in &request.messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = msg.get("content").unwrap_or(&Value::Null);

        match role {
            "system" | "developer" => {
                system_text.push_str(&content_to_text(content));
                system_text.push('\n');
            }
            // OpenAI carries tool results as their own role; Anthropic
            // expects a `tool_result` block inside a *user* message. Sending
            // role "tool" through unchanged (as this used to) is rejected by
            // the Anthropic API outright.
            "tool" => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id").and_then(Value::as_str).unwrap_or("tool_call"),
                    "content": content_to_text(content),
                });
                push_anthropic_blocks(&mut anthropic_messages, "user", vec![block]);
            }
            "assistant" => {
                let mut blocks = Vec::new();
                let text = content_to_text(content);
                if !text.is_empty() {
                    blocks.push(json!({ "type": "text", "text": text }));
                }
                if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
                    for call in tool_calls {
                        let id = call.get("id").and_then(Value::as_str).unwrap_or("call_xyz");
                        let function = call.get("function").unwrap_or(&Value::Null);
                        let name = function.get("name").and_then(Value::as_str).unwrap_or("tool");
                        let args_str = function.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": serde_json::from_str::<Value>(args_str).unwrap_or(json!({}))
                        }));
                    }
                }
                push_anthropic_blocks(&mut anthropic_messages, "assistant", blocks);
            }
            _ => {
                let mut blocks = Vec::new();
                if let Some(text) = content.as_str() {
                    if !text.is_empty() {
                        blocks.push(json!({ "type": "text", "text": text }));
                    }
                } else if let Some(arr) = content.as_array() {
                    for part in arr {
                        if let Some(typ) = part.get("type").and_then(Value::as_str) {
                            if typ == "image_url" {
                                if let Some(url) = part.pointer("/image_url/url").and_then(Value::as_str) {
                                    if url.starts_with("data:") {
                                        if let Some(comma_idx) = url.find(',') {
                                            let meta = &url[5..comma_idx];
                                            let data = &url[comma_idx + 1..];
                                            let media_type = meta.split(';').next().unwrap_or("image/jpeg");
                                            blocks.push(json!({
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": media_type,
                                                    "data": data
                                                }
                                            }));
                                        }
                                    } else {
                                        blocks.push(json!({
                                            "type": "image",
                                            "source": { "type": "url", "url": url }
                                        }));
                                    }
                                }
                            } else {
                                blocks.push(part.clone());
                            }
                        } else {
                            blocks.push(part.clone());
                        }
                    }
                }
                push_anthropic_blocks(&mut anthropic_messages, "user", blocks);
            }
        }
    }

    let system = if system_text.is_empty() {
        None
    } else {
        Some(Value::String(system_text.trim().to_string()))
    };

    let mut anthropic_tools = None;
    if let Some(tools) = &request.tools {
        let mut t_vec = Vec::new();
        for t in tools {
            if let Some(function) = t.get("function") {
                t_vec.push(AnthropicTool {
                    name: function.get("name").and_then(Value::as_str).unwrap_or("tool").to_string(),
                    description: function.get("description").and_then(Value::as_str).map(String::from),
                    input_schema: function.get("parameters").cloned(),
                    extra: Default::default(),
                });
            }
        }
        if !t_vec.is_empty() {
            anthropic_tools = Some(t_vec);
        }
    }

    let tool_choice = if let Some(tc) = &request.tool_choice {
        match tc.as_str() {
            Some("auto") => Some(json!({"type": "auto"})),
            Some("required") => Some(json!({"type": "any"})),
            _ => if let Some(function) = tc.get("function") {
                Some(json!({
                    "type": "tool",
                    "name": function.get("name").and_then(Value::as_str).unwrap_or("tool")
                }))
            } else {
                Some(json!({"type": "auto"}))
            }
        }
    } else {
        None
    };

    // OpenAI `stop` (string or array) → Anthropic `stop_sequences`. The
    // AnthropicRequest struct keeps it in `extra` (same place inbound
    // Anthropic requests park it via the serde flatten).
    let mut extra = serde_json::Map::new();
    if let Some(stop) = &request.stop {
        let sequences: Vec<Value> = match stop {
            Value::String(s) => vec![Value::String(s.clone())],
            Value::Array(items) => items
                .iter()
                .filter(|item| item.is_string())
                .cloned()
                .collect(),
            _ => Vec::new(),
        };
        if !sequences.is_empty() {
            extra.insert("stop_sequences".into(), Value::Array(sequences));
        }
    }

    Ok(AnthropicRequest {
        model: upstream_model.to_string(),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        stream: Some(request.stream),
        system,
        messages: anthropic_messages,
        tools: anthropic_tools,
        tool_choice,
        metadata: None,
        output_config: None,
        // OpenAI's ChatCompletionsRequest has no equivalent field to round-trip
        // extended-thinking config from, so this is intentionally always None
        // in this direction (not a dropped/forgotten value).
        thinking: None,
        extra,
    })
}

pub fn translate_openai_to_anthropic_response(
    response: &ChatCompletionsResponse,
) -> AnthropicResponse {
    let mut content = Vec::new();
    let mut stop_reason = None;

    if let Some(choice) = response.choices.first() {
        if let Some(text) = &choice.message.content {
            if !text.is_empty() {
                content.push(crate::response::anthropic::AnthropicContentBlock {
                    kind: "text".into(),
                    text: Some(text.clone()),
                    id: None,
                    name: None,
                    input: None,
                });
            }
        }
        if let Some(tool_calls) = &choice.message.tool_calls {
            for call in tool_calls {
                let function = call.get("function").unwrap_or(&Value::Null);
                // `input` is REQUIRED on an Anthropic tool_use block (and
                // `id`/`name` too). Empty or unparsable arguments used to
                // yield `input: None`, which serde then *omitted entirely*
                // from the serialized block — Anthropic clients (Claude Code
                // included) reject a tool_use block with no input. Default to
                // `{}` instead.
                let input = function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|args| serde_json::from_str::<Value>(args).ok())
                    .filter(Value::is_object)
                    .unwrap_or_else(|| json!({}));
                content.push(crate::response::anthropic::AnthropicContentBlock {
                    kind: "tool_use".into(),
                    text: None,
                    id: Some(
                        call.get("id")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4().simple())),
                    ),
                    name: Some(
                        function
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("tool")
                            .to_string(),
                    ),
                    input: Some(input),
                });
            }
        }
        stop_reason = choice.finish_reason.clone();
    }

    let usage = match &response.usage {
        Some(u) => crate::response::anthropic::AnthropicUsage {
            input_tokens: u.prompt_tokens as u64,
            output_tokens: u.completion_tokens as u64,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        },
        None => crate::response::anthropic::AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        },
    };

    let stop_reason_mapped = stop_reason
        .as_deref()
        .map(crate::utils::finish_reason::openai_finish_reason_to_anthropic_stop_reason);

    AnthropicResponse {
        id: response.id.clone(),
        kind: "message".into(),
        role: "assistant".into(),
        model: response.model.clone(),
        content,
        stop_reason: stop_reason_mapped,
        stop_sequence: None,
        usage,
    }
}

pub fn translate_anthropic_to_openai_response(
    response: &AnthropicResponse,
) -> ChatCompletionsResponse {
    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    for block in &response.content {
        match block.kind.as_str() {
            "text" => {
                if let Some(text) = &block.text {
                    text_content.push_str(text);
                }
            }
            "tool_use" => {
                let function = json!({
                    "name": block.name.as_deref().unwrap_or("tool"),
                    "arguments": block.input.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "{}".into())
                });
                tool_calls.push(json!({
                    "id": block.id.as_deref().unwrap_or("call_xyz"),
                    "type": "function",
                    "function": function
                }));
            }
            _ => {}
        }
    }

    let message = crate::response::openai::ChatCompletionsMessage {
        role: "assistant".into(),
        content: if text_content.is_empty() { None } else { Some(text_content) },
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
    };

    let stop_reason_mapped = response
        .stop_reason
        .as_deref()
        .map(crate::utils::finish_reason::anthropic_stop_reason_to_openai_finish_reason);

    let choice = crate::response::openai::ChatCompletionsChoice {
        index: 0,
        message,
        finish_reason: stop_reason_mapped,
    };

    let usage = crate::response::openai::ChatCompletionsUsage {
        prompt_tokens: response.usage.input_tokens as u32,
        completion_tokens: response.usage.output_tokens as u32,
        total_tokens: (response.usage.input_tokens + response.usage.output_tokens) as u32,
    };

    ChatCompletionsResponse {
        id: response.id.clone(),
        object: "chat.completion".into(),
        created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        model: response.model.clone(),
        choices: vec![choice],
        usage: Some(usage),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::anthropic::AnthropicMessage;
    use serde_json::json;

    fn anthropic_request(messages: Vec<AnthropicMessage>) -> AnthropicRequest {
        AnthropicRequest {
            model: "claude".into(),
            max_tokens: Some(1024),
            temperature: None,
            top_p: None,
            stream: Some(true),
            system: None,
            messages,
            tools: None,
            tool_choice: None,
            metadata: None,
            output_config: None,
            thinking: None,
            extra: Default::default(),
        }
    }

    fn message(role: &str, content: Value) -> AnthropicMessage {
        AnthropicMessage { role: role.into(), content, extra: Default::default() }
    }

    #[test]
    fn thinking_blocks_never_reach_the_openai_prompt() {
        let request = anthropic_request(vec![message(
            "assistant",
            json!([
                { "type": "thinking", "thinking": "secret chain of thought", "signature": "sig" },
                { "type": "text", "text": "visible answer" }
            ]),
        )]);
        let translated = translate_anthropic_to_openai_request(&request, "m").unwrap();
        let content = translated.messages[0]["content"].as_str().unwrap();
        assert_eq!(content, "visible answer");
    }

    #[test]
    fn assistant_with_no_text_and_no_tools_gets_empty_string_content() {
        let request = anthropic_request(vec![message("assistant", json!([]))]);
        let translated = translate_anthropic_to_openai_request(&request, "m").unwrap();
        assert_eq!(translated.messages[0]["content"], json!(""));
    }

    #[test]
    fn stop_sequences_map_to_openai_stop() {
        let mut request = anthropic_request(vec![message("user", json!("hi"))]);
        request.extra.insert("stop_sequences".into(), json!(["\n\nHuman:"]));
        let translated = translate_anthropic_to_openai_request(&request, "m").unwrap();
        assert_eq!(translated.stop, Some(json!(["\n\nHuman:"])));
    }

    #[test]
    fn tool_result_with_image_does_not_leak_base64_json() {
        let request = anthropic_request(vec![message(
            "user",
            json!([{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [
                    { "type": "text", "text": "screenshot taken" },
                    { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AAAA" } }
                ]
            }]),
        )]);
        let translated = translate_anthropic_to_openai_request(&request, "m").unwrap();
        let tool_msg = &translated.messages[0];
        assert_eq!(tool_msg["role"], "tool");
        let content = tool_msg["content"].as_str().unwrap();
        assert!(content.contains("screenshot taken"));
        assert!(!content.contains("AAAA"));
        assert!(!content.contains("base64"));
    }

    #[test]
    fn openai_tool_role_becomes_anthropic_tool_result_user_message() {
        let request = ChatCompletionsRequest {
            model: "m".into(),
            messages: vec![
                json!({ "role": "assistant", "content": null, "tool_calls": [
                    { "id": "call_1", "type": "function", "function": { "name": "read", "arguments": "{\"p\":1}" } }
                ]}),
                json!({ "role": "tool", "tool_call_id": "call_1", "content": "file contents" }),
                json!({ "role": "tool", "tool_call_id": "call_2", "content": "second result" }),
            ],
            ..Default::default()
        };
        let translated = translate_openai_to_anthropic_request(&request, "claude").unwrap();
        // assistant turn, then ONE merged user turn holding both tool_results
        assert_eq!(translated.messages.len(), 2);
        assert_eq!(translated.messages[1].role, "user");
        let blocks = translated.messages[1].content.as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "call_1");
        assert_eq!(blocks[1]["tool_use_id"], "call_2");
    }

    #[test]
    fn openai_response_tool_use_always_has_input() {
        let response = ChatCompletionsResponse {
            id: "chatcmpl-1".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "m".into(),
            choices: vec![crate::response::openai::ChatCompletionsChoice {
                index: 0,
                message: crate::response::openai::ChatCompletionsMessage {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: Some(vec![json!({
                        "id": "call_1", "type": "function",
                        "function": { "name": "ping", "arguments": "" }
                    })]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        };
        let anthropic = translate_openai_to_anthropic_response(&response);
        let block = &anthropic.content[0];
        assert_eq!(block.kind, "tool_use");
        assert_eq!(block.input, Some(json!({})));
        // and the serialized form must actually contain the field
        let serialized = serde_json::to_value(&anthropic).unwrap();
        assert_eq!(serialized["content"][0]["input"], json!({}));
        assert_eq!(anthropic.stop_reason.as_deref(), Some("tool_use"));
    }
}
