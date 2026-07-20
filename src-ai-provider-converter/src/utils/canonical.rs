use crate::request::anthropic::{AnthropicRequest, AnthropicTool};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt::Write;

pub fn canonicalize_anthropic_request(request: &mut AnthropicRequest) {
    if let Some(tools) = request.tools.take() {
        let tools = canonicalize_anthropic_tools(tools);
        request.tools = (!tools.is_empty()).then_some(tools);
    }
}

pub fn canonicalize_anthropic_tools(tools: Vec<AnthropicTool>) -> Vec<AnthropicTool> {
    let mut indexed = tools
        .into_iter()
        .enumerate()
        .map(|(index, mut tool)| {
            if let Some(schema) = tool.input_schema.take() {
                tool.input_schema = Some(canonicalize_json_schema(schema));
            }
            tool.extra = canonicalize_json_map(tool.extra);
            let key = tool_sort_key(&tool, index);
            (key, tool)
        })
        .collect::<Vec<_>>();

    indexed.sort_by(|(left, _), (right, _)| left.cmp(right));

    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(indexed.len());
    for (_, tool) in indexed {
        let identity = canonical_tool_identity(&tool);
        if seen.insert(identity) {
            out.push(tool);
        }
    }
    out
}

pub fn canonicalize_json_schema(value: Value) -> Value {
    canonicalize_json_value(value, None)
}

pub fn full_session_hash(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

pub fn is_hosted_web_search_tool(tool: &AnthropicTool) -> bool {
    tool.extra
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.starts_with("web_search_"))
        || tool.name == "web_search"
        || tool.name.starts_with("web_search_")
}

fn tool_sort_key(tool: &AnthropicTool, index: usize) -> (u8, String, String, usize) {
    let kind = if is_hosted_web_search_tool(tool) {
        0
    } else {
        1
    };
    let name = normalized_tool_name(tool);
    let body = canonical_tool_identity(tool);
    (kind, name, body, index)
}

fn normalized_tool_name(tool: &AnthropicTool) -> String {
    if is_hosted_web_search_tool(tool) {
        "web_search".into()
    } else {
        tool.name.clone()
    }
}

fn canonical_tool_identity(tool: &AnthropicTool) -> String {
    serde_json::to_string(tool).unwrap_or_else(|_| format!("{tool:?}"))
}

fn canonicalize_json_map(map: Map<String, Value>) -> Map<String, Value> {
    let value = canonicalize_json_value(Value::Object(map), None);
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

fn canonicalize_json_value(value: Value, parent_key: Option<&str>) -> Value {
    match value {
        Value::Array(items) => {
            let mut items = items
                .into_iter()
                .map(|item| canonicalize_json_value(item, None))
                .collect::<Vec<_>>();
            if parent_key == Some("required") {
                items.sort_by(|left, right| stable_value_key(left).cmp(&stable_value_key(right)));
            }
            Value::Array(items)
        }
        Value::Object(object) => {
            let mut out = Map::new();
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (key, value) in entries {
                out.insert(key.clone(), canonicalize_json_value(value, Some(&key)));
            }
            Value::Object(out)
        }
        scalar => scalar,
    }
}

fn stable_value_key(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str, schema: Value) -> AnthropicTool {
        AnthropicTool {
            name: name.into(),
            description: Some(format!("{name} tool")),
            input_schema: Some(schema),
            extra: Map::new(),
        }
    }

    #[test]
    fn canonicalizes_tool_order_and_required_schema_order() {
        let tools = canonicalize_anthropic_tools(vec![
            tool(
                "Write",
                json!({
                    "required": ["content", "path"],
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "type": "object"
                }),
            ),
            tool(
                "Read",
                json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
            ),
        ]);

        assert_eq!(tools[0].name, "Read");
        assert_eq!(tools[1].name, "Write");
        assert_eq!(
            tools[1].input_schema.as_ref().unwrap()["required"],
            json!(["content", "path"])
        );
    }

    #[test]
    fn preserves_enum_order_while_sorting_required() {
        let schema = canonicalize_json_schema(json!({
            "type": "object",
            "properties": {
                "mode": {"enum": ["z", "a", "m"]}
            },
            "required": ["mode", "path"]
        }));

        assert_eq!(schema["properties"]["mode"]["enum"], json!(["z", "a", "m"]));
        assert_eq!(schema["required"], json!(["mode", "path"]));
    }

    #[test]
    fn removes_only_exact_duplicate_tools() {
        let schema = json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        });
        let tools = canonicalize_anthropic_tools(vec![
            tool("Read", schema.clone()),
            tool("Read", schema.clone()),
            tool("Read", json!({"type": "object"})),
        ]);

        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "Read");
        assert_eq!(tools[1].name, "Read");
        assert_ne!(tools[0].input_schema, tools[1].input_schema);
    }

    #[test]
    fn hashes_sessions_without_exposing_raw_ids() {
        let raw = "0e377980-02ec-471a-b760-ce1b2f6658a7";
        let hash = full_session_hash(raw);

        assert_eq!(hash.len(), 64);
        assert!(!hash.contains(raw));
    }
}
