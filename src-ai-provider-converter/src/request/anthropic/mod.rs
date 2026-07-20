use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub output_config: Option<Value>,
    #[serde(default)]
    pub thinking: Option<Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl AnthropicRequest {
    pub fn wants_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Value,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}
