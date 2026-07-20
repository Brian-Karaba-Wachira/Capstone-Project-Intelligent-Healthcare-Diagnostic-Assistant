pub mod error;
pub mod request;
pub mod response;
pub mod stream;
pub mod utils;

pub use error::{ConverterError, Result};
pub use utils::translate::{
    translate_anthropic_to_openai_request,
    translate_anthropic_to_openai_request_ex,
    translate_openai_to_anthropic_request,
    translate_anthropic_to_openai_response,
    translate_openai_to_anthropic_response,
    TranslatedChatRequest,
};
pub use utils::finish_reason::{
    openai_finish_reason_to_anthropic_stop_reason,
    anthropic_stop_reason_to_openai_finish_reason,
};
pub use stream::openai::{translate_chat_stream, accumulate_chat_response};
pub use stream::anthropic::parser::translate_anthropic_to_openai_stream;

// Re-export Anthropic Request & Response Types
pub use request::anthropic::{AnthropicRequest, AnthropicTool, AnthropicMessage};
pub use response::anthropic::{AnthropicResponse, AnthropicUsage, AnthropicContentBlock};

// Re-export OpenAI Request & Response Types
pub use request::openai::ChatCompletionsRequest;
pub use response::openai::{ChatCompletionsResponse, ChatCompletionsChoice, ChatCompletionsMessage};

// Re-export Anthropic SSE builders for smooth stream operation
pub use stream::anthropic::{
    message_start,
    message_delta,
    message_stop,
    content_block_start,
    content_block_delta,
    content_block_stop,
    ping as claude_ping,
    error as claude_error,
    response_json,
};

// Re-export OpenAI SSE builders
pub use stream::openai::builders::{
    chat_completion_chunk,
    done as openai_done,
};
