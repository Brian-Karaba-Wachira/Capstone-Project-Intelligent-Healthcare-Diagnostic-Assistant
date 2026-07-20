use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConverterError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
}

pub type Result<T> = std::result::Result<T, ConverterError>;
