use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "lmodel — llama.cpp ↔ Brain tunnel worker")]
pub struct Config {
    /// WebSocket URL of the Brain  (e.g. wss://node05.mikosi.fr.eu.org/worker)
    #[arg(long)]
    pub brain: String,

    /// Model name reported to the brain (e.g. deepseek-coder)
    #[arg(long)]
    pub model: String,

    /// Port where llama.cpp server is listening
    #[arg(long, default_value_t = 18080)]
    pub llamacpp_port: u16,

    /// GPU label reported in Register (informational)
    #[arg(long, default_value = "T4")]
    pub gpu: String,

    /// Free VRAM in MiB to report at registration time
    #[arg(long, default_value_t = 10240)]
    pub vram_free_mb: u32,

    /// Maximum context length to report
    #[arg(long, default_value_t = 16384)]
    pub max_context: u32,

    /// Unique ID for this worker session (auto-generated, do not set manually)
    #[arg(skip)]
    pub worker_id: String,
}

impl Config {
    pub fn from_args() -> Self {
        let mut cfg = <Config as Parser>::parse();
        cfg.worker_id = uuid::Uuid::new_v4().to_string();
        cfg
    }
}
