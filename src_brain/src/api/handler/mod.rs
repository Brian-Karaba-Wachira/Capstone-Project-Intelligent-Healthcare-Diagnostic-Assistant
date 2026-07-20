mod common;
pub mod chat;
mod management;
pub mod anthropic;

use std::sync::Arc;
use bytes::BytesMut;
use monoio::net::TcpStream;

use crate::core::config::Config;
use crate::core::idempotency::IdempotencyStore;
use crate::db::memory::Database;
use crate::worker::registry::WorkerRegistry;
use crate::api::router::Router;
use crate::metrics::Metrics;

use chat::{ModelObject, ModelsResponse};
use common::{send_json_200, send_json_error, extract_token, now_secs};

pub use management::handle_management_api;

// ─────────────────────────────────────────────────────────────────────────────
// Main API handler — /v1/*
// Supports both OpenAI and Anthropic format detection via `is_anthropic` flag.
// ─────────────────────────────────────────────────────────────────────────────

pub async fn handle_request(
    mut stream:       TcpStream,
    method:           String,
    path:             String,
    auth:             Option<String>,
    session_id:       Option<String>,
    idempotency_key:  Option<String>,
    content_length:   usize,
    body_buf:         BytesMut,
    registry:         Arc<WorkerRegistry>,
    router:           Arc<Router>,
    db:               Arc<Database>,
    cfg:              Arc<Config>,
    metrics:          Arc<Metrics>,
    idempotency:      Arc<IdempotencyStore>,
    
    
    is_anthropic:     bool,
) {
    // ── Anthropic API: different auth (x-api-key) and endpoints ───────────
    if is_anthropic {
        // Anthropic uses x-api-key header for auth, but tools like claude-code use Authorization Bearer
        let auth_token = extract_token(auth.as_deref()).unwrap_or_default();
        // Validate the x-api-key against DB API keys; resolve to user_id
        let anthropic_user_id = if cfg.allow_psk_as_token && auth_token == cfg.psk {
            None
        } else if let Some(id) = db.lookup_api_key_with_user(&auth_token) {
            Some(id)
        } else if let Some(claims) = crate::api::auth::verify_jwt(&auth_token, &cfg.jwt_secret) {
            claims.sub.parse::<i64>().ok()
        } else {
            // Anthropic format error response
            let err = serde_json::json!({
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid x-api-key"}
            });
            let body = serde_json::to_vec(&err).unwrap_or_default();
            let resp = format!(
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(), String::from_utf8_lossy(&body)
            );
            use monoio::io::AsyncWriteRentExt;
            let _ = stream.write_all(resp.into_bytes()).await;
            return;
        };

        // FIX: session_id used to be silently dropped here — every Anthropic
        // request (Claude Code, codewhip, etc.) collapsed onto session_id=""
        // inside handle_anthropic. Now threaded through like the OpenAI path.
        anthropic::handle_anthropic(
            stream, path, body_buf, content_length,
            session_id,
            registry, router, db, cfg,
            metrics, idempotency,
            idempotency_key,
            anthropic_user_id,
        ).await;
        return;
    }

    // ── Step 1: Auth (OpenAI format) ────────────────────────────────────────
    let token = match extract_token(auth.as_deref()) {
        Some(t) => t,
        None    => {
            send_json_error(&mut stream, 401, "missing_auth", "Authorization header required").await;
            return;
        }
    };

    let user_id = if cfg.allow_psk_as_token && token == cfg.psk {
        None
    } else if let Some(id) = db.lookup_api_key_with_user(&token) {
        Some(id)
    } else {
        if let Some(claims) = crate::api::auth::verify_jwt(&token, &cfg.jwt_secret) {
            claims.sub.parse::<i64>().ok()
        } else {
            send_json_error(&mut stream, 401, "invalid_auth", "Invalid or expired token").await;
            return;
        }
    };

    // Check idempotency: if key is cached, replay it immediately (§11.1)
    if let Some(ref idem_key) = idempotency_key {
        if let Some(cached) = idempotency.get(idem_key) {
            let reason = match cached.status {
                200 => "OK", 400 => "Bad Request", 401 => "Unauthorized",
                403 => "Forbidden", 404 => "Not Found", 503 => "Service Unavailable",
                _   => "Internal Server Error",
            };
            let resp = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n",
                cached.status, reason, cached.content_type, cached.body.len()
            );
            use monoio::io::AsyncWriteRentExt;
            let _ = stream.write_all(resp.into_bytes()).await;
            let _ = stream.write_all(cached.body).await;
            return;
        }
    }

    // ── Route ─────────────────────────────────────────────────────────────────
    let clean_path = path.split('?').next().unwrap_or(&path).to_string();

    match (method.as_str(), clean_path.as_str()) {

        // ── GET /v1/models (OpenAI format) ──────────────────────────────────
        ("GET", "/v1/models") => {
            let now = now_secs();
            let models: Vec<ModelObject> = registry.active_models()
                .into_iter()
                .map(|id| ModelObject { id, object: "model", created: now, owned_by: "brain" })
                .collect();

            let mut alias_models: Vec<ModelObject> = cfg.model_aliases.keys()
                .map(|alias| ModelObject {
                    id:       alias.clone(),
                    object:   "model",
                    created:  now,
                    owned_by: "brain",
                })
                .collect();

            let mut all = models;
            all.append(&mut alias_models);
            all.sort_by(|a, b| a.id.cmp(&b.id));
            all.dedup_by(|a, b| a.id == b.id);

            let resp = ModelsResponse { object: "list", data: all };
            send_json_200(&mut stream, &resp).await;
        }

        // ── POST /v1/chat/completions ────────────────────────────────────────
        ("POST", p) if p.contains("chat/completions") => {
            chat::handle_chat(
                stream, body_buf, content_length,
                session_id, user_id, idempotency_key,
                registry, router, db, cfg,
                metrics, idempotency,
            ).await;
        }

        _ => {
            send_json_error(&mut stream, 404, "not_found",
                            &format!("Unknown endpoint: {} {}", method, clean_path)).await;
        }
    }
}
