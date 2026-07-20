use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use bytes::BytesMut;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::TcpStream;
use serde::Serialize;
use serde_json::{json, Value};

use crate::core::config::Config;
use crate::db::memory::Database;
use crate::api::auth::{verify_jwt, lookup_api_key_full};

// ─────────────────────────────────────────────────────────────────────────────
// Logging helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Keep "we got a shape we didn't expect" log lines from dumping an entire
/// multi-KB payload into the log — enough context to recognize what the
/// worker actually sent, not the whole thing.
pub fn truncate_for_log(s: &str) -> String {
    const MAX_CHARS: usize = 300;
    if s.chars().count() <= MAX_CHARS {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(MAX_CHARS).collect();
        format!("{}… ({} bytes total)", truncated, s.len())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Auth helpers
// ─────────────────────────────────────────────────────────────────────────────

pub fn extract_token(auth: Option<&str>) -> Option<String> {
    auth.and_then(|a| {
        if a.starts_with("Bearer ") {
            Some(a[7..].trim().to_string())
        } else if !a.is_empty() {
            Some(a.trim().to_string())
        } else {
            None
        }
    })
}

#[allow(dead_code)]
pub fn validate_token(token: &str, db: &Arc<Database>, cfg: &Config) -> bool {
    // PSK shortcut (for Claude Code / simple setups)
    if cfg.allow_psk_as_token && token == cfg.psk {
        return true;
    }
    // DB API key
    if lookup_api_key_full(db, token) {
        return true;
    }
    // JWT (for management UI users)
    verify_jwt(token, &cfg.jwt_secret).is_some()
}

pub fn require_admin_auth(auth: &Option<String>, db: &Arc<Database>, cfg: &Config) -> bool {
    let token = match extract_token(auth.as_deref()) {
        Some(t) => t,
        None    => return false,
    };
    // PSK shortcut — always admin
    if cfg.allow_psk_as_token && token == cfg.psk {
        return true;
    }
    // JWT — trust the is_admin claim
    if let Some(claims) = verify_jwt(&token, &cfg.jwt_secret) {
        return claims.is_admin;
    }
    // API key — resolve to user_id, then check DB for is_admin flag
    if let Some(user_id) = db.lookup_api_key_with_user(&token) {
        return db.is_user_admin(user_id);
    }
    false
}

pub fn get_user_id_from_auth(auth: &Option<String>, db: &Arc<Database>, cfg: &Config) -> Option<i64> {
    let token = extract_token(auth.as_deref())?;
    // Try JWT first
    if let Some(claims) = verify_jwt(&token, &cfg.jwt_secret) {
        return claims.sub.parse::<i64>().ok();
    }
    // PSK has no specific user_id
    if cfg.allow_psk_as_token && token == cfg.psk {
        return None;
    }
    // Try API key → returns associated user_id
    db.lookup_api_key_with_user(&token)
}

// ─────────────────────────────────────────────────────────────────────────────
// Body / time helpers
// ─────────────────────────────────────────────────────────────────────────────

pub async fn read_body(stream: &mut TcpStream, mut existing: BytesMut, content_length: usize) -> Result<Vec<u8>, ()> {
    let mut buf = vec![0u8; 4096];
    while existing.len() < content_length {
        let (res, chunk) = stream.read(buf).await;
        let n = match res {
            Ok(n) if n > 0 => n,
            _ => return Err(()),
        };
        existing.extend_from_slice(&chunk[..n]);
        buf = chunk; // Reuse buffer for the next iteration
    }
    Ok(existing[..content_length].to_vec())
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP response helpers
// ─────────────────────────────────────────────────────────────────────────────

pub fn build_error_response(status: u16, code: &str, message: &str) -> Value {
    json!({
        "error": {
            "type": code,
            "message": message,
            "code": status
        }
    })
}

pub async fn send_json_200<T: Serialize>(stream: &mut TcpStream, body: &T) {
    let json = serde_json::to_vec(body).unwrap_or_default();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n",
        json.len()
    );
    let _ = stream.write_all(header.into_bytes()).await;
    let _ = stream.write_all(json).await;
}

pub async fn send_raw_json(stream: &mut TcpStream, status: u16, body: &Value) {
    let json = serde_json::to_vec(body).unwrap_or_default();
    let reason = match status {
        400 => "Bad Request", 401 => "Unauthorized", 403 => "Forbidden",
        404 => "Not Found",   503 => "Service Unavailable", 504 => "Gateway Timeout",
        _   => "Internal Server Error",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n",
        status, reason, json.len()
    );
    let _ = stream.write_all(header.into_bytes()).await;
    let _ = stream.write_all(json).await;
}

pub async fn send_json_error(stream: &mut TcpStream, status: u16, code: &str, message: &str) {
    let body = build_error_response(status, code, message);
    send_raw_json(stream, status, &body).await;
}