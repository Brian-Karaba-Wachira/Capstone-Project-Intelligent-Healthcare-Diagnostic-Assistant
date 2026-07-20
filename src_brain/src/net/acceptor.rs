use httparse::{Request, Status};
use std::sync::Arc;
use bytes::BytesMut;
use monoio::net::TcpStream;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};

use sha1::{Digest, Sha1};
use base64::Engine as _;

use crate::core::config::Config;
use crate::core::idempotency::IdempotencyStore;
use crate::worker::registry::WorkerRegistry;
use crate::api::router::Router;
use crate::db::memory::Database;
use crate::metrics::Metrics;

/// RFC 6455 §1.3 handshake: base64(SHA1(client_key + magic_guid)).
fn compute_ws_accept(client_key: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(client_key.as_bytes());
    hasher.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    base64::engine::general_purpose::STANDARD.encode(hasher.finalize())
}

pub async fn handle_conn(
    mut stream: TcpStream,
    registry: Arc<WorkerRegistry>,
    router: Arc<Router>,
    db: Arc<Database>,
    cfg: Arc<Config>,
    metrics: Arc<Metrics>,
    idempotency: Arc<IdempotencyStore>,
    
    
) {
    let mut buf = BytesMut::with_capacity(4096);

    loop {
        let (res, chunk) = stream.read(vec![0u8; 4096]).await;
        let n = match res {
            Ok(n) if n > 0 => n,
            _ => return,
        };
        buf.extend_from_slice(&chunk[..n]);

        let mut headers = [httparse::EMPTY_HEADER; 32];
        let mut req = Request::new(&mut headers);

        match req.parse(&buf) {
            Ok(Status::Complete(header_len)) => {
                let method = req.method.unwrap_or("GET").to_string();
                let mut path   = req.path.unwrap_or("/").to_string();

                log::info!("Incoming Raw Path: {} {}", method, path);

                // Handle CORS Preflight
                if method == "OPTIONS" {
                    let res = "HTTP/1.1 204 No Content\r\n\
                               Access-Control-Allow-Origin: *\r\n\
                               Access-Control-Allow-Methods: POST, GET, OPTIONS, DELETE, PUT\r\n\
                               Access-Control-Allow-Headers: Content-Type, Authorization, x-worker-token, x-session-id, Idempotency-Key, x-api-key, anthropic-version\r\n\
                               Access-Control-Max-Age: 86400\r\n\
                               \r\n";
                    let _ = stream.write_all(res.as_bytes()).await;
                    return;
                }

                // If path is an absolute URI, strip to just the path
                if path.starts_with("http://") || path.starts_with("https://") {
                    if let Some(idx) = path.find("://").map(|i| i + 3) {
                        if let Some(slash_idx) = path[idx..].find('/') {
                            path = path[idx + slash_idx..].to_string();
                        }
                    }
                }

                // ── Extract common headers ────────────────────────────────────
                let mut content_length = 0usize;
                let mut auth: Option<String> = None;
                let mut ws_key: Option<String> = None;
                let mut session_id: Option<String> = None;
                let mut idempotency_key: Option<String> = None;
                let mut is_anthropic = false;

                for h in req.headers.iter() {
                    match h.name.to_ascii_lowercase().as_str() {
                        "content-length" => {
                            if let Ok(s) = std::str::from_utf8(h.value) {
                                content_length = s.trim().parse().unwrap_or(0);
                            }
                        }
                        "authorization" => {
                            auth = std::str::from_utf8(h.value).ok().map(|s| s.trim().to_string());
                        }
                        "x-worker-token" => {
                            auth = std::str::from_utf8(h.value).ok().map(|s| s.trim().to_string());
                        }
                        "x-api-key" => {
                            // Anthropic-style auth: x-api-key header = the API key directly
                            auth = std::str::from_utf8(h.value).ok().map(|s| s.trim().to_string());
                            is_anthropic = true;
                        }
                        "anthropic-version" => {
                            is_anthropic = true;
                        }
                        "x-session-id" => {
                            session_id = std::str::from_utf8(h.value).ok().map(|s| s.trim().to_string());
                        }
                        "sec-websocket-key" => {
                            ws_key = std::str::from_utf8(h.value).ok().map(|s| s.trim().to_string());
                        }
                        "idempotency-key" => {
                            idempotency_key = std::str::from_utf8(h.value).ok().map(|s| s.trim().to_string());
                        }
                        _ => {}
                    }
                }

                // ── /metrics endpoint (§12) ──────────────────────────────────
                if cfg.metrics_enabled && method == "GET" && path == "/metrics" {
                    let prom_text = metrics.render_prometheus();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\n\r\n{}",
                        prom_text.len(), prom_text
                    );
                    let _ = stream.write_all(resp.into_bytes()).await;
                    return;
                }

                // ── Route: /worker  (WebSocket upgrade for lmodel) ────────────
                if path.starts_with("/worker") {
                    let token_from_query = path.find('?').and_then(|qi| {
                        let query = &path[qi + 1..];
                        query.split('&').find_map(|pair| {
                            pair.strip_prefix("token=").map(|t| t.to_string())
                        })
                    });

                    let effective_auth = token_from_query.or(auth);
                    let authed = match &effective_auth {
                        Some(t) if t == &cfg.psk || t == &format!("Bearer {}", cfg.psk) => true,
                        _ => false,
                    };

                    if !authed {
                        log::warn!("Unauthorized WebSocket attempt on /worker");
                        let _ = stream.write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n").await;
                        return;
                    }

                    match ws_key {
                        Some(key) => {
                            let accept = compute_ws_accept(&key);
                            let upgrade = format!(
                                "HTTP/1.1 101 Switching Protocols\r\n\
                                 Upgrade: websocket\r\n\
                                 Connection: Upgrade\r\n\
                                 Sec-WebSocket-Accept: {}\r\n\
                                 \r\n",
                                accept
                            );
                            if stream.write_all(upgrade.into_bytes()).await.0.is_err() {
                                return;
                            }
                            let leftover = buf.split_off(header_len).freeze();
                            crate::worker::tunnel::handle_worker(
                                stream, leftover, registry, router, db, cfg,
                                metrics,
                            ).await;
                        }
                        None => {
                            log::warn!("WebSocket upgrade missing Sec-WebSocket-Key");
                            let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n").await;
                        }
                    }
                    return;
                }

                // ── Idempotency check (§11.1) ────────────────────────────────
                if let Some(ref idem_key) = idempotency_key {
                    if let Some((status, body, content_type)) =
                        db.get_idempotent_response(idem_key, cfg.idempotency_ttl_s as i64)
                    {
                        let reason = match status {
                            200 => "OK",
                            400 => "Bad Request",
                            401 => "Unauthorized",
                            403 => "Forbidden",
                            404 => "Not Found",
                            429 => "Too Many Requests",
                            503 => "Service Unavailable",
                            _   => "Internal Server Error",
                        };
                        let resp = format!(
                            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n",
                            status, reason, content_type, body.len()
                        );
                        let _ = stream.write_all(resp.into_bytes()).await;
                        let _ = stream.write_all(body).await;
                        return;
                    }
                }

                // ── Admin UI (Serve embedded static files) ──────────────────
                if method == "GET" && (path == "/v1/cc" || path == "/brain/ui") {
                    let res = format!(
                        "HTTP/1.1 301 Moved Permanently\r\nLocation: {}/\r\nContent-Length: 0\r\n\r\n",
                        path
                    );
                    let _ = stream.write_all(res.into_bytes()).await;
                    return;
                }

                // Support both /v1/cc (legacy) and /brain/ui (new admin dashboard)
                let is_ui_path = method == "GET" &&
                    (path.starts_with("/v1/cc") || path.starts_with("/brain/ui"));
                if is_ui_path {
                    let asset_path = if path.starts_with("/brain/ui") {
                        path.strip_prefix("/brain/ui").unwrap_or(&path).to_string()
                    } else {
                        path.strip_prefix("/v1/cc").unwrap_or(&path).to_string()
                    };
                    if let Some((content, mime_type)) = crate::api::ui::get_asset(&asset_path) {
                        let res = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n",
                            mime_type, content.len()
                        );
                        let _ = stream.write_all(res.into_bytes()).await;
                        let _ = stream.write_all(content.into_owned()).await;
                        return;
                    }
                }

                // ── Route: /v1/download/* (Unauthenticated file download) ─────
                if method == "GET" && path.starts_with("/v1/download/") {
                    let filename = path.strip_prefix("/v1/download/").unwrap_or("").to_string();
                    let decoded_filename = urlencoding::decode(&filename)
                        .unwrap_or(std::borrow::Cow::Borrowed(&filename))
                        .into_owned();

                    // Prevent path traversal
                    if decoded_filename.contains('/') || decoded_filename.contains('\\') || decoded_filename.contains("..") {
                        let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n").await;
                        return;
                    }

                    let file_path = format!("{}/{}", cfg.downloads_dir, decoded_filename);
                    match std::fs::read(&file_path) {
                        Ok(data) => {
                            let mime_type = mime_guess::from_path(&file_path).first_or_octet_stream().to_string();
                            let res = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nContent-Disposition: attachment; filename=\"{}\"\r\n\r\n",
                                mime_type, data.len(), decoded_filename
                            );
                            let _ = stream.write_all(res.into_bytes()).await;
                            let _ = stream.write_all(data).await;
                        }
                        Err(_) => {
                            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n").await;
                        }
                    }
                    return;
                }

                // ── Route: /v1/api/* (Management UI API) ─────────────────────────
                if path.starts_with("/v1/api/") {
                    let body_buf = buf.split_off(header_len);
                    crate::api::handler::handle_management_api(
                        stream,
                        method,
                        path,
                        auth,
                        content_length,
                        body_buf,
                        registry,
                        db,
                        cfg,
                        metrics,
                    ).await;
                    return;
                }

                // ── Route: /v1/*  (API) ───────────────────────────────────────
                if path.starts_with("/v1/") {
                    let body_buf = buf.split_off(header_len);
                    crate::api::handler::handle_request(
                        stream,
                        method,
                        path,
                        auth,
                        session_id,
                        idempotency_key,
                        content_length,
                        body_buf,
                        registry,
                        router,
                        db,
                        cfg,
                        metrics,
                        idempotency,
                        is_anthropic,
                    ).await;
                    return;
                }

                // ── 404 fallthrough ───────────────────────────────────────────
                log::warn!("Acceptor 404 Fallthrough for Path: {}", path);
                let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n").await;
                return;
            }
            Ok(Status::Partial) => continue,
            Err(e) => {
                log::warn!("httparse error: {:?}", e);
                return;
            }
        }
    }
}
