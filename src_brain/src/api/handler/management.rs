use std::sync::Arc;
use bytes::BytesMut;
use monoio::net::TcpStream;
use serde_json::{json, Value};

use crate::core::config::Config;
use crate::db::memory::Database;
use crate::worker::registry::WorkerRegistry;
use crate::api::keys::{generate_api_key, list_api_keys, delete_api_key, list_all_api_keys, generate_api_key_for_user, delete_api_key_by_value};
use crate::api::admin::{list_users, approve_user};
use crate::metrics::Metrics;

use super::common::{
    get_user_id_from_auth, read_body, require_admin_auth,
    send_json_200, send_json_error,
};

pub async fn handle_management_api(
    mut stream:     TcpStream,
    method:         String,
    path:           String,
    auth:           Option<String>,
    content_length: usize,
    body_buf:       BytesMut,
    registry:       Arc<WorkerRegistry>,
    db:             Arc<Database>,
    cfg:            Arc<Config>,
    metrics:        Arc<Metrics>,
) {
    let clean = path.split('?').next().unwrap_or(&path).to_string();

    if method == "DELETE" && clean.starts_with("/v1/api/keys/") {
        let key_id = clean.strip_prefix("/v1/api/keys/").unwrap_or("").to_string();
        if key_id.is_empty() { send_json_error(&mut stream, 400, "missing_field", "key in URL path required").await; return; }
        let user_id = match get_user_id_from_auth(&auth, &db, &cfg) { Some(id) => id, None => { send_json_error(&mut stream, 401, "unauthorized", "auth required").await; return; } };
        match delete_api_key(&db, user_id, &key_id) { Ok(_) => send_json_200(&mut stream, &json!({"ok": true})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        return;
    }


    // Admin key revocation: DELETE /v1/api/admin/keys/:key
    if method == "DELETE" && clean.starts_with("/v1/api/admin/keys/") {
        if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
        let key_id = clean.strip_prefix("/v1/api/admin/keys/").unwrap_or("").to_string();
        if key_id.is_empty() { send_json_error(&mut stream, 400, "missing_field", "key in URL path required").await; return; }
        match delete_api_key_by_value(&db, &key_id) { Ok(_) => send_json_200(&mut stream, &json!({"ok": true})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        return;
    }

    if method == "POST" && clean.contains("/v1/api/admin/users/") && clean.ends_with("/approve") {
        if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
        let id_str = clean.strip_prefix("/v1/api/admin/users/").unwrap_or("").strip_suffix("/approve").unwrap_or("");
        let user_id: i64 = match id_str.parse() { Ok(id) => id, Err(_) => { send_json_error(&mut stream, 400, "bad_id", "invalid user id").await; return; } };
        match approve_user(&db, user_id) { Ok(_) => send_json_200(&mut stream, &json!({"ok": true})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        return;
    }

    // MISSING-1: Delete a user account (admin only)
    if method == "DELETE" && clean.contains("/v1/api/admin/users/") {
        if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
        let id_str = clean.strip_prefix("/v1/api/admin/users/").unwrap_or("");
        let user_id: i64 = match id_str.parse() { Ok(id) => id, Err(_) => { send_json_error(&mut stream, 400, "bad_id", "invalid user id").await; return; } };
        match crate::api::admin::delete_user(&db, user_id) { Ok(_) => send_json_200(&mut stream, &json!({"ok": true})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        return;
    }

    // MISSING-1: Toggle admin status for a user (admin only)
    if method == "POST" && clean.contains("/v1/api/admin/users/") && (clean.ends_with("/admin") || clean.ends_with("/revoke-admin")) {
        if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
        let is_grant = clean.ends_with("/admin");
        let id_str = clean.strip_prefix("/v1/api/admin/users/").unwrap_or("")
            .strip_suffix(if is_grant { "/admin" } else { "/revoke-admin" }).unwrap_or("");
        let user_id: i64 = match id_str.parse() { Ok(id) => id, Err(_) => { send_json_error(&mut stream, 400, "bad_id", "invalid user id").await; return; } };
        match crate::api::admin::set_user_admin(&db, user_id, is_grant) { Ok(_) => send_json_200(&mut stream, &json!({"ok": true, "is_admin": is_grant})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        return;
    }

    match (method.as_str(), clean.as_str()) {

        // ── Auth ──────────────────────────────────────────────────────────
        ("POST", "/v1/api/auth/google") => {
            let body = read_body(&mut stream, body_buf, content_length).await.unwrap_or_default();
            let val: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
            let id_token = match val.get("id_token").and_then(|v| v.as_str()) { Some(t) => t, None => { send_json_error(&mut stream, 400, "missing_field", "id_token required").await; return; } };
            match crate::api::auth::handle_google_auth(&db, id_token, &cfg.jwt_secret) { Ok(jwt) => send_json_200(&mut stream, &json!({"token": jwt})).await, Err(e) => send_json_error(&mut stream, 401, "auth_failed", &e).await, }
        }

        ("GET", "/v1/api/auth/me") => {
            let user_id = match get_user_id_from_auth(&auth, &db, &cfg) { Some(id) => id, None => { send_json_error(&mut stream, 401, "unauthorized", "auth required").await; return; } };
            match crate::api::auth::get_user(&db, user_id) { Ok(u) => send_json_200(&mut stream, &u).await, Err(e) => send_json_error(&mut stream, 404, "not_found", &e).await, }
        }

        // ── Workers ───────────────────────────────────────────────────────
        ("GET", "/v1/api/workers") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let workers = registry.list_all();
            send_json_200(&mut stream, &json!({ "workers": workers })).await;
        }

        // ── Admin stats ───────────────────────────────────────────────────
        ("GET", "/v1/api/admin/stats") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let (total_users, active_users) = db.get_admin_stats().unwrap_or((0, 0));
            let workers = registry.list_all();
            let active_workers = workers.len();
            let mut total_requests = 0;
            for w in &workers { total_requests += w.active_requests; }
            let resource_load = if active_workers > 0 { ((total_requests as f64 / (active_workers as f64 * 100.0)) * 100.0) as u32 } else { 0 };
            let uptime = crate::START_TIME.get().map(|start| start.elapsed().as_secs()).unwrap_or(0);
            send_json_200(&mut stream, &json!({
                "compute_workers": { "active": active_workers, "total": active_workers },
                "active_users": { "active": active_users, "total": total_users },
                "uptime": uptime, "resource_load": resource_load
            })).await;
        }

        ("GET", "/v1/api/admin/users") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            match list_users(&db) { Ok(u) => send_json_200(&mut stream, &json!({"users": u})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        }

        ("POST", "/v1/api/admin/users/approve") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let body = read_body(&mut stream, body_buf, content_length).await.unwrap_or_default();
            let val: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
            let user_id = match val.get("user_id").and_then(|v| v.as_i64()) { Some(id) => id, None => { send_json_error(&mut stream, 400, "missing_field", "user_id required").await; return; } };
            match approve_user(&db, user_id) { Ok(_) => send_json_200(&mut stream, &json!({"ok": true})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        }

        ("GET", "/v1/api/admin/downloads") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let mut files = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&cfg.downloads_dir) {
                for entry in entries.flatten() { if let Ok(m) = entry.metadata() { if m.is_file() { if let Ok(n) = entry.file_name().into_string() { files.push(n); } } } }
            }
            send_json_200(&mut stream, &json!({"files": files, "dir": cfg.downloads_dir})).await;
        }

        // ── Metrics snapshot ──────────────────────────────────────────────
        ("GET", "/v1/api/admin/metrics") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let prom = metrics.render_prometheus();
            send_json_200(&mut stream, &json!({"prometheus": prom})).await;
        }

        // ── Telemetry ─────────────────────────────────────────────────────
        ("GET", "/v1/api/admin/telemetry") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let session = path.split("session=").nth(1).unwrap_or("");
            if session.is_empty() {
                send_json_200(&mut stream, &json!({"metrics": [], "note": "add ?session=SESSION_ID"})).await;
            } else {
                match db.get_turn_metrics(session) {
                    Ok(data) => {
                        let items: Vec<Value> = data.into_iter().map(|(w, tps, pt, ct)| json!({
                            "worker_id": w, "tokens_per_sec": tps, "prompt_tokens": pt, "completion_tokens": ct
                        })).collect();
                        send_json_200(&mut stream, &json!({"session": session, "metrics": items})).await;
                    }
                    Err(e) => send_json_error(&mut stream, 500, "db_error", &e.to_string()).await,
                }
            }
        }

        // ── API Keys ──────────────────────────────────────────────────────
        ("GET", "/v1/api/keys") => {
            let user_id = match get_user_id_from_auth(&auth, &db, &cfg) { Some(id) => id, None => { send_json_error(&mut stream, 401, "unauthorized", "auth required").await; return; } };
            match list_api_keys(&db, user_id) { Ok(keys) => send_json_200(&mut stream, &json!({"keys": keys})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        }
        ("POST", "/v1/api/keys") => {
            let user_id = match get_user_id_from_auth(&auth, &db, &cfg) { Some(id) => id, None => { send_json_error(&mut stream, 401, "unauthorized", "auth required").await; return; } };
            let body = read_body(&mut stream, body_buf, content_length).await.unwrap_or_default();
            let val: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
            let name = val.get("name").and_then(|v| v.as_str()).unwrap_or("default");
            match generate_api_key(&db, user_id, name) { Ok(key) => send_json_200(&mut stream, &json!({"key": key})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        }


        // ── Admin Key Management ──────────────────────────────────────────
        ("GET", "/v1/api/admin/keys") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            match list_all_api_keys(&db) { Ok(keys) => send_json_200(&mut stream, &json!({"keys": keys})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        }
        ("POST", "/v1/api/admin/keys") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let body = read_body(&mut stream, body_buf, content_length).await.unwrap_or_default();
            let val: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
            let user_id = val.get("user_id").and_then(|v| v.as_i64()).unwrap_or(0);
            let name = val.get("name").and_then(|v| v.as_str()).unwrap_or("default");
            if user_id == 0 { send_json_error(&mut stream, 400, "missing_field", "user_id required").await; return; }
            match generate_api_key_for_user(&db, user_id, name) { Ok(key) => send_json_200(&mut stream, &json!({"key": key})).await, Err(e) => send_json_error(&mut stream, 500, "db_error", &e).await, }
        }

        ("GET", "/v1/api/admin/settings/hosted-tools") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let s_url = db.get_setting("searxng_url");
            send_json_200(&mut stream, &json!({
                "searxng_url": s_url,
                "searxng_configured": s_url.as_deref().filter(|s| !s.trim().is_empty()).is_some(),
                "hosted_tool_max_rounds": cfg.hosted_tool_max_rounds,
            })).await;
        }

        ("POST", "/v1/api/admin/settings/hosted-tools") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let body = read_body(&mut stream, body_buf, content_length).await.unwrap_or_default();
            let val: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
            if let Some(s) = val.get("searxng_url").and_then(|v| v.as_str()) {
                let _ = db.set_setting("searxng_url", s);
            }
            send_json_200(&mut stream, &json!({"ok": true})).await;
        }

        ("POST", "/v1/api/admin/settings/hosted-tools/test") => {
            if !require_admin_auth(&auth, &db, &cfg) { send_json_error(&mut stream, 403, "forbidden", "Admin required").await; return; }
            let s_url = db.get_setting("searxng_url");
            match s_url.as_deref().filter(|s| !s.trim().is_empty()) {
                None => send_json_error(&mut stream, 400, "not_configured", "searxng_url is not set in db").await,
                Some(base) => {
                    let url = format!("{}/search?q=connectivity+check&format=json", base.trim_end_matches('/'));
                    // Blocking call, same tradeoff noted in hosted_tools.rs —
                    // acceptable here since this only runs when an admin
                    // clicks "Test Connection", not on the request hot path.
                    match ureq::get(&url).call() {
                        Ok(resp) => {
                            let status = resp.status().as_u16();
                            let body_text = resp.into_body().read_to_string().unwrap_or_default();
                            let json_ok = serde_json::from_str::<Value>(&body_text).is_ok();
                            send_json_200(&mut stream, &json!({
                                "ok": true, "status": status,
                                "returned_json": json_ok,
                                "note": if json_ok { "Reachable and returning JSON." } else { "Reachable, but the response wasn't valid JSON — check that `search.formats` includes `json` in SearxNG's settings.yml." },
                            })).await;
                        }
                        Err(e) => send_json_error(&mut stream, 502, "unreachable", &format!("Could not reach SearxNG at {base}: {e}")).await,
                    }
                }
            }
        }

        // ── Usage & Health ────────────────────────────────────────────────
        ("GET", "/v1/api/usage") => {
            let user_id = match get_user_id_from_auth(&auth, &db, &cfg) { Some(id) => id, None => { send_json_error(&mut stream, 401, "unauthorized", "auth required").await; return; } };
            let (today, month) = db.get_user_usage(user_id).unwrap_or((0, 0));
            send_json_200(&mut stream, &json!({"todayTokens": today, "monthTokens": month, "estCost": (month as f64) * 0.000002})).await;
        }

        ("GET", "/v1/api/health") => {
            let worker_count = registry.count();
            send_json_200(&mut stream, &json!({"ok": true, "workers": worker_count, "status": if worker_count > 0 { "ready" } else { "no_workers" }})).await;
        }

        _ => {
            send_json_error(&mut stream, 404, "not_found", &format!("Unknown: {}", clean)).await;
        }
    }
}