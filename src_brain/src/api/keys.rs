use crate::db::memory::Database;
use serde::Serialize;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use rand::{distr::Alphanumeric, RngExt};

#[derive(Serialize)]
pub struct ApiKey {
    pub key: String,
    pub name: String,
    pub user_id: i64,
    pub user_email: Option<String>,
    pub created_at: i64,
    pub last_used: Option<i64>,
}

pub fn generate_api_key(db: &Arc<Database>, user_id: i64, name: &str) -> Result<ApiKey, String> {
    let conn = db.conn.lock().unwrap();

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM api_keys WHERE user_id = ?1 AND name = ?2",
        rusqlite::params![user_id, name],
        |row| row.get(0),
    ).unwrap_or(0);

    if count > 0 {
        return Err("An API key with this name already exists".to_string());
    }

    let raw_key: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(48)
        .map(char::from)
        .collect();

    let key = format!("sk-{}", raw_key);
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

    conn.execute(
        "INSERT INTO api_keys (key, user_id, name, created_at) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![key, user_id, name, now],
    ).map_err(|e| e.to_string())?;

    // Fetch user email for response
    let email: Option<String> = conn.query_row(
        "SELECT email FROM users WHERE id = ?1",
        rusqlite::params![user_id],
        |row| row.get(0),
    ).ok();

    Ok(ApiKey {
        key,
        name: name.to_string(),
        user_id,
        user_email: email,
        created_at: now,
        last_used: None,
    })
}

pub fn list_api_keys(db: &Arc<Database>, user_id: i64) -> Result<Vec<ApiKey>, String> {
    let conn = db.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT k.key, k.name, k.created_at, k.last_used, u.email \
         FROM api_keys k LEFT JOIN users u ON k.user_id = u.id WHERE k.user_id = ?1"
    ).unwrap();
    let rows = stmt.query_map([user_id], |row| {
        Ok(ApiKey {
            key: row.get(0)?,
            name: row.get(1)?,
            user_id,
            user_email: row.get(4)?,
            created_at: row.get(2)?,
            last_used: row.get(3)?,
        })
    }).map_err(|e| e.to_string())?;

    let mut keys = Vec::new();
    for row in rows {
        if let Ok(k) = row {
            keys.push(k);
        }
    }
    Ok(keys)
}

pub fn delete_api_key(db: &Arc<Database>, user_id: i64, key: &str) -> Result<(), String> {
    let conn = db.conn.lock().unwrap();
    conn.execute(
        "DELETE FROM api_keys WHERE user_id = ?1 AND key = ?2",
        rusqlite::params![user_id, key],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

// ── Admin key management ──────────────────────────────────────────────────

/// List ALL API keys across all users (admin only).
pub fn list_all_api_keys(db: &Arc<Database>) -> Result<Vec<ApiKey>, String> {
    let conn = db.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT k.key, k.name, k.user_id, k.created_at, k.last_used, u.email \
         FROM api_keys k LEFT JOIN users u ON k.user_id = u.id ORDER BY k.created_at DESC"
    ).unwrap();
    let rows = stmt.query_map([], |row| {
        let uid: i64 = row.get(2)?;
        Ok(ApiKey {
            key: row.get(0)?,
            name: row.get(1)?,
            user_id: uid,
            user_email: row.get(5)?,
            created_at: row.get(3)?,
            last_used: row.get(4)?,
        })
    }).map_err(|e| e.to_string())?;

    let mut keys = Vec::new();
    for row in rows {
        if let Ok(k) = row { keys.push(k); }
    }
    Ok(keys)
}

/// Generate an API key for any user (admin only).
pub fn generate_api_key_for_user(db: &Arc<Database>, user_id: i64, name: &str) -> Result<ApiKey, String> {
    generate_api_key(db, user_id, name)
}

/// Delete an API key by its literal key value (admin only).
pub fn delete_api_key_by_value(db: &Arc<Database>, key: &str) -> Result<(), String> {
    let conn = db.conn.lock().unwrap();
    conn.execute(
        "DELETE FROM api_keys WHERE key = ?1",
        rusqlite::params![key],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Validates an API key from the DB. Returns true if valid, and updates last_used.
pub fn lookup_api_key(db: &Arc<Database>, token: &str) -> bool {
    let conn = db.conn.lock().unwrap();
    let exists: rusqlite::Result<i64> = conn.query_row(
        "SELECT user_id FROM api_keys WHERE key = ?1",
        rusqlite::params![token],
        |row| row.get(0),
    );
    if exists.is_ok() {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        let _ = conn.execute(
            "UPDATE api_keys SET last_used = ?1 WHERE key = ?2",
            rusqlite::params![now, token],
        );
        true
    } else {
        false
    }
}
