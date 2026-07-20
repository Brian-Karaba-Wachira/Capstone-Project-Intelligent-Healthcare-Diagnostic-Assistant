use serde::{Deserialize, Serialize};
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::db::memory::Database;
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: usize,
    pub is_admin: bool,
    pub email: String,
}

pub fn create_jwt(user_id: i64, email: &str, is_admin: bool, secret: &str) -> String {
    let expiration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() + 24 * 3600; // 1 day

    let claims = Claims {
        sub: user_id.to_string(),
        email: email.to_string(),
        is_admin,
        exp: expiration as usize,
    };

    encode(&Header::default(), &claims, &EncodingKey::from_secret(secret.as_ref())).unwrap()
}

pub fn verify_jwt(token: &str, secret: &str) -> Option<Claims> {
    let mut validation = Validation::default();
    validation.validate_exp = true;

    match decode::<Claims>(token, &DecodingKey::from_secret(secret.as_ref()), &validation) {
        Ok(c) => Some(c.claims),
        Err(_) => None,
    }
}

/// DB-backed API key lookup ("full" = goes all the way to the DB rather than
/// being satisfied by the PSK shortcut or a JWT). This is what `handler`
/// imports — it didn't exist before, which is why the build was failing.
/// Bumps `last_used` on the key as a side effect (via Database::lookup_api_key).
pub fn lookup_api_key_full(db: &Arc<Database>, token: &str) -> bool {
    db.lookup_api_key(token)
}

// Handler for /api/auth/google
pub fn handle_google_auth(db: &Arc<Database>, id_token: &str, secret: &str) -> Result<String, String> {
    // We use a simple ureq call to Google's tokeninfo endpoint
    let url = format!("https://oauth2.googleapis.com/tokeninfo?id_token={}", id_token);
    let resp = ureq::get(&url).call().map_err(|e| e.to_string())?;

    let body = resp.into_body().read_to_string().map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;

    if let Some(error) = json.get("error_description") {
        return Err(error.as_str().unwrap_or("Invalid token").to_string());
    }

    let email = json.get("email").and_then(|v| v.as_str()).ok_or("No email in token")?;
    let name = json.get("name").and_then(|v| v.as_str()).unwrap_or("");

    let conn = db.conn.lock().unwrap();

    // Check if user exists
    let mut stmt = conn.prepare("SELECT id, is_admin, is_approved FROM users WHERE email = ?1").unwrap();
    let mut rows = stmt.query([email]).unwrap();

    let (user_id, is_admin, is_approved) = if let Some(row) = rows.next().unwrap() {
        let id: i64 = row.get(0).unwrap();
        let is_admin: bool = row.get(1).unwrap();
        let is_approved: bool = row.get(2).unwrap();
        (id, is_admin, is_approved)
    } else {
        // Drop rows lock
        drop(rows);
        drop(stmt);

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

        // If this is the very first user, make them admin and approved automatically
        let mut count_stmt = conn.prepare("SELECT COUNT(*) FROM users").unwrap();
        let count: i64 = count_stmt.query_row([], |row| row.get(0)).unwrap();

        let is_admin = count == 0;
        let is_approved = count == 0;

        conn.execute(
            "INSERT INTO users (email, name, is_admin, is_approved, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![email, name, is_admin, is_approved, now],
        ).unwrap();

        let id = conn.last_insert_rowid();
        (id, is_admin, is_approved)
    };

    if !is_approved {
        return Err("Account pending administrator approval".to_string());
    }

    Ok(create_jwt(user_id, email, is_admin, secret))
}

pub fn get_user(db: &std::sync::Arc<crate::db::memory::Database>, user_id: i64) -> Result<crate::api::admin::UserInfo, String> {
    let conn = db.conn.lock().unwrap();
    let mut stmt = conn.prepare("SELECT id, email, name, is_admin, is_approved, created_at FROM users WHERE id = ?1").unwrap();
    stmt.query_row([user_id], |row| {
        Ok(crate::api::admin::UserInfo {
            id: row.get(0)?,
            email: row.get(1)?,
            name: row.get(2)?,
            is_admin: row.get(3)?,
            is_approved: row.get(4)?,
            created_at: row.get(5)?,
        })
    }).map_err(|e| e.to_string())
}