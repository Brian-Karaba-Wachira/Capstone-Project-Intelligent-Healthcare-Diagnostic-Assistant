use crate::db::memory::Database;
use serde::Serialize;
use std::sync::Arc;

#[derive(Serialize)]
pub struct UserInfo {
    pub id: i64,
    pub email: String,
    pub name: Option<String>,
    pub is_admin: bool,
    pub is_approved: bool,
    pub created_at: i64,
}

pub fn list_users(db: &Arc<Database>) -> Result<Vec<UserInfo>, String> {
    let conn = db.conn.lock().unwrap();
    let mut stmt = conn.prepare("SELECT id, email, name, is_admin, is_approved, created_at FROM users").unwrap();
    let rows = stmt.query_map([], |row| {
        Ok(UserInfo {
            id: row.get(0)?,
            email: row.get(1)?,
            name: row.get(2)?,
            is_admin: row.get(3)?,
            is_approved: row.get(4)?,
            created_at: row.get(5)?,
        })
    }).map_err(|e| e.to_string())?;

    let mut users = Vec::new();
    for row in rows {
        if let Ok(u) = row {
            users.push(u);
        }
    }
    Ok(users)
}

pub fn approve_user(db: &Arc<Database>, user_id: i64) -> Result<(), String> {
    let conn = db.conn.lock().unwrap();
    conn.execute(
        "UPDATE users SET is_approved = TRUE WHERE id = ?1",
        rusqlite::params![user_id],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Revoke approval / block a user account.
pub fn revoke_user(db: &Arc<Database>, user_id: i64) -> Result<(), String> {
    let conn = db.conn.lock().unwrap();
    conn.execute(
        "UPDATE users SET is_approved = FALSE WHERE id = ?1",
        rusqlite::params![user_id],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Permanently delete a user account and all associated API keys.
pub fn delete_user(db: &Arc<Database>, user_id: i64) -> Result<(), String> {
    let conn = db.conn.lock().unwrap();
    // Delete API keys first (FK constraint)
    conn.execute(
        "DELETE FROM api_keys WHERE user_id = ?1",
        rusqlite::params![user_id],
    ).map_err(|e| e.to_string())?;
    conn.execute(
        "DELETE FROM users WHERE id = ?1",
        rusqlite::params![user_id],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Grant or revoke admin privileges for a user.
pub fn set_user_admin(db: &Arc<Database>, user_id: i64, is_admin: bool) -> Result<(), String> {
    let conn = db.conn.lock().unwrap();
    conn.execute(
        "UPDATE users SET is_admin = ?1 WHERE id = ?2",
        rusqlite::params![is_admin, user_id],
    ).map_err(|e| e.to_string())?;
    Ok(())
}
