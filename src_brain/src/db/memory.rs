use rusqlite::{params, Connection, Result, OpenFlags};
use std::sync::Mutex;


pub struct Database {
    pub conn: Mutex<Connection>,
}

impl Database {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_URI,
        )?;

        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS workers (
                worker_id    TEXT    PRIMARY KEY,
                model        TEXT    NOT NULL,
                gpu          TEXT,
                connected_at INTEGER NOT NULL,
                last_seen    INTEGER,
                total_tasks  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email       TEXT    UNIQUE NOT NULL,
                name        TEXT,
                is_admin    BOOLEAN DEFAULT FALSE,
                is_approved BOOLEAN DEFAULT FALSE,
                created_at  INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                key         TEXT    PRIMARY KEY,
                user_id     INTEGER NOT NULL REFERENCES users(id),
                name        TEXT    NOT NULL,
                created_at  INTEGER NOT NULL,
                last_used   INTEGER
            );

            -- §9.3 Turn-level telemetry + token/cost accounting
            CREATE TABLE IF NOT EXISTS turn_metrics (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT NOT NULL,
                request_id        TEXT NOT NULL,
                worker_id         TEXT NOT NULL,
                user_id           INTEGER,
                model             TEXT,
                ttft_ms           INTEGER,
                tokens_per_sec    REAL,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                cache_hit_tokens  INTEGER DEFAULT 0,
                cost_usd          REAL DEFAULT 0,
                created_at        INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_turn_metrics_user_time ON turn_metrics(user_id, created_at);

            -- §11.1 Idempotency cache (persisted across restarts)
            CREATE TABLE IF NOT EXISTS idempotency_cache (
                idem_key    TEXT PRIMARY KEY,
                status      INTEGER NOT NULL,
                body        BLOB NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'application/json',
                created_at  INTEGER NOT NULL
            );

            -- Settings (dynamic configuration)
            CREATE TABLE IF NOT EXISTS kv_settings (
                key         TEXT PRIMARY KEY,
                value       TEXT
            );
")?;

        // Migration for existing databases: turn_metrics predates the
        // user_id/model/cost_usd columns, so add them if missing.
        {
            let mut existing_cols = std::collections::HashSet::new();
            let mut stmt = conn.prepare("PRAGMA table_info(turn_metrics)")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
            for r in rows { if let Ok(name) = r { existing_cols.insert(name); } }
            drop(stmt);
            if !existing_cols.contains("user_id") {
                conn.execute_batch("ALTER TABLE turn_metrics ADD COLUMN user_id INTEGER;")?;
            }
            if !existing_cols.contains("model") {
                conn.execute_batch("ALTER TABLE turn_metrics ADD COLUMN model TEXT;")?;
            }
            if !existing_cols.contains("cost_usd") {
                conn.execute_batch("ALTER TABLE turn_metrics ADD COLUMN cost_usd REAL DEFAULT 0;")?;
            }
        }

        // Migration to initialize kv_settings defaults if missing
        {
            let mut stmt = conn.prepare("SELECT COUNT(*) FROM kv_settings WHERE key = 'searxng_url'")?;
            let count: i64 = stmt.query_row([], |row| row.get(0)).unwrap_or(0);
            if count == 0 {
                conn.execute(
                    "INSERT INTO kv_settings (key, value) VALUES (?1, ?2)",
                    params!["searxng_url", "http://127.0.0.1:8119"],
                )?;
            }
        }

        Ok(Self { conn: Mutex::new(conn) })
    }

    fn now() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
    }

    // ── Settings ──────────────────────────────────────────────────────────────

    pub fn get_setting(&self, key: &str) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT value FROM kv_settings WHERE key = ?1", params![key], |row| row.get(0)).ok()
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO kv_settings (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Record turn-level telemetry for a completed request, tagged with the
    /// user_id (if authenticated via API key/JWT) and model, and the USD
    /// cost of the turn (computed by the caller via `Config::calc_cost_usd`).
    /// This is what makes token counts and cost visible per-user for the
    /// admin/usage endpoints, instead of only being recorded per-session.
    pub fn record_turn_metrics(
        &self,
        session_id:    &str,
        request_id:    &str,
        worker_id:     &str,
        user_id:       Option<i64>,
        model:         &str,
        ttft_ms:       u32,
        tokens_per_sec: f32,
        prompt_tokens:  u32,
        comp_tokens:    u32,
        cache_hit_tokens: u32,
        cost_usd:       f64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Self::now();
        conn.execute(
            "INSERT INTO turn_metrics (session_id, request_id, worker_id, user_id, model,
             ttft_ms, tokens_per_sec, prompt_tokens, completion_tokens, cache_hit_tokens,
             cost_usd, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![session_id, request_id, worker_id, user_id, model, ttft_ms as i64,
                    tokens_per_sec, prompt_tokens as i64, comp_tokens as i64,
                    cache_hit_tokens as i64, cost_usd, now],
        )?;
        Ok(())
    }

    /// Get turn metrics for a session (for debugging/analysis).
    pub fn get_turn_metrics(&self, session_id: &str) -> Result<Vec<(String, f32, u32, u32)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT worker_id, tokens_per_sec, prompt_tokens, completion_tokens
             FROM turn_metrics WHERE session_id = ?1 ORDER BY created_at"
        )?;
        let rows = stmt.query_map(params![session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, f32>(1)?,
                row.get::<_, u32>(2)?,
                row.get::<_, u32>(3)?,
            ))
        })?;
        rows.collect()
    }

    // ── API keys ──────────────────────────────────────────────────────────────

    /// Returns true + updates last_used if key exists.
    pub fn lookup_api_key(&self, token: &str) -> bool {
        self.lookup_api_key_with_user(token).is_some()
    }

    /// Returns the user_id if key is valid, updating last_used.
    pub fn lookup_api_key_with_user(&self, token: &str) -> Option<i64> {
        let conn = self.conn.lock().unwrap();
        let user_id: rusqlite::Result<i64> = conn.query_row(
            "SELECT user_id FROM api_keys WHERE key = ?1",
            params![token],
            |row| row.get(0),
        );
        if let Ok(id) = user_id {
            let now = Self::now();
            let _ = conn.execute(
                "UPDATE api_keys SET last_used = ?1 WHERE key = ?2",
                params![now, token],
            );
            Some(id)
        } else {
            None
        }
    }

    // ── Usage & Stats ─────────────────────────────────────────────────────────

    /// Sum of (prompt+completion) tokens for a user: (today, this calendar month).
    /// Backed by turn_metrics.user_id, populated by record_turn_metrics on every
    /// completed turn.
    pub fn get_user_usage(&self, user_id: i64) -> Result<(i64, i64)> {
        let conn = self.conn.lock().unwrap();
        let now = Self::now();
        let today_start = now - (now % 86400);
        let month_start = now - 30 * 86400; // rolling 30-day window, not calendar month

        let today: i64 = conn.query_row(
            "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM turn_metrics
             WHERE user_id = ?1 AND created_at >= ?2",
            params![user_id, today_start],
            |row| row.get(0),
        ).unwrap_or(0);

        let month: i64 = conn.query_row(
            "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM turn_metrics
             WHERE user_id = ?1 AND created_at >= ?2",
            params![user_id, month_start],
            |row| row.get(0),
        ).unwrap_or(0);

        Ok((today, month))
    }

    /// USD cost for a user over the same two windows as `get_user_usage`.
    pub fn get_user_cost(&self, user_id: i64) -> Result<(f64, f64)> {
        let conn = self.conn.lock().unwrap();
        let now = Self::now();
        let today_start = now - (now % 86400);
        let month_start = now - 30 * 86400;

        let today: f64 = conn.query_row(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM turn_metrics WHERE user_id = ?1 AND created_at >= ?2",
            params![user_id, today_start],
            |row| row.get(0),
        ).unwrap_or(0.0);

        let month: f64 = conn.query_row(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM turn_metrics WHERE user_id = ?1 AND created_at >= ?2",
            params![user_id, month_start],
            |row| row.get(0),
        ).unwrap_or(0.0);

        Ok((today, month))
    }

    pub fn get_admin_stats(&self) -> Result<(i64, i64)> {
        let conn = self.conn.lock().unwrap();
        let now = Self::now();

        let total_users: i64 = conn.query_row("SELECT COUNT(*) FROM users", [], |row| row.get(0)).unwrap_or(0);
        // Count distinct users who used any API key in the last 24 hours
        let active_users: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT user_id) FROM api_keys WHERE last_used >= ?1",
            params![now - 86400],   // 24-hour activity window
            |row| row.get(0)
        ).unwrap_or(0);

        Ok((total_users, active_users))
    }

    /// Check if a user account has admin privileges.
    pub fn is_user_admin(&self, user_id: i64) -> bool {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT is_admin FROM users WHERE id = ?1",
            params![user_id],
            |row| row.get::<_, bool>(0),
        ).unwrap_or(false)
    }

    // ── Workers DB persistence ────────────────────────────────────────────────

    /// Upsert a worker record when it registers or sends a heartbeat.
    pub fn upsert_worker(&self, worker_id: &str, model: &str, gpu: &str) {
        let conn = self.conn.lock().unwrap();
        let now = Self::now();
        let _ = conn.execute(
            "INSERT INTO workers (worker_id, model, gpu, connected_at, last_seen)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(worker_id) DO UPDATE SET
                 model    = excluded.model,
                 gpu      = excluded.gpu,
                 last_seen = excluded.last_seen",
            params![worker_id, model, gpu, now],
        );
    }

    /// Remove a worker record when it disconnects or times out.
    pub fn remove_worker(&self, worker_id: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute("DELETE FROM workers WHERE worker_id = ?1", params![worker_id]);
    }

    /// Bump last_seen timestamp on heartbeat.
    pub fn touch_worker(&self, worker_id: &str) {
        let conn = self.conn.lock().unwrap();
        let now = Self::now();
        let _ = conn.execute(
            "UPDATE workers SET last_seen = ?1 WHERE worker_id = ?2",
            params![now, worker_id],
        );
    }

    // ── Idempotency cache (§11.1) ─────────────────────────────────────────────

    /// Check for a cached idempotent response. Returns None if not found or expired.
    pub fn get_idempotent_response(&self, idem_key: &str, ttl_s: i64) -> Option<(u16, Vec<u8>, String)> {
        let conn = self.conn.lock().unwrap();
        let now = Self::now();
        conn.query_row(
            "SELECT status, body, content_type FROM idempotency_cache WHERE idem_key = ?1 AND created_at > ?2",
            params![idem_key, now - ttl_s],
            |row| {
                Ok((
                    row.get::<_, i64>(0)? as u16,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        ).ok()
    }

    /// Store an idempotent response.
    pub fn put_idempotent_response(
        &self,
        idem_key:     &str,
        status:       u16,
        body:         &[u8],
        content_type: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Self::now();
        // Clean old entries periodically
        let _ = conn.execute(
            "DELETE FROM idempotency_cache WHERE created_at < ?1",
            params![now - 3600], // 1 hour cleanup
        );
        conn.execute(
            "INSERT OR REPLACE INTO idempotency_cache (idem_key, status, body, content_type, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![idem_key, status as i64, body, content_type, now],
        )?;
        Ok(())
    }

}

