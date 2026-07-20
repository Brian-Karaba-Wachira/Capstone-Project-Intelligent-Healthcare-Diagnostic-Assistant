// ── Idempotency key support — §11.1 ─────────────────────────────────────────
// Accept optional `Idempotency-Key` header, cache (key → response) for a short
// window, and replay the cached response on duplicates to prevent double-execution
// of side-effectful tool calls or double-append of conversation turns.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdempotentResponse {
    pub status:      u16,
    pub body:        Vec<u8>,
    pub content_type: String,
    pub created_at:  i64,
}

struct IdemEntry {
    response:   IdempotentResponse,
    stored_at:  Instant,
    ttl:        Duration,
}

impl IdemEntry {
    fn is_expired(&self) -> bool {
        self.stored_at.elapsed() >= self.ttl
    }
}

pub struct IdempotencyStore {
    entries: Mutex<HashMap<String, IdemEntry>>,
    max_entries: usize,
    default_ttl: Duration,
}

impl IdempotencyStore {
    pub fn new(max_entries: usize, ttl_secs: u64) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries,
            default_ttl: Duration::from_secs(ttl_secs),
        }
    }

    /// Look up a cached idempotent response. Returns None if not found or expired.
    pub fn get(&self, key: &str) -> Option<IdempotentResponse> {
        let guard = self.entries.lock().unwrap();
        guard.get(key).and_then(|entry| {
            if entry.is_expired() {
                None
            } else {
                Some(entry.response.clone())
            }
        })
    }

    /// Store an idempotent response.
    pub fn put(&self, key: &str, response: IdempotentResponse) {
        let mut guard = self.entries.lock().unwrap();

        if guard.len() >= self.max_entries {
            // First pass: remove expired entries
            guard.retain(|_, e| !e.is_expired());

            // Second pass: if still over limit, evict the single oldest entry (FIFO)
            if guard.len() >= self.max_entries {
                let oldest_key = guard
                    .iter()
                    .min_by_key(|(_, e)| e.stored_at)
                    .map(|(k, _)| k.clone());
                if let Some(k) = oldest_key {
                    guard.remove(&k);
                }
            }
        }

        guard.insert(key.to_string(), IdemEntry {
            response,
            stored_at: Instant::now(),
            ttl: self.default_ttl,
        });
    }

    /// Periodic cleanup.
    pub fn cleanup(&self) {
        let mut guard = self.entries.lock().unwrap();
        guard.retain(|_, entry| !entry.is_expired());
    }
}

impl Default for IdempotencyStore {
    fn default() -> Self {
        Self::new(500, 300) // 500 entries, 5 min TTL
    }
}
