use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;
use flume::Sender;
use serde::Serialize;
use crate::core::protocol::WorkerMessage;

pub struct WorkerInfo {
    pub worker_id:       String,
    pub model:           String,
    pub gpu:             String,
    pub vram_free_mb:    u32,
    pub max_context:     u32,
    pub active_requests: u32,
    pub last_heartbeat:  Instant,
    pub connected_at:    Instant,
    pub tx:              Sender<WorkerMessage>,
    /// Wire format this worker speaks natively ("openai" today; reserved
    /// for future non-OpenAI-shaped workers). Drives whether the gateway
    /// needs to run a translation pass for requests routed to it.
    pub api_type:        String,
}

/// Serializable snapshot of a worker — safe to send to the admin UI.
#[derive(Serialize)]
pub struct WorkerStatus {
    pub worker_id:       String,
    pub model:           String,
    pub gpu:             String,
    pub vram_free_mb:    u32,
    pub max_context:     u32,
    pub active_requests: u32,
    pub last_heartbeat_s: u64,   // seconds since last heartbeat
    pub connected_s:      u64,   // seconds since connected
    pub api_type:         String,
}

pub struct WorkerRegistry {
    workers: RwLock<HashMap<String, WorkerInfo>>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self { workers: RwLock::new(HashMap::new()) }
    }

    pub fn register(&self, info: WorkerInfo) {
        log::info!("Worker {} registered (model={}, gpu={}, vram={}MB, ctx={}, api_type={})",
            info.worker_id, info.model, info.gpu, info.vram_free_mb, info.max_context, info.api_type);
        self.workers.write().unwrap().insert(info.worker_id.clone(), info);
    }

    pub fn deregister(&self, id: &str) {
        if self.workers.write().unwrap().remove(id).is_some() {
            log::warn!("Worker {} deregistered", id);
        }
    }

    pub fn increment_active(&self, id: &str) {
        if let Some(w) = self.workers.write().unwrap().get_mut(id) {
            w.active_requests = w.active_requests.saturating_add(1);
        }
    }

    pub fn decrement_active(&self, id: &str) {
        if let Some(w) = self.workers.write().unwrap().get_mut(id) {
            w.active_requests = w.active_requests.saturating_sub(1);
        }
    }

    pub fn update_load(&self, id: &str, active: u32, vram: u32) {
        if let Some(w) = self.workers.write().unwrap().get_mut(id) {
            w.active_requests = active;
            w.vram_free_mb    = vram;
            w.last_heartbeat  = Instant::now();
        }
    }

    pub fn heartbeat(&self, id: &str) {
        if let Some(w) = self.workers.write().unwrap().get_mut(id) {
            w.last_heartbeat = Instant::now();
        }
    }

    /// Update the model name for a worker (sent on each heartbeat).
    pub fn update_model(&self, id: &str, model: &str) {
        if let Some(w) = self.workers.write().unwrap().get_mut(id) {
            if w.model != model {
                log::info!("Worker {} model updated: {} → {}", id, w.model, model);
                w.model = model.to_string();
            }
        }
    }

    /// Pick the worker with the fewest active requests.
    /// Tries an exact model match first. If no worker matches the requested model,
    /// automatically falls back to any available worker (lenient routing).
    /// Logs a warning on fallback so operators know the requested model was not found.
    ///
    /// Returns the worker's `tx` AND its reported `max_context` — callers use this
    /// instead of guessing a context-window constant, since different workers can
    /// be running the same crate against very different `n_ctx` values (32k vs
    /// 256k vs 1M depending on the model / how llama-server was launched).
    /// Returns (tx, worker_id, max_context). `worker_id` lets callers register
    /// the pending request against the specific tunnel connection it was
    /// dispatched to, so the tunnel can fail it fast if that connection dies
    /// mid-flight instead of the client waiting out the full idle timeout.
    pub fn pick_best(&self, model: Option<&str>) -> Option<(Sender<WorkerMessage>, String, u32)> {
        let workers = self.workers.read().unwrap();

        // Try model-specific match first
        if let Some(m) = model {
            let candidates: Vec<_> = workers
                .values()
                .filter(|w| {
                    w.model == m || w.model.contains(m) || m.contains(&w.model)
                })
                .collect();

            if let Some(best) = candidates.iter().min_by_key(|w| w.active_requests) {
                return Some((best.tx.clone(), best.worker_id.clone(), best.max_context));
            }

            // No worker matches the requested model — lenient fallback to any available worker
            log::warn!(
                "model={} has no matching active workers — falling back to any available worker",
                m
            );
            // Log what models ARE available so operators can fix the alias mapping
            let available: Vec<&str> = workers.values().map(|w| w.model.as_str()).collect();
            if !available.is_empty() {
                log::info!("available models for fallback: {}", available.join(", "));
            }
        }

        // Fallback: pick any worker (least loaded)
        workers
            .values()
            .min_by_key(|w| w.active_requests)
            .map(|w| (w.tx.clone(), w.worker_id.clone(), w.max_context))
    }

    /// Lenient variant: same as pick_best but guaranteed to fall back to
    /// any active worker if the requested model is not found.
    /// This is the preferred method for chat routing.
    /// Returns (tx, worker_id, actual_model_name, max_context). `worker_id`
    /// lets callers register the pending request against the specific
    /// tunnel connection it was dispatched to (see `pick_best`).
    pub fn pick_best_lenient(&self, model: &str) -> Option<(Sender<WorkerMessage>, String, String, u32)> {
        let workers = self.workers.read().unwrap();

        // Try model-specific match first
        let candidates: Vec<_> = workers
            .values()
            .filter(|w| {
                w.model == model || w.model.contains(model) || model.contains(&w.model)
            })
            .collect();

        if let Some(best) = candidates.iter().min_by_key(|w| w.active_requests) {
            return Some((best.tx.clone(), best.worker_id.clone(), best.model.clone(), best.max_context));
        }

        // No matching worker — fall back to any worker with warning
        log::warn!(
            "model={} has no active workers — lenient auto-fallback to any worker",
            model
        );
        let available: Vec<&str> = workers.values().map(|w| w.model.as_str()).collect();
        if !available.is_empty() {
            log::info!("available models for fallback: {}", available.join(", "));
        }

        workers
            .values()
            .min_by_key(|w| w.active_requests)
            .map(|w| (w.tx.clone(), w.worker_id.clone(), w.model.clone(), w.max_context))
    }

    pub fn active_models(&self) -> Vec<String> {
        let workers = self.workers.read().unwrap();
        let mut models: Vec<String> = workers.values().map(|w| w.model.clone()).collect();
        models.sort();
        models.dedup();
        models
    }

    pub fn count(&self) -> usize {
        self.workers.read().unwrap().len()
    }

    /// Serializable snapshot for admin UI.
    pub fn list_all(&self) -> Vec<WorkerStatus> {
        let now = Instant::now();
        self.workers.read().unwrap().values().map(|w| WorkerStatus {
            worker_id:        w.worker_id.clone(),
            model:            w.model.clone(),
            gpu:              w.gpu.clone(),
            vram_free_mb:     w.vram_free_mb,
            max_context:      w.max_context,
            active_requests:  w.active_requests,
            last_heartbeat_s: now.duration_since(w.last_heartbeat).as_secs(),
            connected_s:      now.duration_since(w.connected_at).as_secs(),
            api_type:         w.api_type.clone(),
        }).collect()
    }

    /// Drop workers that haven't heartbeated in >45s.
    pub async fn run_health_check(&self) {
        loop {
            monoio::time::sleep(std::time::Duration::from_secs(10)).await;
            let now = Instant::now();
            let stale: Vec<String> = {
                let workers = self.workers.read().unwrap();
                workers.iter()
                    .filter(|(_, info)| now.duration_since(info.last_heartbeat).as_secs() > 45)
                    .map(|(id, _)| id.clone())
                    .collect()
            };
            if !stale.is_empty() {
                let mut workers = self.workers.write().unwrap();
                for id in stale {
                    if workers.remove(&id).is_some() {
                        log::error!("Worker {} timed out (>45s no heartbeat), deregistered", id);
                    }
                }
            }
        }
    }
}
