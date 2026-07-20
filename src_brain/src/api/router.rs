use std::collections::HashMap;
use std::sync::RwLock;
use flume::Sender;
use crate::core::protocol::{ChunkMessage, DoneMessage, FinishReason};

pub struct PendingRequest {
    pub chunk_tx:  Sender<ChunkMessage>,
    pub done_tx:   Sender<DoneMessage>,
    pub worker_tx: Sender<crate::core::protocol::WorkerMessage>,
    /// Which worker/tunnel this request was dispatched to. Lets
    /// `fail_all_for_worker` find every request stranded by a specific
    /// tunnel dying, without needing `Sender` to be comparable.
    pub worker_id: String,
}

pub struct Router {
    pending: RwLock<HashMap<String, PendingRequest>>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
        }
    }

    pub fn register_pending(
        &self,
        id:       String,
        chunk_tx:  Sender<ChunkMessage>,
        done_tx:   Sender<DoneMessage>,
        worker_tx: Sender<crate::core::protocol::WorkerMessage>,
        worker_id: String,
    ) {
        self.pending.write().unwrap().insert(id, PendingRequest { chunk_tx, done_tx, worker_tx, worker_id });
    }

    /// Called by the tunnel handler the moment a worker's connection dies
    /// (read error, write error, or idle timeout — see worker/tunnel.rs).
    ///
    /// Previously nothing told the router that a worker's in-flight
    /// requests would never receive a Done/Chunk again, so each request's
    /// streaming/blocking handler just sat on its own `IDLE_TIMEOUT`
    /// (default 120s, see BRAIN_WORKER_TIMEOUT) before finally giving up.
    /// From the client's perspective a mid-generation reconnect (which
    /// happens routinely — proxy hiccup, brain restart, etc.) looked like a
    /// silent multi-minute hang instead of a fast, clear error. This sends
    /// a synthetic Done(Error) to every request still pointing at the dead
    /// worker so its handler wakes up immediately instead of idling out.
    pub async fn fail_all_for_worker(&self, worker_id: &str) {
        let stranded: Vec<(String, Sender<DoneMessage>)> = {
            let pending = self.pending.read().unwrap();
            pending.iter()
                .filter(|(_, req)| req.worker_id == worker_id)
                .map(|(id, req)| (id.clone(), req.done_tx.clone()))
                .collect()
        };

        if stranded.is_empty() {
            return;
        }

        log::warn!(
            "Tunnel for worker {} died with {} request(s) in flight — failing them immediately",
            worker_id, stranded.len()
        );

        for (id, done_tx) in stranded {
            let synthetic_done = DoneMessage {
                request_id:    id.clone(),
                finish_reason: FinishReason::Error,
                prompt_tokens: 0,
                comp_tokens:   0,
            };
            let _ = done_tx.send_async(synthetic_done).await;
            self.pending.write().unwrap().remove(&id);
        }
    }

    pub async fn forward_chunk(&self, id: &str, chunk: ChunkMessage) {
        let tx = self.pending.read().unwrap().get(id).map(|req| req.chunk_tx.clone());
        if let Some(tx) = tx {
            let _ = tx.send_async(chunk).await;
        }
    }

    /// See `forward_chunk` — same fix, same reasoning (`done_tx` is
    /// bounded(1), so this was just as capable of stalling the reactor
    /// thread if the receiver hadn't picked up the previous Done yet).
    pub async fn forward_done(&self, id: &str, done: DoneMessage) {
        let tx = self.pending.read().unwrap().get(id).map(|req| req.done_tx.clone());
        if let Some(tx) = tx {
            let _ = tx.send_async(done).await;
        }
    }

    /// Called after done/error — removes the request from the map.
    pub fn finish(&self, id: &str) {
        self.pending.write().unwrap().remove(id);
    }

    /// Called to abruptly cancel a pending request (e.g. client disconnect).
    /// Sends a Cancel message to the worker before removing the request.
    pub fn cancel(&self, id: &str) {
        if let Some(req) = self.pending.write().unwrap().remove(id) {
            let _ = req.worker_tx.send(crate::core::protocol::WorkerMessage::Cancel(
                crate::core::protocol::CancelMessage { request_id: id.to_string() }
            ));
        }
    }
}
