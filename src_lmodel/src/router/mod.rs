// ═════════════════════════════════════════════════════════════════════════════
//  lmodel/src/router/mod.rs
//
//  Dispatches an incoming Task to bridge::run_task, and provides a fallback
//  send_error() for failures that happen before the bridge ever gets the task.
//
//  UPDATED: send_error()'s ErrorMessage now sets `code` — required since
//  protocol::ErrorMessage gained that field (see protocol/mod.rs); leaving
//  it out here would fail to compile, not just fail to send.
// ═════════════════════════════════════════════════════════════════════════════

use crate::bridge;
use crate::protocol::{ErrorMessage, Message, TaskMessage};
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::RwLock;

pub type CancelMap = Arc<RwLock<HashMap<String, flume::Sender<()>>>>;

/// Dispatch a task received from the brain.
///
/// All WS writes go through `write_tx` (a flume channel) so that multiple
/// concurrent bridge tasks never race on the same WsWriter.  The main loop
/// drains `write_tx` and calls `writer.send()` exclusively from one place.
pub async fn dispatch(
    task: TaskMessage,
    write_tx: flume::Sender<String>,
    llamacpp_port: u16,
    active_requests: Arc<AtomicU32>,
    cancel_map: CancelMap,
) {
    log::debug!(
        "Dispatching task {} (session={})",
        task.request_id,
        task.session_id
    );

    let (cancel_tx, cancel_rx) = flume::bounded::<()>(1);
    cancel_map.write().unwrap().insert(task.request_id.clone(), cancel_tx);

    let req_id = task.request_id.clone();
    let cmap = cancel_map.clone();

    // All inference goes through bridge::run_task.
    // We spawn so the recv loop stays unblocked.
    monoio::spawn(async move {
        bridge::run_task(task, write_tx, llamacpp_port, active_requests, cancel_rx).await;
        cmap.write().unwrap().remove(&req_id);
    });
}

pub fn cancel(request_id: &str, cancel_map: &CancelMap) {
    if let Some(tx) = cancel_map.read().unwrap().get(request_id) {
        let _ = tx.send(());
    }
}

/// Helper: send an immediate error back over the write channel without going
/// through the bridge (e.g. when dispatch itself fails).
pub async fn send_error(write_tx: &flume::Sender<String>, request_id: &str, message: &str) {
    let msg = Message::Error(ErrorMessage {
        request_id: request_id.to_string(),
        message:    message.to_string(),
        code:       502,
    });
    if let Ok(json) = serde_json::to_string(&msg) {
        let _ = write_tx.send_async(json).await;
    }
}