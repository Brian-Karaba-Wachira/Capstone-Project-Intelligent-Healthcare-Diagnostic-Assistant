use std::sync::Arc;
use std::time::{Duration, Instant};
use flume::unbounded;
use bytes::BytesMut;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt, Splitable};
use monoio::net::TcpStream;

use crate::core::config::Config;
use crate::core::protocol::{DoneMessage, FinishReason, WorkerMessage};
use crate::worker::registry::{WorkerInfo, WorkerRegistry};
use crate::api::router::Router;
use crate::db::memory::Database;
use crate::metrics::Metrics;

/// Brain proactively pings the worker on this interval (seconds).
/// Must be shorter than any proxy/NAT idle timeout (nginx default is 60s).
const PING_INTERVAL_S: u64 = 25;

/// If no message arrives from the worker for this long, the connection is
/// considered dead and the tunnel is torn down so brain cleans up resources
/// instead of holding a zombie connection that serves nobody.
const IDLE_TIMEOUT_S: u64 = 90;

// ── NDJSON framing ────────────────────────────────────────────────────────────

pub struct NdjsonReader<R> {
    reader: R,
    buf:    BytesMut,
}

impl<R: AsyncReadRent> NdjsonReader<R> {
    pub fn new(reader: R, leftover: bytes::Bytes) -> Self {
        let mut buf = BytesMut::with_capacity(8192);
        if !leftover.is_empty() {
            buf.extend_from_slice(&leftover);
        }
        Self { reader, buf }
    }

    pub async fn recv(&mut self) -> Result<WorkerMessage, String> {
        loop {
            if let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
                let line = self.buf.split_to(pos + 1);
                let text = std::str::from_utf8(&line)
                    .map_err(|e| e.to_string())?
                    .trim();
                if text.is_empty() { continue; }
                return serde_json::from_str(text).map_err(|e| e.to_string());
            }

            let tmp = vec![0u8; 4096];
            let (res, tmp) = self.reader.read(tmp).await;
            let n = res.map_err(|e| e.to_string())?;
            if n == 0 { return Err("Connection closed".into()); }
            self.buf.extend_from_slice(&tmp[..n]);
        }
    }
}

pub struct NdjsonWriter<W> {
    writer: W,
}

impl<W: AsyncWriteRentExt> NdjsonWriter<W> {
    pub fn new(writer: W) -> Self { Self { writer } }

    pub async fn send_json(&mut self, json_str: &str) -> Result<(), String> {
        let mut data = String::with_capacity(json_str.len() + 1);
        data.push_str(json_str);
        data.push('\n');
        let (res, _) = self.writer.write_all(data.into_bytes()).await;
        res.map(|_| ()).map_err(|e| e.to_string())
    }
}

// ── Tunnel handler ────────────────────────────────────────────────────────────

pub async fn handle_worker(
    stream:   TcpStream,
    leftover: bytes::Bytes,
    registry: Arc<WorkerRegistry>,
    router:   Arc<Router>,
    db:       Arc<Database>,
    _cfg:     Arc<Config>,
    metrics:  Arc<Metrics>,
) {
    let (read_half, write_half) = stream.into_split();
    let mut reader = NdjsonReader::new(read_half, leftover);
    let mut writer = NdjsonWriter::new(write_half);

    // ── Step 1: Register ───────────────────────────────────────────────────
    let (worker_id, rx) = match reader.recv().await {
        Ok(WorkerMessage::Register(reg)) => {
            log::info!(
                "Worker {} registered (model={}, gpu={}, vram={}MB, ctx={}, api_type={})",
                reg.worker_id, reg.model, reg.gpu, reg.vram_free_mb, reg.max_context, reg.api_type
            );
            let (tx, rx) = unbounded::<WorkerMessage>();
            let info = WorkerInfo {
                worker_id:       reg.worker_id.clone(),
                model:           reg.model.clone(),
                gpu:             reg.gpu.clone(),
                vram_free_mb:    reg.vram_free_mb,
                max_context:     reg.max_context,
                active_requests: reg.active_requests,
                last_heartbeat:  Instant::now(),
                connected_at:    Instant::now(),
                tx,
                api_type:        reg.api_type.clone(),
            };
            registry.register(info);
            db.upsert_worker(&reg.worker_id, &reg.model, &reg.gpu);
            (reg.worker_id, rx)
        }
        Ok(other) => {
            log::warn!("First message was not Register, got: {:?}", other);
            return;
        }
        Err(e) => {
            log::warn!("Failed to parse first worker message: {}", e);
            return;
        }
    };

    // ── Step 2: Main loop ─────────────────────────────────────────────────────
    //
    // Two stability mechanisms:
    //
    //  1. Proactive pings — brain sends Ping every PING_INTERVAL_S seconds.
    //     Without this, a reverse proxy (nginx default: 60s) or NAT sees an
    //     idle connection and silently drops it mid-session. The worker
    //     reconnects but any in-flight tasks lose their result, making users
    //     experience sluggish or broken sessions.
    //
    //  2. Idle timeout — if no message arrives from the worker for
    //     IDLE_TIMEOUT_S seconds, brain tears down the tunnel immediately
    //     rather than holding a zombie connection that serves nobody and
    //     prevents new registrations from the same worker.
    //
    let mut last_rx      = Instant::now();
    let mut next_ping_at = Instant::now() + Duration::from_secs(PING_INTERVAL_S);

    loop {
        let now = Instant::now();

        // ── Idle timeout check ────────────────────────────────────────────
        if now.duration_since(last_rx).as_secs() > IDLE_TIMEOUT_S {
            log::warn!(
                "Worker {} idle for >{}s — closing tunnel (likely dead connection)",
                worker_id, IDLE_TIMEOUT_S
            );
            break;
        }

        // ── Proactive ping ────────────────────────────────────────────────
        if now >= next_ping_at {
            log::debug!("Sending keepalive Ping to worker {}", worker_id);
            if let Ok(json) = serde_json::to_string(&WorkerMessage::Ping) {
                if writer.send_json(&json).await.is_err() {
                    log::warn!("Keepalive Ping write failed for worker {} — closing tunnel", worker_id);
                    break;
                }
            }
            next_ping_at = Instant::now() + Duration::from_secs(PING_INTERVAL_S);
        }

        // Sleep until the next ping is due, capped at 5s so the idle-timeout
        // check above fires frequently enough to be useful.
        let sleep_dur = next_ping_at
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs(5));

        monoio::select! {
            msg_res = reader.recv() => {
                last_rx = Instant::now(); // any inbound message resets the idle clock
                match msg_res {
                    Ok(msg) => match msg {

                        WorkerMessage::Heartbeat(hb) => {
                            registry.update_load(&worker_id, hb.active_requests, hb.vram_free_mb);
                            registry.update_model(&worker_id, &hb.model);
                            db.touch_worker(&worker_id);
                            metrics.set_queue_depth(&worker_id, hb.active_requests as u64);
                            metrics.set_worker_health(&worker_id, true);
                            log::debug!("Heartbeat {} (model={}, active={}, vram={}MB)",
                                worker_id, hb.model, hb.active_requests, hb.vram_free_mb);
                            // Acknowledge so the worker also resets its idle timer.
                            if let Ok(json) = serde_json::to_string(&WorkerMessage::Ping) {
                                let _ = writer.send_json(&json).await;
                            }
                        }

                        WorkerMessage::Chunk(chunk) => {
                            let rid = chunk.request_id.clone();
                            router.forward_chunk(&rid, chunk).await;
                        }

                        WorkerMessage::Done(done) => {
                            log::debug!("Done: request={} finish={:?}",
                                done.request_id, done.finish_reason);
                            registry.decrement_active(&worker_id);
                            metrics.set_queue_depth(&worker_id,
                                registry.list_all().iter()
                                    .find(|w| w.worker_id == worker_id)
                                    .map(|w| w.active_requests as u64)
                                    .unwrap_or(0));
                            let rid = done.request_id.clone();
                            router.forward_done(&rid, done).await;
                            router.finish(&rid);
                        }

                        WorkerMessage::Error(err) => {
                            log::warn!("Worker error for {}: {}", err.request_id, err.message);
                            registry.decrement_active(&worker_id);
                            let synthetic_done = DoneMessage {
                                request_id:    err.request_id.clone(),
                                finish_reason: FinishReason::Error,
                                prompt_tokens: 0,
                                comp_tokens:   0,
                            };
                            router.forward_done(&err.request_id, synthetic_done).await;
                            router.finish(&err.request_id);
                        }

                        WorkerMessage::Register(_) => {
                            log::warn!("Unexpected re-Register from {}", worker_id);
                        }
                        WorkerMessage::Task(_) | WorkerMessage::Ping | WorkerMessage::Cancel(_) => {
                            // Workers shouldn't send these — but silently
                            // swallowing them means a protocol drift between
                            // brain and a worker build would never surface
                            // anywhere. Log it so it's visible instead.
                            log::warn!(
                                "[unhandled-translation] Worker {} sent a message type brain doesn't expect from a worker: {:?}",
                                worker_id, msg
                            );
                        }
                    },

                    Err(e) => {
                        log::warn!("Read error from worker {}: {}", worker_id, e);
                        break;
                    }
                }
            },

            msg_out_res = rx.recv_async() => {
                match msg_out_res {
                    Ok(msg_out) => {
                        if let WorkerMessage::Task(ref t) = msg_out {
                            log::debug!("→ worker {} task={} model={} tools={}",
                                worker_id, t.request_id, t.model,
                                t.tools.as_ref().map(|x| x.len()).unwrap_or(0));
                            registry.increment_active(&worker_id);
                        } else if let WorkerMessage::Cancel(ref c) = msg_out {
                            log::debug!("→ worker {} cancel={}", worker_id, c.request_id);
                        }

                        match serde_json::to_string(&msg_out) {
                            Ok(json) => {
                                if writer.send_json(&json).await.is_err() {
                                    log::warn!("Write failed to worker {}", worker_id);
                                    if matches!(msg_out, WorkerMessage::Task(_)) {
                                        registry.decrement_active(&worker_id);
                                    }
                                    break;
                                }
                            }
                            Err(e) => {
                                log::error!("Serialize message failed: {}", e);
                                if matches!(msg_out, WorkerMessage::Task(_)) {
                                    registry.decrement_active(&worker_id);
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            },

            _ = monoio::time::sleep(sleep_dur) => {
                // Timer fired — loop back to evaluate ping schedule and idle timeout.
            }
        }
    }

    log::info!("Tunnel closed for worker {}", worker_id);
    registry.deregister(&worker_id);
    db.remove_worker(&worker_id);
    // Any request still waiting on this worker will never see another
    // Chunk/Done — fail it now rather than making the client's streaming/
    // blocking handler sit out the full per-request idle timeout.
    router.fail_all_for_worker(&worker_id).await;
}