pub mod bridge;
pub mod config;
pub mod protocol;
pub mod registry;
pub mod router;
pub mod ws;

use std::rc::Rc;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::collections::HashMap;
use std::sync::RwLock;

// Removed unbounded import
use protocol::Message;

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    monoio::RuntimeBuilder::<monoio::FusionDriver>::new()
        .enable_timer()
        .with_blocking_strategy(monoio::blocking::BlockingStrategy::ExecuteLocal)
        .build()
        .expect("Failed to build monoio runtime")
        .block_on(run())
}

async fn run() {
    let cfg = config::Config::from_args();
    log::info!("lmodel starting");
    log::info!("  worker_id   : {}", cfg.worker_id);
    log::info!("  brain       : {}", cfg.brain);
    log::info!("  model       : {}", cfg.model);
    log::info!("  llamacpp    : :{}", cfg.llamacpp_port);

    let active_requests: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));
    let cancel_map: router::CancelMap = Arc::new(RwLock::new(HashMap::new()));
    let mut backoff = ws::Backoff::new();

    loop {
        match ws::connect(&cfg).await {
            Ok((mut reader, writer)) => {
                log::info!("Connected to brain ✓");
                backoff.reset();

                let (write_tx, write_rx) = flume::bounded::<String>(256);
                let shared_writer = Rc::new(RefCell::new(writer));

                registry::wait_for_llamacpp(cfg.llamacpp_port).await;

                // Ask llama.cpp what it actually loaded with, rather than
                // trusting the --max-context CLI flag (see registry.rs).
                let real_ctx = registry::query_real_context(cfg.llamacpp_port).await;
                let reg_msg = Message::Register(registry::build_register_msg(&cfg, real_ctx));
                match serde_json::to_string(&reg_msg) {
                    Ok(json) => {
                        if shared_writer.borrow_mut().send_json(&json).await.is_err() {
                            log::error!("Failed to send Register — reconnecting");
                            continue;
                        }
                        log::info!("Registered as worker {}", cfg.worker_id);
                    }
                    Err(e) => {
                        log::error!("Failed to serialize Register: {}", e);
                        continue;
                    }
                }

                let (conn_dead_tx, conn_dead_rx) = flume::bounded::<()>(1);

                {
                    let write_tx_hb  = write_tx.clone();
                    let worker_id    = cfg.worker_id.clone();
                    let model_name   = cfg.model.clone();
                    let active_clone = active_requests.clone();
                    let dead_tx      = conn_dead_tx.clone();
                    monoio::spawn(async move {
                        use std::time::Duration;
                        loop {
                            monoio::time::sleep(Duration::from_secs(15)).await;
                            let active = active_clone.load(Ordering::Relaxed);
                            let msg = Message::Heartbeat(protocol::HeartbeatMessage {
                                worker_id:       worker_id.clone(),
                                model:           model_name.clone(),
                                vram_free_mb:    1024,
                                active_requests: active,
                            });
                            match serde_json::to_string(&msg) {
                                Ok(json) => {
                                    if write_tx_hb.send_async(json).await.is_err() {
                                        log::warn!("Heartbeat send failed — write path to brain is dead, forcing reconnect");
                                        let _ = dead_tx.try_send(());
                                        break;
                                    }
                                }
                                Err(e) => log::error!("Heartbeat serialize error: {}", e),
                            }
                        }
                    });
                }

                {
                    let writer_rc  = shared_writer.clone();
                    let write_rx_w = write_rx;
                    let dead_tx    = conn_dead_tx.clone();
                    monoio::spawn(async move {
                        while let Ok(json) = write_rx_w.recv_async().await {
                            if writer_rc.borrow_mut().send_json(&json).await.is_err() {
                                // Previously this just ended the forwarding
                                // task silently. Everything that still holds
                                // a `write_tx` clone (bridge tasks streaming
                                // llama.cpp output, the heartbeat task above)
                                // would start getting "channel closed" the
                                // next time they tried to send — but the
                                // *read* loop below had no idea any of this
                                // happened, so it kept accepting new Task
                                // assignments from brain and dispatching them
                                // to llama.cpp, which would generate a full
                                // response nobody could ever deliver: from
                                // the client's point of view the request just
                                // hangs, while llama.cpp logs show it working
                                // the whole time. Signal the read loop so the
                                // whole connection tears down and reconnects
                                // together instead of leaving read and write
                                // out of sync.
                                log::warn!("WsWriter send failed — connection closed, forcing reconnect");
                                let _ = dead_tx.try_send(());
                                break;
                            }
                        }
                    });
                }

                let llamacpp_port    = cfg.llamacpp_port;
                let active_for_recv  = active_requests.clone();

                loop {
                    monoio::select! {
                        msg_res = reader.recv() => {
                            match msg_res {
                                Ok(Message::Task(task)) => {
                                    router::dispatch(
                                        task,
                                        write_tx.clone(),
                                        llamacpp_port,
                                        active_for_recv.clone(),
                                        cancel_map.clone(),
                                    ).await;
                                }
                                Ok(Message::Cancel(cancel)) => {
                                    log::info!("Received Cancel for request {}", cancel.request_id);
                                    router::cancel(&cancel.request_id, &cancel_map);
                                }
                                Ok(Message::Ping) => {
                                    // Keep-alive received from brain, do nothing
                                }
                                Ok(other) => {
                                    // This means brain sent a message shape this
                                    // worker build doesn't know how to handle —
                                    // a protocol mismatch, not a normal event.
                                    // Was previously debug-level (invisible under
                                    // the default "info" filter this binary
                                    // actually runs with), so a live protocol
                                    // drift between brain and lmodel could go
                                    // completely unnoticed. Promote to a warning.
                                    log::warn!("[unhandled-translation] Received message type from brain this worker doesn't act on: {:?}", other);
                                }
                                Err(e) => {
                                    log::error!("WS read error: {:?}", e);
                                    break;
                                }
                            }
                        },
                        _ = conn_dead_rx.recv_async() => {
                            log::warn!("Write path to brain died — tearing down connection to force a clean reconnect");
                            break;
                        }
                    }
                }

                log::warn!("Connection to brain lost — will reconnect");
            }

            Err(e) => {
                log::error!("Failed to connect to brain: {}", e);
            }
        }

        backoff.wait().await;
    }
}
