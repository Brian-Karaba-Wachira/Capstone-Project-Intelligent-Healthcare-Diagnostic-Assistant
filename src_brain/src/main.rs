pub mod api;
pub mod core;
pub mod db;
pub mod net;
pub mod worker;
pub mod metrics;

pub static START_TIME: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

use std::sync::Arc;
use monoio::net::TcpListener;

use crate::core::config::Config;
use crate::core::idempotency::IdempotencyStore;
use crate::db::memory::Database;
use crate::worker::registry::WorkerRegistry;
use crate::api::router::Router;
use crate::metrics::Metrics;

async fn run(
    cfg: Arc<Config>,
    registry: Arc<WorkerRegistry>,
    db: Arc<Database>,
    router: Arc<Router>,
    metrics: Arc<Metrics>,
    idempotency: Arc<IdempotencyStore>,
    
    
    is_primary: bool,
) {
    if is_primary {
        let reg = registry.clone();
        monoio::spawn(async move {
            reg.run_health_check().await;
        });

        // Periodic cache cleanup task
                let idm = idempotency.clone();
        monoio::spawn(async move {
            loop {
                monoio::time::sleep(std::time::Duration::from_secs(60)).await;
                                idm.cleanup();
            }
        });


    }

    let listener = match TcpListener::bind(&cfg.addr) {
        Ok(l) => l,
        Err(e) => {
            log::error!("Failed to bind {}: {}", cfg.addr, e);
            return;
        }
    };

    if is_primary {
        log::info!("Brain listening on {}", cfg.addr);
    }

    loop {
        match listener.accept().await {
            Ok((stream, peer)) => {
                log::debug!("Connection from {}", peer);
                let registry = registry.clone();
                let router   = router.clone();
                let db       = db.clone();
                let cfg      = cfg.clone();
                let metrics  = metrics.clone();
                let idempotency = idempotency.clone();
                                                monoio::spawn(async move {
                    crate::net::acceptor::handle_conn(
                        stream, registry, router, db, cfg,
                        metrics, idempotency).await;
                });
            }
            Err(e) => log::error!("Accept error: {}", e),
        }
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    START_TIME.get_or_init(std::time::Instant::now);

    let cfg = Arc::new(Config::from_env());

    log::info!("╔══════════════════════════════════════════════╗");
    log::info!("║         Colab Brain starting up              ║");
    log::info!("╠══════════════════════════════════════════════╣");
    log::info!("║  addr    : {}                      ║", cfg.addr);
    log::info!("║  db      : {}                ║", cfg.db_path);
    log::info!("║  dl      : {}        ║", cfg.downloads_dir);
    log::info!("║  psk     : {}  ║", cfg.psk);
    log::info!("║  egress  : {:?}          ║", cfg.egress_policy);
    log::info!("║  metrics : {}              ║", cfg.metrics_enabled);
    log::info!("╚══════════════════════════════════════════════╝");

    ctrlc::set_handler(|| {
        log::info!("Shutdown signal received — exiting");
        std::process::exit(0);
    })
    .expect("Failed to set Ctrl-C handler");

    // ── Create working directories ───────────────────────────────────────────
    for dir in &[
        crate::core::config::WORK_DIR,
        &cfg.downloads_dir,
    ] {
        match std::fs::create_dir_all(dir) {
            Ok(_)  => log::info!("Working dir: {}", dir),
            Err(e) => log::warn!("Could not create {}: {} — may need root/sudo", dir, e),
        }
    }

    let registry     = Arc::new(WorkerRegistry::new());
    let router       = Arc::new(Router::new());
    let metrics      = Arc::new(Metrics::new());
    let idempotency  = Arc::new(IdempotencyStore::new(500, cfg.idempotency_ttl_s));
        
    let db_exists = std::path::Path::new(&cfg.db_path).exists();
    if db_exists {
        log::info!("Opening existing database: {}", cfg.db_path);
    } else {
        log::info!("Creating new database: {}", cfg.db_path);
    }
    let db = match Database::open(&cfg.db_path) {
        Ok(db) => Arc::new(db),
        Err(e) => {
            log::error!("Failed to open database at {}: {}", cfg.db_path, e);
            std::process::exit(1);
        }
    };

    let mut handles = Vec::new();
    // Two threads sharing the same listener port via SO_REUSEPORT (monoio default on Linux)
    for i in 0..2usize {
        let cfg      = cfg.clone();
        let registry = registry.clone();
        let router   = router.clone();
        let db       = db.clone();
        let metrics  = metrics.clone();
        let idempotency = idempotency.clone();
                
        handles.push(std::thread::spawn(move || {
            let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
                .enable_timer()
                .build()
                .unwrap();
            rt.block_on(run(
                cfg, registry, db, router, metrics,
                idempotency, i == 0,
            ));
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}
