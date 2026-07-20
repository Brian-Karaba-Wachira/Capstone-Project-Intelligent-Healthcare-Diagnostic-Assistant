use crate::config::Config;
use crate::protocol::RegisterMessage;
use std::time::Duration;

/// Poll llama.cpp's /v1/models endpoint until it returns 200.
/// We use /v1/models because llama-cpp-python does NOT expose /health —
/// that endpoint returns 404.
pub async fn wait_for_llamacpp(port: u16) {
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    loop {
        let url_clone = url.clone();
        let ok = monoio::spawn_blocking(move || {
            match ureq::get(&url_clone)
                .timeout(Duration::from_secs(3))
                .call()
            {
                Ok(resp) => resp.status() == 200,
                Err(_) => false,
            }
        })
        .await
        .unwrap_or(false);

        if ok {
            println!("✅ llama.cpp is ready on :{}", port);
            break;
        }
        println!("⏳ Waiting for llama.cpp on :{}…", port);
        monoio::time::sleep(Duration::from_secs(5)).await;
    }
}

/// Query llama.cpp's own `/props` endpoint for the REAL per-slot context
/// size (`default_generation_settings.n_ctx`) and slot count (`total_slots`).
///
/// FIX (was BUG): `cfg.max_context` is just an operator-supplied CLI flag
/// (`--max-context`, default 16384) with no connection whatsoever to the
/// `-c` value actually passed to `llama-server`. Nothing enforced that they
/// stay in sync — change `-c` on the llama-server launch line and forget to
/// also update `--max-context` here, and Brain keeps truncating every
/// conversation to whatever stale number this worker announced at
/// registration. `/props` is the server's own runtime truth: it reports
/// what it actually loaded with, not what an operator typed into a flag six
/// deploys ago.
///
/// Returns `None` if `/props` is unreachable or doesn't have the expected
/// shape (e.g. an older llama.cpp build) — caller falls back to
/// `cfg.max_context` in that case, which is still better than hanging.
pub async fn query_real_context(port: u16) -> Option<u32> {
    let url = format!("http://127.0.0.1:{}/props", port);
    let body = monoio::spawn_blocking(move || {
        ureq::get(&url)
            .timeout(Duration::from_secs(5))
            .call()
            .ok()?
            .into_string()
            .ok()
    })
    .await
    .ok()
    .flatten()?;

    let val: serde_json::Value = serde_json::from_str(&body).ok()?;
    let n_ctx = val.get("default_generation_settings")?.get("n_ctx")?.as_u64()? as u32;
    let total_slots = val.get("total_slots").and_then(|v| v.as_u64()).unwrap_or(1);

    if n_ctx == 0 {
        return None;
    }

    log::info!(
        "llama.cpp /props: n_ctx={} per slot, total_slots={} (={} total -c)",
        n_ctx, total_slots, n_ctx as u64 * total_slots
    );
    Some(n_ctx)
}

pub fn build_register_msg(cfg: &Config, real_max_context: Option<u32>) -> RegisterMessage {
    let max_context = real_max_context.unwrap_or_else(|| {
        log::warn!(
            "Could not query llama.cpp /props for real context size — \
             falling back to --max-context={} (verify this actually matches \
             the -c you launched llama-server with, divided by -np if >1)",
            cfg.max_context
        );
        cfg.max_context
    });

    RegisterMessage {
        worker_id:       cfg.worker_id.clone(),
        model:           cfg.model.clone(),
        gpu:             cfg.gpu.clone(),
        vram_free_mb:    cfg.vram_free_mb,
        max_context,
        active_requests: 0,
    }
}
