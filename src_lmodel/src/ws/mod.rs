use crate::config::Config;
use crate::protocol::Message;
use bytes::BytesMut;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt, Splitable};
use monoio::net::TcpStream;
use monoio_rustls::{TlsConnector, ClientTlsStream};
use rustls::ClientConfig;
use rustls_pki_types::ServerName;
use std::sync::Arc;
use std::time::Duration;
use url::Url;

// ── Exponential backoff ───────────────────────────────────────────────────────

pub struct Backoff {
    pub attempt: u32,
}

impl Backoff {
    pub fn new() -> Self {
        Self { attempt: 0 }
    }

    pub async fn wait(&mut self) {
        let secs = (2u64.pow(self.attempt)).min(30);
        println!("Reconnecting in {}s (attempt {})…", secs, self.attempt + 1);
        self.attempt += 1;
        monoio::time::sleep(Duration::from_secs(secs)).await;
    }

    pub fn reset(&mut self) {
        self.attempt = 0;
    }
}

// ── NDJSON ────────────────────────────────────────────────────────────────────

pub struct NdjsonReader<R> {
    reader: R,
    buf: BytesMut,
}

impl<R: AsyncReadRent> NdjsonReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf: BytesMut::with_capacity(8192),
        }
    }

    pub async fn recv(&mut self) -> Result<Message, String> {
        loop {
            if let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
                let line = self.buf.split_to(pos + 1);
                let text = std::str::from_utf8(&line).map_err(|e| e.to_string())?.trim();
                if text.is_empty() {
                    continue;
                }
                return serde_json::from_str(text).map_err(|e| format!("JSON parse error on '{}': {}", text, e));
            }

            let tmp = vec![0u8; 4096];
            let (res, tmp) = self.reader.read(tmp).await;
            let n = res.map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("Connection closed".into());
            }
            self.buf.extend_from_slice(&tmp[..n]);
        }
    }
}

pub struct NdjsonWriter<W> {
    writer: W,
}

impl<W: AsyncWriteRentExt> NdjsonWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub async fn send_json(&mut self, json_str: &str) -> Result<(), String> {
        let mut data = String::with_capacity(json_str.len() + 1);
        data.push_str(json_str);
        data.push('\n');
        
        let (res, _) = self.writer.write_all(data.into_bytes()).await;
        res.map(|_| ()).map_err(|e| e.to_string())
    }
}

// ── Connect ───────────────────────────────────────────────────────────────────

pub async fn connect(
    cfg: &Config,
) -> Result<(NdjsonReader<monoio::io::OwnedReadHalf<ClientTlsStream<TcpStream>>>, NdjsonWriter<monoio::io::OwnedWriteHalf<ClientTlsStream<TcpStream>>>), String> {
    let secret = option_env!("BRAIN_BOOTSTRAP")
        .map(|s| s.to_string())
        .or_else(|| std::env::var("BRAIN_BOOTSTRAP").ok())
        .ok_or_else(|| {
            "BRAIN_BOOTSTRAP is neither baked into the binary (compile-time) \
             nor set as a runtime environment variable. \
             Build with: BRAIN_BOOTSTRAP=<secret> TARGETS=x86_64 ./static.sh"
                .to_string()
        })?;

    let url = Url::parse(&cfg.brain).map_err(|e| format!("Invalid brain URL: {}", e))?;
    let host = url.host_str().ok_or("URL missing host")?.to_string();
    let port = url.port_or_known_default().ok_or("URL missing port")?;

    let addr_str = format!("{}:{}", host, port);
    let addr = monoio::spawn_blocking(move || {
        use std::net::ToSocketAddrs;
        addr_str
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "No addresses resolved"))
    })
    .await
    .map_err(|e| format!("Spawn error: {:?}", e))?
    .map_err(|e| format!("DNS error: {}", e))?;

    let ws_path = url.path().to_string();
    let ws_path_with_token = match url.query() {
        Some(q) => format!("{}?{}&token={}", ws_path, q, secret),
        None    => format!("{}?token={}", ws_path, secret),
    };

    log::info!("Connecting to brain at {}:{}{}", host, port, ws_path);

    let stream = TcpStream::connect(addr)
        .await
        .map_err(|e| format!("TCP connect error: {:?}", e))?;

    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(
        webpki_roots::TLS_SERVER_ROOTS
            .iter()
            .cloned()
    );
    let mut config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    
    let tls_connector = TlsConnector::from(Arc::new(config));
    let server_name = ServerName::try_from(host.clone())
        .map_err(|_| format!("Invalid DNS name: {}", host))?
        .to_owned();

    let mut tls_stream = tls_connector
        .connect(server_name, stream)
        .await
        .map_err(|e| format!("TLS connect error: {:?}", e))?;

    // Send HTTP Upgrade Request
    let req = format!(
        "GET {} HTTP/1.1\r\n\
         Host: {}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
         Sec-WebSocket-Version: 13\r\n\
         \r\n",
        ws_path_with_token, host
    );

    let (res, _) = tls_stream.write_all(req.into_bytes()).await;
    res.map_err(|e| format!("Write upgrade request failed: {}", e))?;

    // Read HTTP Response
    let mut buf = BytesMut::with_capacity(1024);
    loop {
        let (res, chunk) = tls_stream.read(vec![0u8; 1024]).await;
        let n = res.map_err(|e| format!("Read upgrade response failed: {}", e))?;
        if n == 0 {
            return Err("Connection closed during handshake".into());
        }
        buf.extend_from_slice(&chunk[..n]);

        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let header_str = std::str::from_utf8(&buf[..pos]).unwrap_or("");
            if !header_str.contains("101 Switching Protocols") {
                return Err(format!("Upgrade rejected: {}", header_str.lines().next().unwrap_or("")));
            }
            
            let leftover = buf.split_off(pos + 4);
            let (read_half, write_half) = tls_stream.into_split();
            
            let mut reader = NdjsonReader::new(read_half);
            if !leftover.is_empty() {
                reader.buf.extend_from_slice(&leftover);
            }
            let writer = NdjsonWriter::new(write_half);
            
            return Ok((reader, writer));
        }
    }
}
