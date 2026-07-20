use rust_embed::RustEmbed;
use std::borrow::Cow;

#[derive(RustEmbed)]
#[folder = "ui/dist"]
pub struct Asset;

pub fn get_asset(path: &str) -> Option<(Cow<'static, [u8]>, String)> {
    let mut file_path = path;
    if file_path.is_empty() || file_path == "/" {
        file_path = "index.html";
    } else {
        file_path = file_path.strip_prefix("/").unwrap_or(file_path);
    }

    // Try finding the asset
    match Asset::get(file_path) {
        Some(content) => {
            let mime_type = mime_guess::from_path(file_path).first_or_octet_stream().to_string();
            Some((content.data, mime_type))
        }
        None => {
            // For SPA routing, fallback to index.html if not found and doesn't have an extension
            if !file_path.contains('.') {
                if let Some(index) = Asset::get("index.html") {
                    return Some((index.data, "text/html".to_string()));
                }
            }
            None
        }
    }
}
