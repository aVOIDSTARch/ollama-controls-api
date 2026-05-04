//! Resolve and load `.env` when the process cwd is not the directory that contains the file.

use std::path::{Path, PathBuf};

/// Expands a leading `~/` using `HOME` (or `USERPROFILE` on Windows). Rust does not expand `~` in paths.
fn expand_user_path(raw: &str) -> PathBuf {
    let raw = raw.trim();
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(raw)
}

/// Loads env vars from a dotenv file.
///
/// 1. If `OLLAMA_CONTROLS_DOTENV` is set in the process environment, load that path only.
/// 2. Otherwise try `./.env` from [`std::env::current_dir`], then walk parents (up to 12 levels)
///    looking for `.env` (common when `cargo run` uses the crate directory as cwd).
///
/// Returns the path that was loaded, if any.
pub fn load_dotenv() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OLLAMA_CONTROLS_DOTENV") {
        let path = expand_user_path(&p);
        return match dotenvy::from_filename(&path) {
            Ok(_) => {
                eprintln!("loaded environment from {}", path.display());
                Some(path)
            }
            Err(e) => {
                eprintln!("warning: could not load {}: {e}", path.display());
                None
            }
        };
    }

    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("warning: could not read current directory for .env search: {e}");
            return dotenvy::dotenv().ok();
        }
    };

    for dir in cwd.ancestors().take(12) {
        let candidate = dir.join(".env");
        if candidate.is_file() {
            match dotenvy::from_filename(&candidate) {
                Ok(_) => {
                    eprintln!("loaded environment from {}", candidate.display());
                    return Some(candidate);
                }
                Err(e) => eprintln!(
                    "warning: found {} but could not parse/load: {e}",
                    candidate.display()
                ),
            }
        }
    }

    eprintln!(
        "warning: no `.env` found starting from `{}` (set OLLAMA_CONTROLS_DOTENV to an absolute path)",
        cwd.display()
    );
    None
}

/// Like [`load_dotenv`] but also tries paths next to the executable (useful for `target/release/` runs).
///
/// If `OLLAMA_CONTROLS_DOTENV` is set and loading fails, this does **not** fall back to other paths.
pub fn load_dotenv_with_exe_fallback() -> Option<PathBuf> {
    if std::env::var_os("OLLAMA_CONTROLS_DOTENV").is_some() {
        return load_dotenv();
    }
    if let Some(p) = load_dotenv() {
        return Some(p);
    }

    let Ok(exe) = std::env::current_exe() else {
        return None;
    };
    let Some(mut dir) = exe.parent().map(Path::to_path_buf) else {
        return None;
    };
    for _ in 0..6 {
        let candidate = dir.join(".env");
        if candidate.is_file() {
            if dotenvy::from_filename(&candidate).is_ok() {
                eprintln!("loaded environment from {}", candidate.display());
                return Some(candidate);
            }
        }
        if !dir.pop() {
            break;
        }
    }
    None
}
