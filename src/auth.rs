//! API key material for gateway authentication: strong random keys, Argon2id hashes, and an
//! age-encrypted local keystore (passphrase-protected at rest).
//!
//! The HTTP gateway loads the keystore at startup when API key auth is enabled (see
//! `OLLAMA_CONTROLS_REQUIRE_API_KEY` / `OLLAMA_CONTROLS_KEYSTORE_PASSPHRASE` in the server docs).
//! Use [`add_user_api_key`] from the `ollama-api` CLI to append keys to the encrypted file.

use age::secrecy::{ExposeSecret, SecretString};
use argon2::password_hash::rand_core::{OsRng, RngCore};
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

const KEYSTORE_VERSION: u32 = 1;
const RANDOM_KEY_LEN: usize = 32;

#[derive(Debug)]
pub enum KeyStoreError {
    Io(std::io::Error),
    AgeEncrypt(age::EncryptError),
    AgeDecrypt(age::DecryptError),
    Json(serde_json::Error),
    InvalidUsername(String),
    DuplicateUsername(String),
    UnsupportedKeystoreVersion(u32),
    PasswordHash(argon2::password_hash::Error),
    EmptyPassphrase,
}

impl fmt::Display for KeyStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyStoreError::Io(e) => write!(f, "{e}"),
            KeyStoreError::AgeEncrypt(e) => write!(f, "keystore encryption failed: {e}"),
            KeyStoreError::AgeDecrypt(e) => write!(f, "keystore decryption failed: {e}"),
            KeyStoreError::Json(e) => write!(f, "keystore JSON error: {e}"),
            KeyStoreError::InvalidUsername(s) => write!(f, "invalid username: {s}"),
            KeyStoreError::DuplicateUsername(u) => {
                write!(f, "username `{u}` already has a key in this keystore")
            }
            KeyStoreError::UnsupportedKeystoreVersion(v) => {
                write!(f, "unsupported keystore version {v} (expected {KEYSTORE_VERSION})")
            }
            KeyStoreError::PasswordHash(e) => write!(f, "password hash error: {e}"),
            KeyStoreError::EmptyPassphrase => write!(f, "keystore passphrase must not be empty"),
        }
    }
}

impl std::error::Error for KeyStoreError {}

impl From<std::io::Error> for KeyStoreError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for KeyStoreError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

/// Default path: `$XDG_CONFIG_HOME/ollama-controls/api-keys.age`, or `~/.config/ollama-controls/api-keys.age`.
pub fn default_keystore_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ollama-controls")
        .join("api-keys.age")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyStoreData {
    pub version: u32,
    pub entries: Vec<KeyEntry>,
}

/// Set on the request by the API-key middleware after a successful lookup.
#[derive(Clone, Debug)]
pub struct AuthenticatedUser {
    pub username: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyEntry {
    pub username: String,
    /// Argon2 PHC string of the API key (never store the raw key).
    pub key_hash: String,
}

impl KeyStoreData {
    pub fn empty() -> Self {
        Self {
            version: KEYSTORE_VERSION,
            entries: Vec::new(),
        }
    }
}

/// Returns a high-entropy URL-safe key (32 random bytes, base64url without padding).
pub fn generate_strong_api_key() -> String {
    let mut buf = [0u8; RANDOM_KEY_LEN];
    OsRng.fill_bytes(&mut buf);
    URL_SAFE_NO_PAD.encode(buf)
}

pub fn hash_api_key(secret: &str) -> Result<String, KeyStoreError> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2
        .hash_password(secret.as_bytes(), &salt)
        .map_err(KeyStoreError::PasswordHash)?;
    Ok(hash.to_string())
}

pub fn verify_api_key(secret: &str, phc_hash: &str) -> Result<bool, KeyStoreError> {
    let parsed = PasswordHash::new(phc_hash).map_err(KeyStoreError::PasswordHash)?;
    Ok(Argon2::default()
        .verify_password(secret.as_bytes(), &parsed)
        .is_ok())
}

fn encrypt_bytes(passphrase: &SecretString, plaintext: &[u8]) -> Result<Vec<u8>, KeyStoreError> {
    let recipient = age::scrypt::Recipient::new(passphrase.clone());
    age::encrypt(&recipient, plaintext).map_err(KeyStoreError::AgeEncrypt)
}

fn decrypt_bytes(passphrase: &SecretString, ciphertext: &[u8]) -> Result<Vec<u8>, KeyStoreError> {
    let identity = age::scrypt::Identity::new(passphrase.clone());
    age::decrypt(&identity, ciphertext).map_err(KeyStoreError::AgeDecrypt)
}

pub fn load_keystore(path: &Path, passphrase: &SecretString) -> Result<KeyStoreData, KeyStoreError> {
    if passphrase.expose_secret().is_empty() {
        return Err(KeyStoreError::EmptyPassphrase);
    }
    if !path.exists() {
        return Ok(KeyStoreData::empty());
    }
    let ciphertext = fs::read(path)?;
    let plaintext = decrypt_bytes(passphrase, &ciphertext)?;
    let data: KeyStoreData = serde_json::from_slice(&plaintext)?;
    if data.version != KEYSTORE_VERSION {
        return Err(KeyStoreError::UnsupportedKeystoreVersion(data.version));
    }
    Ok(data)
}

pub fn save_keystore(
    path: &Path,
    passphrase: &SecretString,
    data: &KeyStoreData,
) -> Result<(), KeyStoreError> {
    if passphrase.expose_secret().is_empty() {
        return Err(KeyStoreError::EmptyPassphrase);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let plaintext = serde_json::to_vec(data)?;
    let ciphertext = encrypt_bytes(passphrase, &plaintext)?;
    fs::write(path, ciphertext)?;
    Ok(())
}

/// Normalizes and validates a username label for the keystore.
pub fn normalize_username(raw: &str) -> Result<String, KeyStoreError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(KeyStoreError::InvalidUsername(
            "username must not be empty".into(),
        ));
    }
    if trimmed.len() > 128 {
        return Err(KeyStoreError::InvalidUsername(
            "username must be at most 128 characters".into(),
        ));
    }
    let ok = trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-');
    if !ok {
        return Err(KeyStoreError::InvalidUsername(
            "username may only contain ASCII letters, digits, '.', '_', and '-'".into(),
        ));
    }
    Ok(trimmed.to_string())
}

/// Looks up the stored Argon2 PHC string for a normalized username.
pub fn key_hash_for_username<'a>(data: &'a KeyStoreData, username: &str) -> Option<&'a str> {
    let username = normalize_username(username).ok()?;
    data.entries
        .iter()
        .find(|e| e.username == username)
        .map(|e| e.key_hash.as_str())
}

/// Returns the keystore username whose Argon2 hash matches `secret`, if any.
pub fn resolve_username_for_api_key(
    data: &KeyStoreData,
    secret: &str,
) -> Result<Option<String>, KeyStoreError> {
    for e in &data.entries {
        if verify_api_key(secret, &e.key_hash)? {
            return Ok(Some(e.username.clone()));
        }
    }
    Ok(None)
}

/// Generates a new API key, appends an Argon2 hash for `username`, and writes the encrypted file.
/// Returns the raw API key exactly once (print it for the operator; it cannot be recovered from the keystore).
pub fn add_user_api_key(
    username: &str,
    keystore_path: &Path,
    passphrase: &SecretString,
) -> Result<String, KeyStoreError> {
    let username = normalize_username(username)?;
    let mut data = load_keystore(keystore_path, passphrase)?;
    if data.entries.iter().any(|e| e.username == username) {
        return Err(KeyStoreError::DuplicateUsername(username));
    }
    let api_key = generate_strong_api_key();
    let key_hash = hash_api_key(&api_key)?;
    data.entries.push(KeyEntry { username, key_hash });
    save_keystore(keystore_path, passphrase, &data)?;
    Ok(api_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn resolve_username_for_api_key_matches() {
        let mut data = KeyStoreData::empty();
        let k = generate_strong_api_key();
        let h = hash_api_key(&k).unwrap();
        data.entries.push(KeyEntry {
            username: "bob".into(),
            key_hash: h,
        });
        assert_eq!(
            resolve_username_for_api_key(&data, &k).unwrap(),
            Some("bob".into())
        );
        assert_eq!(resolve_username_for_api_key(&data, "wrong").unwrap(), None);
    }

    #[test]
    fn verify_round_trip_hash() {
        let key = generate_strong_api_key();
        let h = hash_api_key(&key).unwrap();
        assert!(verify_api_key(&key, &h).unwrap());
        assert!(!verify_api_key("wrong", &h).unwrap());
    }

    #[test]
    fn normalize_username_accepts_expected_chars() {
        assert_eq!(normalize_username(" alice_1 ").unwrap(), "alice_1");
        assert!(normalize_username("bad name").is_err());
    }

    #[test]
    fn load_save_keystore_roundtrip() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let passphrase = SecretString::new("unit-test-passphrase".into());
        let mut data = KeyStoreData::empty();
        data.entries.push(KeyEntry {
            username: "u1".into(),
            key_hash: hash_api_key("secret").unwrap(),
        });
        save_keystore(&path, &passphrase, &data).unwrap();
        let loaded = load_keystore(&path, &passphrase).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].username, "u1");
        tmp.as_file_mut().write_all(b"nope").unwrap();
        assert!(load_keystore(&path, &passphrase).is_err());
    }
}
