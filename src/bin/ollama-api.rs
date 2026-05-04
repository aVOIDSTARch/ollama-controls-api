//! CLI for API key management (`ollama-api key-gen <username>`).

use age::secrecy::SecretString;
use clap::{Parser, Subcommand};
use ollama_controls_api::auth::{add_user_api_key, default_keystore_path};
use ollama_controls_api::dotenv_loader::load_dotenv_with_exe_fallback;
use std::path::PathBuf;
use std::process::ExitCode;

const PASSPHRASE_ENV: &str = "OLLAMA_CONTROLS_KEYSTORE_PASSPHRASE";

#[derive(Parser)]
#[command(
    name = "ollama-api",
    about = "Manage encrypted API key material for ollama-controls-api",
    version
)]
struct Cli {
    /// Encrypted keystore path (default: XDG config `ollama-controls/api-keys.age`).
    #[arg(long, global = true)]
    keystore: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a strong API key for a user and store only its Argon2 hash (encrypted file).
    KeyGen {
        /// Account label (ASCII letters, digits, '.', '_', '-').
        username: String,
    },
}

fn read_keystore_passphrase(path: &std::path::Path) -> Result<SecretString, String> {
    if let Ok(p) = std::env::var(PASSPHRASE_ENV) {
        if p.is_empty() {
            return Err(format!("{PASSPHRASE_ENV} is set but empty"));
        }
        return Ok(SecretString::new(p.into()));
    }
    if path.exists() {
        let p = rpassword::prompt_password("Keystore passphrase: ")
            .map_err(|e| format!("failed to read passphrase: {e}"))?;
        if p.is_empty() {
            return Err("passphrase must not be empty".into());
        }
        return Ok(SecretString::new(p.into()));
    }
    let a = rpassword::prompt_password("Create keystore passphrase: ")
        .map_err(|e| format!("failed to read passphrase: {e}"))?;
    let b = rpassword::prompt_password("Confirm keystore passphrase: ")
        .map_err(|e| format!("failed to read passphrase: {e}"))?;
    if a != b {
        return Err("passphrases do not match".into());
    }
    if a.is_empty() {
        return Err("passphrase must not be empty".into());
    }
    Ok(SecretString::new(a.into()))
}

fn main() -> ExitCode {
    load_dotenv_with_exe_fallback();

    let cli = Cli::parse();
    let keystore = cli.keystore.unwrap_or_else(default_keystore_path);

    let result = match cli.command {
        Commands::KeyGen { username } => {
            let passphrase = match read_keystore_passphrase(&keystore) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(1);
                }
            };
            add_user_api_key(&username, &keystore, &passphrase)
                .map(|api_key| (api_key, username))
        }
    };

    match result {
        Ok((api_key, username)) => {
            eprintln!(
                "Wrote encrypted keystore for user `{username}` to `{}`.",
                keystore.display()
            );
            if std::env::var(PASSPHRASE_ENV).is_err() {
                eprintln!(
                    "Passphrase was read interactively; set `{PASSPHRASE_ENV}` for non-interactive use."
                );
            }
            eprintln!("The API key is shown once below; it cannot be recovered from the keystore.");
            println!("{api_key}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
