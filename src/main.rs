//! HTTP gateway that exposes [`ollama_controls::OllamaClient`] and a few CLI-backed helpers over REST.
//!
//! Environment:
//! - `OLLAMA_HOST` — forwarded to Ollama (same as the `ollama` CLI), default in client is `http://127.0.0.1:11434`.
//! - `OLLAMA_CONTROLS_BIND` — listen address (default `127.0.0.1:3000`).
//! - `OLLAMA_MODELS` — models directory (also configurable via `POST /api/settings/models-path` and applied when using `POST /api/service/start`).

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use ollama_controls::{
    inspect_model, inspect_model_details, list_local_downloaded_models, models_path_info,
    ollama_start_serve, ollama_stop_serve, set_models_download_path, Model, ModelsPathInfo,
    OllamaClient, PullProgressLine, PsResponse, ShowResponse, TagsResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

#[derive(Clone)]
struct AppState {
    client: OllamaClient,
}

#[tokio::main]
async fn main() {
    let bind = std::env::var("OLLAMA_CONTROLS_BIND")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string());
    let client = OllamaClient::from_env();
    let state = Arc::new(AppState { client });

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/tags", get(list_tags))
        .route("/api/ps", get(list_ps))
        .route("/api/show", get(show_model))
        .route("/api/pull", post(pull_model))
        .route("/api/delete", delete(delete_model))
        .route("/api/copy", post(copy_model))
        .route("/api/create", post(create_model))
        .route("/api/create-from-base", post(create_from_base))
        .route("/api/unload", post(unload_model))
        .route("/api/update-all", post(update_all))
        .route("/api/remove-except", post(remove_except))
        .route("/api/inspect/raw", get(inspect_raw))
        .route("/api/inspect/details", get(inspect_details))
        .route("/api/service/start", post(service_start))
        .route("/api/service/stop", post(service_stop))
        .route("/api/local/models", get(list_local_models))
        .route("/api/settings/models-path", get(get_models_path))
        .route("/api/settings/models-path", post(set_models_path))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .unwrap_or_else(|e| panic!("bind {bind}: {e}"));
    eprintln!("ollama-controls-api listening on http://{bind}");
    axum::serve(listener, app).await.expect("server");
}

async fn health() -> impl IntoResponse {
    Json(json!({ "ok": true }))
}

async fn list_tags(State(state): State<Arc<AppState>>) -> Result<Json<TagsResponse>, ApiError> {
    let client = state.client.clone();
    let models = tokio::task::spawn_blocking(move || client.list_models())
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(TagsResponse { models }))
}

async fn list_ps(State(state): State<Arc<AppState>>) -> Result<Json<PsResponse>, ApiError> {
    let client = state.client.clone();
    let models = tokio::task::spawn_blocking(move || client.list_running())
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(PsResponse { models }))
}

#[derive(Deserialize)]
struct NameQuery {
    name: String,
}

async fn show_model(
    State(state): State<Arc<AppState>>,
    Query(q): Query<NameQuery>,
) -> Result<Json<ShowResponse>, ApiError> {
    let client = state.client.clone();
    let name = q.name;
    let show = tokio::task::spawn_blocking(move || client.show(&name))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(show))
}

#[derive(Deserialize)]
struct PullBody {
    name: String,
}

#[derive(Serialize)]
struct PullResponse {
    lines: Vec<PullProgressLine>,
}

async fn pull_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PullBody>,
) -> Result<Json<PullResponse>, ApiError> {
    let client = state.client.clone();
    let name = body.name;
    let lines = tokio::task::spawn_blocking(move || {
        let mut lines = Vec::new();
        client.pull_stream(&name, |line| {
            lines.push(line.clone());
            Ok(())
        })?;
        Ok::<_, String>(lines)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(PullResponse { lines }))
}

async fn delete_model(
    State(state): State<Arc<AppState>>,
    Query(q): Query<NameQuery>,
) -> Result<StatusCode, ApiError> {
    let client = state.client.clone();
    let name = q.name;
    tokio::task::spawn_blocking(move || client.delete_model(&name))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct CopyBody {
    source: String,
    destination: String,
}

async fn copy_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CopyBody>,
) -> Result<StatusCode, ApiError> {
    let client = state.client.clone();
    tokio::task::spawn_blocking(move || client.copy_model(&body.source, &body.destination))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct CreateBody {
    name: String,
    modelfile: String,
}

#[derive(Serialize)]
struct CreateResponse {
    lines: Vec<serde_json::Value>,
}

async fn create_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateBody>,
) -> Result<Json<CreateResponse>, ApiError> {
    let client = state.client.clone();
    let name = body.name;
    let modelfile = body.modelfile;
    let lines = tokio::task::spawn_blocking(move || {
        let mut lines = Vec::new();
        client.create_model(&name, &modelfile, |v| {
            lines.push(v.clone());
            Ok(())
        })?;
        Ok::<_, String>(lines)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(CreateResponse { lines }))
}

#[derive(Deserialize)]
struct CreateFromBaseBody {
    new_name: String,
    #[serde(rename = "from")]
    from_model: String,
}

async fn create_from_base(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateFromBaseBody>,
) -> Result<Json<CreateResponse>, ApiError> {
    let client = state.client.clone();
    let new_name = body.new_name;
    let from_model = body.from_model;
    let lines = tokio::task::spawn_blocking(move || {
        let mut lines = Vec::new();
        client.create_from_base(&new_name, &from_model, |v| {
            lines.push(v.clone());
            Ok(())
        })?;
        Ok::<_, String>(lines)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(CreateResponse { lines }))
}

#[derive(Deserialize)]
struct UnloadBody {
    model: String,
}

async fn unload_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<UnloadBody>,
) -> Result<StatusCode, ApiError> {
    let client = state.client.clone();
    let model = body.model;
    tokio::task::spawn_blocking(move || client.unload_model(&model))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn update_all(State(state): State<Arc<AppState>>) -> Result<Json<serde_json::Value>, ApiError> {
    let client = state.client.clone();
    let names = tokio::task::spawn_blocking(move || client.update_all_models(|_name, _line| Ok(())))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(json!({ "updated": names })))
}

#[derive(Deserialize)]
struct RemoveExceptBody {
    keep: Vec<String>,
}

async fn remove_except(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RemoveExceptBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let client = state.client.clone();
    let keep = body.keep;
    let removed = tokio::task::spawn_blocking(move || {
        let keep_refs: Vec<&str> = keep.iter().map(|s| s.as_str()).collect();
        client.remove_all_except(&keep_refs)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(json!({ "removed": removed })))
}

async fn inspect_raw(
    Query(q): Query<NameQuery>,
) -> Result<(StatusCode, String), ApiError> {
    let model = Model::new(
        q.name,
        String::new(),
        String::new(),
        String::new(),
    );
    let text = tokio::task::spawn_blocking(move || inspect_model(&model))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok((StatusCode::OK, text))
}

async fn inspect_details(
    Query(q): Query<NameQuery>,
) -> Result<Json<ollama_controls::ModelDetails>, ApiError> {
    let model = Model::new(
        q.name,
        String::new(),
        String::new(),
        String::new(),
    );
    let details = tokio::task::spawn_blocking(move || inspect_model_details(&model))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(details))
}

#[derive(Deserialize)]
struct ModelsPathBody {
    path: String,
}

async fn service_start() -> Result<Json<serde_json::Value>, ApiError> {
    let pid = tokio::task::spawn_blocking(ollama_start_serve)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(json!({ "pid": pid })))
}

async fn service_stop() -> Result<StatusCode, ApiError> {
    tokio::task::spawn_blocking(ollama_stop_serve)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_local_models() -> Result<Json<Vec<String>>, ApiError> {
    let lines = tokio::task::spawn_blocking(list_local_downloaded_models)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(lines))
}

async fn get_models_path() -> Json<ModelsPathInfo> {
    Json(models_path_info())
}

async fn set_models_path(
    Json(body): Json<ModelsPathBody>,
) -> Result<Json<ModelsPathInfo>, ApiError> {
    let path = body.path;
    let info = tokio::task::spawn_blocking(move || {
        set_models_download_path(Path::new(&path))?;
        Ok::<_, String>(models_path_info())
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, e))?;
    Ok(Json(info))
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({ "error": self.message }));
        (self.status, body).into_response()
    }
}
