"""Main FastAPI application for SpatialAI."""
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Resolve repository root reliably regardless of caller CWD (critical for extracted archives)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.isdir(os.path.join(REPO_ROOT, "frontend")) and os.path.isdir(os.path.join(REPO_ROOT, "backend")):
    if os.path.abspath(os.getcwd()) != REPO_ROOT:
        os.chdir(REPO_ROOT)

_ENV_FILE = os.path.join(REPO_ROOT, ".env")
if os.path.exists(_ENV_FILE):
    load_dotenv(dotenv_path=_ENV_FILE, override=False)
else:
    load_dotenv(override=False)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from backend.api.routes import router as api_router
from backend.perception.device import get_device_info
from backend.game.facility import seed_facility_scene
from backend.language_model.llm_provider import global_llm_provider

FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")
ASSETS_DIR = os.path.join(FRONTEND_DIR, "assets")

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

ASSET_MEDIA_TYPES = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    info = get_device_info()
    llm_st = global_llm_provider.get_status()
    print("=" * 60)
    print("🚀 Spatial Quest Backend Initialized")
    print(f"Project Root: {REPO_ROOT}")
    print(f"Frontend Dir: {FRONTEND_DIR}")
    print(f"Hardware Compute Device: {info['selected_device']}")
    print(f"MPS Available (Apple Silicon): {info['mps_available']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"AI Conversation Provider: {llm_st['provider_name']} (mode={llm_st['active_mode']})")
    print("Endpoints active:")
    seeded = seed_facility_scene()
    print(f"Facility scene graph seeded with {seeded} objects")
    print("  - GET  /health")
    print("  - POST /perception")
    print("  - GET  /scene")
    print("  - POST /agent/query")
    print("  - POST /agent/query/stream")
    print("  - POST /agent/clear")
    print("  - GET  /game/state")
    print("  - POST /game/start")
    print("  - POST /game/agent")
    print("  - POST /unity/action-result")
    print("=" * 60)
    yield


app = FastAPI(
    title="Spatial Quest Backend API",
    description="Backend for From-Scratch Computer Vision & Gemini/Transformer Investigation Agent in 3D Virtual Facility",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_index():
    """Serves the authoritative Spatial Quest 3D Investigation Command frontend with no-cache headers."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, headers=NO_CACHE_HEADERS)
    return {"message": "Spatial Quest Backend API Active. Visit /docs for Swagger UI."}


@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon():
    """Returns 204 No Content for favicon requests to prevent 404 console noise."""
    return Response(status_code=204)


@app.get("/three.min.js", include_in_schema=False)
async def serve_three_js():
    """Serves the bundled local Three.js engine."""
    three_path = os.path.join(FRONTEND_DIR, "three.min.js")
    if os.path.exists(three_path):
        return FileResponse(three_path, media_type="application/javascript", headers=NO_CACHE_HEADERS)
    return {"error": "three.min.js not found"}


@app.get("/assets/{file_path:path}", include_in_schema=False)
async def serve_asset(file_path: str):
    """Serves static frontend assets (CSS, JS modules, 3D GLB character models)."""
    safe_target = os.path.abspath(os.path.join(ASSETS_DIR, file_path))
    if os.path.commonpath([ASSETS_DIR, safe_target]) != ASSETS_DIR or not os.path.isfile(safe_target):
        raise HTTPException(status_code=404, detail="Asset not found")
    ext = os.path.splitext(safe_target)[1].lower()
    media_type = ASSET_MEDIA_TYPES.get(ext)
    return FileResponse(safe_target, media_type=media_type, headers=NO_CACHE_HEADERS)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
