"""Main FastAPI application for SpatialAI."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router as api_router
from backend.perception.device import get_device_info
from backend.game.facility import seed_facility_scene

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    info = get_device_info()
    print("=" * 60)
    print("🚀 SpatialAI Backend Initialized")
    print(f"Hardware Compute Device: {info['selected_device']}")
    print(f"MPS Available (Apple Silicon): {info['mps_available']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print("Endpoints active:")
    seeded = seed_facility_scene()
    print(f"Facility scene graph seeded with {seeded} objects")
    print("  - GET  /health")
    print("  - POST /perception")
    print("  - GET  /scene")
    print("  - POST /agent/query")
    print("  - GET  /game/state")
    print("  - POST /game/start")
    print("  - POST /game/agent")
    print("  - POST /unity/action-result")
    print("=" * 60)
    yield
    # Shutdown logic (if any)

app = FastAPI(
    title="Spatial Quest Backend API",
    description="Backend for From-Scratch Computer Vision & Transformer Agent in 3D Virtual Facility",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Unity and Web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="")

import os
from fastapi.responses import FileResponse, Response

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the interactive Spatial Quest AI Investigation Web Terminal."""
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html"))
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Spatial Quest Backend API Active. Visit /docs for Swagger UI."}

@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon():
    """Returns 204 No Content for favicon requests to prevent 404 console noise."""
    return Response(status_code=204)

@app.get("/three.min.js", include_in_schema=False)
async def serve_three_js():
    """Serves the bundled local Three.js engine."""
    three_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "three.min.js"))
    if os.path.exists(three_path):
        return FileResponse(three_path, media_type="application/javascript")
    return {"error": "three.min.js not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

