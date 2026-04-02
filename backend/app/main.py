"""
Revalance — FastAPI Application Entry Point
============================================
Complete API with CORS, REST routes, and WebSocket support.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.simulation import router as simulation_router
from app.api.routes.pricing import router as pricing_router
from app.api.routes.dispatch import router as dispatch_router
from app.api.websocket import websocket_endpoint, broadcast_state

# Create the FastAPI app
app = FastAPI(
    title="Revalance API",
    description="Dual-Agent RL Ride-Sharing Optimization System",
    version="1.0.0",
)

# CORS — allow frontend (React) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST routers
app.include_router(simulation_router)
app.include_router(pricing_router)
app.include_router(dispatch_router)

# Mount WebSocket
app.websocket("/ws")(websocket_endpoint)


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "message": "Welcome to Revalance API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    from app.simulation.engine import get_engine
    engine = get_engine()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models": {
            "pricing_loaded": engine.pricing_model is not None,
            "dispatch_loaded": engine.dispatch_agent is not None,
        },
    }


@app.on_event("startup")
async def startup():
    """Load models on server start."""
    print("\n🚀 Revalance API starting...")
    from app.simulation.engine import get_engine
    engine = get_engine()
    print("  ✅ Models loaded, server ready!")
