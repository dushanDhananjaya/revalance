"""
Revalance — Dispatch API Routes
=================================
Endpoints for the SARSA dispatch agent (Model B).
"""

import os
from fastapi import APIRouter

from app.simulation.engine import get_engine
from app.ml.dispatch_agent.tile_coding import DISPATCH_ACTIONS

router = APIRouter(prefix="/api/dispatch", tags=["dispatch"])


@router.get("/predict")
async def predict_dispatch(
    zone_id: int = 161,
    demand_level: int = 3,
    supply_level: int = 2,
    hour: int = 8,
):
    """
    Get the optimal dispatch action for a driver at a given state.
    Uses the trained SARSA(λ) model (Model B).
    """
    engine = get_engine()
    
    if engine.dispatch_agent is None:
        return {"error": "Dispatch model not loaded", "default_action": "Stay"}
    
    state = [zone_id, demand_level, supply_level, hour]
    action_idx = engine.dispatch_agent.select_action(state, training=False)
    all_q = engine.dispatch_agent.get_all_q_values(state)
    
    return {
        "optimal_action": DISPATCH_ACTIONS[action_idx],
        "action_index": action_idx,
        "q_values": {k: round(v, 4) for k, v in all_q.items()},
        "state": {
            "zone_id": zone_id,
            "demand_level": demand_level,
            "supply_level": supply_level,
            "hour": hour,
        },
    }


@router.get("/actions")
async def get_dispatch_actions():
    """Get all available dispatch actions."""
    return {"actions": DISPATCH_ACTIONS}


@router.get("/model-info")
async def get_dispatch_model_info():
    """Get info about the trained dispatch model."""
    results_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'ml', 'dispatch_agent', 'sarsa_training_results.json'
    )
    if os.path.exists(results_path):
        import json
        with open(results_path) as f:
            return json.load(f)
    return {"status": "no training results found"}
