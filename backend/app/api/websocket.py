"""
Revalance — WebSocket Endpoint
================================
Real-time simulation state broadcasting via FastAPI WebSocket.

The frontend connects once, and we push state updates every
simulation step — no polling needed.
"""

import asyncio
import json
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect

# Active WebSocket connections
active_connections: Set[WebSocket] = set()


async def websocket_endpoint(websocket: WebSocket):
    """Handle a WebSocket connection from the frontend."""
    await websocket.accept()
    active_connections.add(websocket)
    print(f"  🔌 WebSocket connected ({len(active_connections)} total)")
    
    try:
        while True:
            # Keep connection alive; receive any client messages
            data = await websocket.receive_text()
            # Client can send commands like {"type": "set_speed", "speed": 10}
            try:
                msg = json.loads(data)
                await handle_client_message(msg, websocket)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        active_connections.discard(websocket)
        print(f"  🔌 WebSocket disconnected ({len(active_connections)} total)")
    except Exception:
        active_connections.discard(websocket)


async def handle_client_message(msg: dict, websocket: WebSocket):
    """Handle commands from the frontend."""
    from app.simulation.engine import get_engine
    engine = get_engine()
    
    msg_type = msg.get("type", "")
    
    if msg_type == "set_speed":
        engine.set_speed(msg.get("speed", 5))
    elif msg_type == "pause":
        engine.pause()
    elif msg_type == "resume":
        engine.resume()
    elif msg_type == "stop":
        engine.stop()


async def broadcast_state(state_dict: dict):
    """Broadcast simulation state to all connected WebSocket clients."""
    if not active_connections:
        return
    
    message = json.dumps(state_dict)
    disconnected = set()
    
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            disconnected.add(connection)
    
    # Clean up disconnected clients
    for conn in disconnected:
        active_connections.discard(conn)
