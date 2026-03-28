# Revalance : Dual-Agent RL Ride-Sharing Optimization Platform

<div align="center">

**An intelligent ride-sharing simulation platform powered by Reinforcement Learning agents for dynamic pricing and fleet dispatch optimization.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19+-61DAFB.svg)](https://react.dev)
[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)](LICENSE)

</div>

---

## Overview

Revalance uses two independent RL agents to optimize ride-sharing operations in real-time:

| Agent | Algorithm | Role | Performance |
|-------|-----------|------|-------------|
| **Model A** | Fitted Q-Iteration (FQI) | Dynamic Pricing | R² = 0.91, 62% win rate |
| **Model B** | SARSA(λ) with Tile Coding | Fleet Dispatch | +10% vs random, +120% vs static |

Both agents are trained on **500K+ NYC Yellow Taxi records** and integrated into a real-time simulation engine with a premium React dashboard.

## Architecture

```
project-revalance/
├── backend/                    # FastAPI + ML Models
│   ├── app/
│   │   ├── api/routes/        # REST endpoints (14 routes)
│   │   ├── ml/                # FQI pricing + SARSA dispatch
│   │   ├── simulation/        # Engine, state, demand model
│   │   └── main.py            # App entry point
│   └── tests/                 # Pytest suite
├── frontend/                   # React + Vite Dashboard
│   └── src/
│       ├── pages/             # Dashboard, Analytics, AI Models
│       ├── components/        # Navbar
│       └── App.jsx            # Multi-page router
├── data/                       # NYC Taxi datasets
├── docker-compose.yml          # Full-stack containerization
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 15+ (optional, for persistence)

### Backend
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Docker (Full Stack)
```bash
docker-compose up --build
# Backend  → http://localhost:8000
# Frontend → http://localhost:5173
```

## Features

### Real-Time Simulation Dashboard
- 🎮 **Simulation Controls**: Start/pause/stop, fleet size, speed adjustment
- 📊 **Live Metrics**: Revenue, rides, demand, service rate, avg price
- 📈 **4 Real-Time Charts**: Revenue area, rides vs demand, price multiplier, service rate
- 🗺️ **20-Zone Grid**: Color-coded pricing badges (surge/normal/discount)

### Analytics Page
- 📉 **3-Tab Layout**: Overview, Zone Analysis, Time Analysis
- 🍩 **Revenue Donut Chart**: Top 8 zones by revenue
- 📊 **Zone Comparison**: Demand vs supply horizontal bars + data table
- ⏰ **Hourly Performance**: Time-series decomposition

### AI Models Explorer
- 🧠 **Interactive Predictions**: Input parameters → real-time model inference
- 📊 **Q-Value Visualization**: Bar charts showing action Q-values
- 📖 **ML Theory Cards**: FQI → Random Forest + Bellman equation; SARSA(λ) → Tile coding + eligibility traces

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/simulation/start` | Start simulation |
| POST | `/api/simulation/stop` | Stop simulation |
| POST | `/api/simulation/pause` | Pause simulation |
| POST | `/api/simulation/resume` | Resume simulation |
| POST | `/api/simulation/speed` | Set speed |
| GET | `/api/simulation/state` | Current state |
| GET | `/api/simulation/analytics` | Analytics data |
| GET | `/api/simulation/history` | Step history |
| GET | `/api/pricing/predict` | FQI price prediction |
| GET | `/api/pricing/model-info` | FQI training results |
| GET | `/api/dispatch/predict` | SARSA dispatch action |
| GET | `/api/dispatch/model-info` | SARSA training results |
| GET | `/health` | Health check |
| WS | `/ws` | Real-time state stream |

## Testing

```bash
cd backend
.\venv\Scripts\activate
pytest tests/ -v
```

## Tech Stack

- **ML**: scikit-learn, NumPy, Pandas, Joblib
- **Backend**: FastAPI, Uvicorn, SQLAlchemy, Alembic, WebSocket
- **Frontend**: React 19, Vite, Recharts, Framer Motion, Lucide Icons
- **Infrastructure**: Docker, PostgreSQL, GitHub Actions

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
