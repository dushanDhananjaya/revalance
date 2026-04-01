/* API and WebSocket helpers for connecting to the backend */

const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

/* ── REST API Calls ── */
export async function startSimulation(fleetSize = 100, simSpeed = 5, startHour = 0) {
    const res = await fetch(
        `${API_BASE}/api/simulation/start?fleet_size=${fleetSize}&sim_speed=${simSpeed}&start_hour=${startHour}`,
        { method: 'POST' }
    );
    return res.json();
}

export async function stopSimulation() {
    const res = await fetch(`${API_BASE}/api/simulation/stop`, { method: 'POST' });
    return res.json();
}

export async function pauseSimulation() {
    const res = await fetch(`${API_BASE}/api/simulation/pause`, { method: 'POST' });
    return res.json();
}

export async function resumeSimulation() {
    const res = await fetch(`${API_BASE}/api/simulation/resume`, { method: 'POST' });
    return res.json();
}

export async function setSpeed(speed) {
    const res = await fetch(`${API_BASE}/api/simulation/speed?speed=${speed}`, { method: 'POST' });
    return res.json();
}

export async function getState() {
    const res = await fetch(`${API_BASE}/api/simulation/state`);
    return res.json();
}

export async function getAnalytics() {
    const res = await fetch(`${API_BASE}/api/simulation/analytics`);
    return res.json();
}

export async function getHistory() {
    const res = await fetch(`${API_BASE}/api/simulation/history`);
    return res.json();
}

export async function predictPrice(params) {
    const q = new URLSearchParams(params).toString();
    const res = await fetch(`${API_BASE}/api/pricing/predict?${q}`);
    return res.json();
}

export async function predictDispatch(params) {
    const q = new URLSearchParams(params).toString();
    const res = await fetch(`${API_BASE}/api/dispatch/predict?${q}`);
    return res.json();
}

export async function getHealthCheck() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
}

export async function getPricingModelInfo() {
    const res = await fetch(`${API_BASE}/api/pricing/model-info`);
    return res.json();
}

export async function getDispatchModelInfo() {
    const res = await fetch(`${API_BASE}/api/dispatch/model-info`);
    return res.json();
}


/* ── WebSocket Connection ── */
export function createWebSocket(onMessage, onOpen, onClose) {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('🔌 WebSocket connected');
        if (onOpen) onOpen();
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (onMessage) onMessage(data);
        } catch (e) {
            console.error('WS parse error:', e);
        }
    };

    ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        if (onClose) onClose();
    };

    ws.onerror = (err) => {
        console.error('WS error:', err);
    };

    return ws;
}
