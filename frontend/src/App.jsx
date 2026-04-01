import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import {
  Play, Pause, Square, Zap, TrendingUp, DollarSign, Users,
  Activity, MapPin, Car, Clock, BarChart3, Brain,
} from 'lucide-react';
import {
  startSimulation, stopSimulation, pauseSimulation, resumeSimulation,
  setSpeed, getState, createWebSocket, getHealthCheck,
  getPricingModelInfo, getDispatchModelInfo,
} from './api';

import Navbar from './components/Navbar';
import AnalyticsPage from './pages/AnalyticsPage';
import ModelsPage from './pages/ModelsPage';

/* ═══════════════════════════════════════════════════
   MAIN APP — Multi-page Router
   ═══════════════════════════════════════════════════ */
export default function App() {
  const [page, setPage] = useState('dashboard');
  const [simState, setSimState] = useState(null);
  const [status, setStatus] = useState('idle');
  const [history, setHistory] = useState([]);
  const [speed, setSpeedVal] = useState(5);
  const [fleetSize, setFleetSize] = useState(100);
  const [connected, setConnected] = useState(false);
  const [pricingInfo, setPricingInfo] = useState(null);
  const [dispatchInfo, setDispatchInfo] = useState(null);
  const wsRef = useRef(null);
  const histRef = useRef([]);

  // Load model info
  useEffect(() => {
    getHealthCheck().then(() => setConnected(true)).catch(() => setConnected(false));
    getPricingModelInfo().then(setPricingInfo).catch(() => { });
    getDispatchModelInfo().then(setDispatchInfo).catch(() => { });
  }, []);

  // WebSocket
  useEffect(() => {
    const ws = createWebSocket(
      (data) => {
        setSimState(data);
        setStatus(data.status || 'idle');
        if (data.current_step > 0) {
          const snap = {
            step: data.current_step,
            hour: data.current_hour,
            rides: data.total_rides,
            revenue: data.total_revenue,
            unserved: data.total_unserved,
            demand: data.total_demand,
            price: data.avg_price_multiplier,
            service: data.service_rate,
          };
          histRef.current = [...histRef.current, snap];
          if (histRef.current.length > 96) histRef.current = histRef.current.slice(-96);
          setHistory([...histRef.current]);
        }
      },
      () => setConnected(true),
      () => setConnected(false),
    );
    wsRef.current = ws;
    return () => ws.close();
  }, []);

  const handleStart = useCallback(async () => {
    histRef.current = [];
    setHistory([]);
    await startSimulation(fleetSize, speed, 0);
    setStatus('running');
  }, [fleetSize, speed]);

  const handlePause = useCallback(async () => { await pauseSimulation(); setStatus('paused'); }, []);
  const handleResume = useCallback(async () => { await resumeSimulation(); setStatus('running'); }, []);
  const handleStop = useCallback(async () => { await stopSimulation(); setStatus('completed'); }, []);
  const handleSpeedChange = useCallback(async (val) => { setSpeedVal(val); await setSpeed(val); }, []);

  // Polling fallback
  useEffect(() => {
    if (status !== 'running') return;
    const interval = setInterval(async () => {
      try {
        const data = await getState();
        setSimState(data);
        setStatus(data.status || status);
      } catch { }
    }, 2000);
    return () => clearInterval(interval);
  }, [status]);

  return (
    <div className="app-container">
      <Navbar activePage={page} onNavigate={setPage} connected={connected} />

      <AnimatePresence mode="wait">
        <motion.div
          key={page}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.25 }}
          style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}
        >
          {page === 'dashboard' && (
            <DashboardPage
              simState={simState}
              status={status}
              history={history}
              speed={speed}
              fleetSize={fleetSize}
              onStart={handleStart}
              onPause={handlePause}
              onResume={handleResume}
              onStop={handleStop}
              onSpeedChange={handleSpeedChange}
              onFleetChange={setFleetSize}
              pricingInfo={pricingInfo}
              dispatchInfo={dispatchInfo}
            />
          )}
          {page === 'analytics' && (
            <AnalyticsPage simState={simState} history={history} />
          )}
          {page === 'models' && (
            <ModelsPage pricingInfo={pricingInfo} dispatchInfo={dispatchInfo} />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}


/* ═══════════════════════════════════════════════════
   DASHBOARD PAGE
   ═══════════════════════════════════════════════════ */
function DashboardPage({
  simState, status, history, speed, fleetSize,
  onStart, onPause, onResume, onStop, onSpeedChange, onFleetChange,
  pricingInfo, dispatchInfo,
}) {
  return (
    <>
      <ControlBar
        status={status} speed={speed} fleetSize={fleetSize}
        onStart={onStart} onPause={onPause} onResume={onResume} onStop={onStop}
        onSpeedChange={onSpeedChange} onFleetChange={onFleetChange}
        simState={simState}
      />

      {simState && simState.current_step > 0 && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <MetricsRow state={simState} />
        </motion.div>
      )}

      {history.length > 2 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
          <ChartsSection history={history} />
        </motion.div>
      )}

      {simState && Object.keys(simState.zones || {}).length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
          <ZoneGrid zones={simState.zones} />
        </motion.div>
      )}

      <ModelInfoSection pricingInfo={pricingInfo} dispatchInfo={dispatchInfo} />
    </>
  );
}


/* ═══════════════════════════════════════════════════
   CONTROL BAR
   ═══════════════════════════════════════════════════ */
function ControlBar({ status, speed, fleetSize, onStart, onPause, onResume, onStop, onSpeedChange, onFleetChange, simState }) {
  const progress = simState ? simState.progress_pct || 0 : 0;
  return (
    <div className="glass-card">
      <div className="controls-bar">
        {status === 'idle' || status === 'completed' ? (
          <button className="btn btn-primary" onClick={onStart} id="btn-start"><Play size={16} /> Start Simulation</button>
        ) : status === 'running' ? (
          <>
            <button className="btn btn-secondary" onClick={onPause} id="btn-pause"><Pause size={16} /> Pause</button>
            <button className="btn btn-danger" onClick={onStop} id="btn-stop"><Square size={16} /> Stop</button>
          </>
        ) : status === 'paused' ? (
          <>
            <button className="btn btn-success" onClick={onResume} id="btn-resume"><Play size={16} /> Resume</button>
            <button className="btn btn-danger" onClick={onStop} id="btn-stop2"><Square size={16} /> Stop</button>
          </>
        ) : null}

        <div className="input-group">
          <label>Fleet</label>
          <input type="number" className="input-number" value={fleetSize}
            onChange={(e) => onFleetChange(Number(e.target.value))} min={10} max={500}
            disabled={status === 'running'} id="input-fleet" />
        </div>

        <div className="speed-control">
          <label>Speed</label>
          <input type="range" min={1} max={20} value={speed}
            onChange={(e) => onSpeedChange(Number(e.target.value))} id="input-speed" />
          <span className="speed-value">{speed}x</span>
        </div>

        {simState && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--text-muted)' }}>
              <Clock size={14} style={{ verticalAlign: '-2px' }} />{' '}
              {String(simState.current_hour).padStart(2, '0')}:{String(simState.current_minute).padStart(2, '0')}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--accent-primary)' }}>
              Step {simState.current_step}/{simState.total_steps}
            </span>
          </div>
        )}
      </div>
      {status === 'running' && (
        <div className="progress-bar-container">
          <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
        </div>
      )}
    </div>
  );
}


/* ═══════════════════════════════════════════════════
   METRICS ROW
   ═══════════════════════════════════════════════════ */
function MetricsRow({ state }) {
  const metrics = [
    { label: 'Total Revenue', value: `$${state.total_revenue?.toLocaleString() || '0'}`, color: 'success', icon: DollarSign },
    { label: 'Rides Completed', value: state.total_rides?.toLocaleString() || '0', color: 'primary', icon: Car },
    { label: 'Total Demand', value: state.total_demand?.toLocaleString() || '0', color: 'info', icon: Users },
    { label: 'Unserved', value: state.total_unserved?.toLocaleString() || '0', color: 'danger', icon: Activity },
    { label: 'Service Rate', value: `${state.service_rate || 0}%`, color: 'warning', icon: Activity },
    { label: 'Avg Price', value: `${state.avg_price_multiplier || 1.0}x`, color: 'pink', icon: TrendingUp },
  ];
  return (
    <div className="metrics-grid">
      {metrics.map((m, i) => (
        <motion.div key={m.label} className="metric-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
          <div className="metric-label"><m.icon size={12} style={{ verticalAlign: '-1px' }} /> {m.label}</div>
          <div className={`metric-value ${m.color}`}>{m.value}</div>
        </motion.div>
      ))}
    </div>
  );
}


/* ═══════════════════════════════════════════════════
   CHARTS
   ═══════════════════════════════════════════════════ */
const TS = {
  background: 'var(--bg-secondary)',
  border: '1px solid var(--border-glass)',
  borderRadius: '8px',
  color: 'var(--text-primary)',
};

function ChartsSection({ history }) {
  const stepData = history.map((h, i) => {
    const prev = i > 0 ? history[i - 1] : h;
    return {
      step: h.step, hour: `${String(h.hour).padStart(2, '0')}:00`,
      revenue: Math.round(h.revenue - prev.revenue) || 0,
      rides: h.rides - prev.rides || 0,
      unserved: h.unserved - prev.unserved || 0,
      demand: h.demand - prev.demand || 0,
      price: h.price, service: h.service,
      cumRevenue: h.revenue, cumRides: h.rides,
    };
  });

  return (
    <div className="charts-grid">
      <div className="glass-card">
        <div className="card-header"><span className="card-title"><DollarSign size={14} className="card-title-icon" /> Revenue</span></div>
        <div className="chart-container">
          <ResponsiveContainer>
            <AreaChart data={stepData}>
              <defs><linearGradient id="rv" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#10b981" stopOpacity={0.3} /><stop offset="95%" stopColor="#10b981" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="step" stroke="#64748b" fontSize={11} /><YAxis stroke="#64748b" fontSize={11} />
              <Tooltip contentStyle={TS} /><Area type="monotone" dataKey="cumRevenue" stroke="#10b981" fill="url(#rv)" strokeWidth={2} name="Revenue ($)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="glass-card">
        <div className="card-header"><span className="card-title"><BarChart3 size={14} className="card-title-icon" /> Rides vs Demand</span></div>
        <div className="chart-container">
          <ResponsiveContainer>
            <BarChart data={stepData.slice(-24)}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="step" stroke="#64748b" fontSize={11} /><YAxis stroke="#64748b" fontSize={11} />
              <Tooltip contentStyle={TS} />
              <Bar dataKey="rides" fill="#6366f1" radius={[4, 4, 0, 0]} name="Rides" />
              <Bar dataKey="demand" fill="rgba(6,182,212,0.4)" radius={[4, 4, 0, 0]} name="Demand" />
              <Bar dataKey="unserved" fill="rgba(239,68,68,0.5)" radius={[4, 4, 0, 0]} name="Unserved" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="glass-card">
        <div className="card-header"><span className="card-title"><TrendingUp size={14} className="card-title-icon" /> Price Multiplier</span></div>
        <div className="chart-container">
          <ResponsiveContainer>
            <LineChart data={stepData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="step" stroke="#64748b" fontSize={11} /><YAxis stroke="#64748b" fontSize={11} domain={[0.5, 2.5]} />
              <Tooltip contentStyle={TS} /><Line type="monotone" dataKey="price" stroke="#f59e0b" strokeWidth={2} dot={false} name="Price" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="glass-card">
        <div className="card-header"><span className="card-title"><Activity size={14} className="card-title-icon" /> Service Rate</span></div>
        <div className="chart-container">
          <ResponsiveContainer>
            <AreaChart data={stepData}>
              <defs><linearGradient id="sr" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} /><stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="step" stroke="#64748b" fontSize={11} /><YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} unit="%" />
              <Tooltip contentStyle={TS} /><Area type="monotone" dataKey="service" stroke="#8b5cf6" fill="url(#sr)" strokeWidth={2} name="Service %" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}


/* ═══════════════════════════════════════════════════
   ZONE GRID
   ═══════════════════════════════════════════════════ */
function ZoneGrid({ zones }) {
  const zoneEntries = Object.entries(zones).sort((a, b) => (b[1].demand || 0) - (a[1].demand || 0));
  return (
    <div className="glass-card">
      <div className="card-header"><span className="card-title"><MapPin size={14} className="card-title-icon" /> Zone Status ({zoneEntries.length} zones)</span></div>
      <div className="zone-grid">
        {zoneEntries.map(([id, zone]) => {
          const price = zone.price || 1.0;
          const priceClass = price > 1.2 ? 'surge' : price < 1.0 ? 'discount' : 'normal';
          return (
            <motion.div key={id} className="zone-tile" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
              <div className="zone-tile-header">
                <span className="zone-id">Zone {id}</span>
                <span className={`zone-price ${priceClass}`}>{price.toFixed(1)}x</span>
              </div>
              <div className="zone-stats">
                <div><span style={{ color: '#06b6d4' }}>D:</span> <span className="zone-stat-val">{zone.demand || 0}</span></div>
                <div><span style={{ color: '#10b981' }}>S:</span> <span className="zone-stat-val">{zone.supply || 0}</span></div>
                <div><span style={{ color: '#6366f1' }}>R:</span> <span className="zone-stat-val">{zone.rides || 0}</span></div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}


/* ═══════════════════════════════════════════════════
   MODEL INFO
   ═══════════════════════════════════════════════════ */
function ModelInfoSection({ pricingInfo, dispatchInfo }) {
  return (
    <div className="model-panels">
      <div className="model-panel">
        <div className="model-badge model-a"><Brain size={12} /> Model A — FQI Pricing</div>
        {pricingInfo?.evaluation ? (
          <div style={{ display: 'grid', gap: '8px' }}>
            <SL label="R² Score" value={pricingInfo.evaluation.r2_score?.toFixed(4)} />
            <SL label="MAE" value={pricingInfo.evaluation.mae?.toFixed(4)} />
            <SL label="Win Rate" value={`${pricingInfo.evaluation.win_rate_pct?.toFixed(1)}%`} />
            <DistBars data={pricingInfo.evaluation.action_distribution} suffix="x" color="primary" />
          </div>
        ) : <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Loading...</p>}
      </div>
      <div className="model-panel">
        <div className="model-badge model-b"><Car size={12} /> Model B — SARSA(λ) Dispatch</div>
        {dispatchInfo?.evaluation ? (
          <div style={{ display: 'grid', gap: '8px' }}>
            <SL label="AI Reward" value={dispatchInfo.evaluation.ai_avg_reward?.toFixed(2)} />
            <SL label="Random" value={dispatchInfo.evaluation.random_avg_reward?.toFixed(2)} />
            <SL label="vs Random" value={`${dispatchInfo.evaluation.improvement_pct?.toFixed(1)}%`} highlight />
            <SL label="vs Static" value={`${dispatchInfo.evaluation.improvement_vs_stay_pct?.toFixed(1)}%`} highlight />
            <DistBars data={dispatchInfo.evaluation.action_distribution} color="success" />
          </div>
        ) : <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Loading...</p>}
      </div>
    </div>
  );
}

function SL({ label, value, highlight }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
      <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 600, color: highlight ? 'var(--accent-success)' : 'var(--text-primary)' }}>{value}</span>
    </div>
  );
}

function DistBars({ data, suffix = '', color = 'primary' }) {
  if (!data) return null;
  const total = Object.values(data).reduce((a, b) => a + b, 0);
  return (
    <div style={{ marginTop: '8px' }}>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>Action Distribution</div>
      {Object.entries(data).map(([action, count]) => {
        const pct = (count / total * 100).toFixed(1);
        return (
          <div key={action} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-secondary)', width: '80px' }}>{action}{suffix}</span>
            <div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: color === 'success' ? 'var(--gradient-success)' : 'var(--gradient-primary)', borderRadius: '3px' }} />
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', width: '45px', textAlign: 'right' }}>{pct}%</span>
          </div>
        );
      })}
    </div>
  );
}
