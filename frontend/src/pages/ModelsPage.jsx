import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';
import {
    Brain, Car, Zap, DollarSign, TrendingUp, MapPin,
    ArrowUpRight, ArrowDown, ArrowUp, ArrowLeft, ArrowRight, Minus,
    Target, Award, Activity,
} from 'lucide-react';
import { predictPrice, predictDispatch, getPricingModelInfo, getDispatchModelInfo } from '../api';

const tooltipStyle = {
    background: '#1e293b',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '8px',
    color: '#f1f5f9',
    fontSize: '12px',
};

export default function ModelsPage({ pricingInfo, dispatchInfo }) {
    const [activeModel, setActiveModel] = useState('pricing');

    return (
        <div className="models-page">
            {/* Model Selector */}
            <div className="glass-card" style={{ padding: '12px 16px' }}>
                <div className="tab-bar">
                    <button
                        className={`tab-btn ${activeModel === 'pricing' ? 'active' : ''}`}
                        onClick={() => setActiveModel('pricing')}
                    >
                        <Brain size={14} />
                        <span>Model A — FQI Pricing</span>
                    </button>
                    <button
                        className={`tab-btn ${activeModel === 'dispatch' ? 'active' : ''}`}
                        onClick={() => setActiveModel('dispatch')}
                    >
                        <Car size={14} />
                        <span>Model B — SARSA Dispatch</span>
                    </button>
                </div>
            </div>

            <motion.div
                key={activeModel}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
            >
                {activeModel === 'pricing' ? (
                    <PricingModel info={pricingInfo} />
                ) : (
                    <DispatchModel info={dispatchInfo} />
                )}
            </motion.div>
        </div>
    );
}


/* ═══════════════════════════════
   PRICING MODEL
   ═══════════════════════════════ */
function PricingModel({ info }) {
    const [params, setParams] = useState({
        zone_id: 161, hour: 8, demand_level: 4, supply_level: 2,
        day_of_week: 1, is_weekend: 0, competitor_price: 1.0,
    });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handlePredict = useCallback(async () => {
        setLoading(true);
        try {
            const data = await predictPrice(params);
            setResult(data);
        } catch (e) {
            console.error(e);
        }
        setLoading(false);
    }, [params]);

    // Action distribution chart data
    const actionDistData = info?.evaluation?.action_distribution
        ? Object.entries(info.evaluation.action_distribution).map(([action, count]) => ({
            action: `${action}x`,
            count,
        }))
        : [];

    // Q-values chart data
    const qValuesData = result?.all_actions
        ? Object.entries(result.all_actions).map(([action, q]) => ({
            action: `${action}x`,
            q,
            fill: action === String(result.optimal_price) ? '#6366f1' : 'rgba(99,102,241,0.3)',
        }))
        : [];

    return (
        <div style={{ display: 'grid', gap: '20px' }}>
            {/* Model Stats + Prediction Side by Side */}
            <div className="model-panels">
                {/* Training Stats */}
                <div className="glass-card">
                    <div className="model-badge model-a">
                        <Brain size={12} /> Training Results
                    </div>
                    {info?.evaluation ? (
                        <div className="model-stats-grid">
                            <ModelStat label="R² Score" value={info.evaluation.r2_score?.toFixed(4)} color="primary" icon={Target} />
                            <ModelStat label="MAE" value={info.evaluation.mae?.toFixed(4)} color="warning" icon={Activity} />
                            <ModelStat label="Win Rate" value={`${info.evaluation.win_rate_pct?.toFixed(1)}%`} color="success" icon={Award} />
                            <ModelStat label="AI Mean Q" value={info.evaluation.ai_mean_q?.toFixed(2)} color="info" icon={TrendingUp} />
                        </div>
                    ) : (
                        <p style={{ color: 'var(--text-muted)' }}>Training results loading...</p>
                    )}

                    {/* Action Distribution Chart */}
                    {actionDistData.length > 0 && (
                        <div style={{ marginTop: '20px' }}>
                            <div className="card-title" style={{ marginBottom: '12px' }}>
                                <Zap size={14} className="card-title-icon" /> Action Distribution
                            </div>
                            <div style={{ height: '200px' }}>
                                <ResponsiveContainer>
                                    <BarChart data={actionDistData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                        <XAxis dataKey="action" stroke="#64748b" fontSize={11} />
                                        <YAxis stroke="#64748b" fontSize={11} />
                                        <Tooltip contentStyle={tooltipStyle} />
                                        <Bar dataKey="count" radius={[4, 4, 0, 0]} name="Count">
                                            {actionDistData.map((_, i) => (
                                                <Cell key={i} fill={['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe'][i] || '#6366f1'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}
                </div>

                {/* Interactive Prediction */}
                <div className="glass-card">
                    <div className="model-badge model-a">
                        <Zap size={12} /> Interactive Prediction
                    </div>
                    <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '16px' }}>
                        Adjust parameters to see what price the AI recommends:
                    </p>

                    <div className="predict-form">
                        <PredictInput label="Zone ID" value={params.zone_id} onChange={(v) => setParams({ ...params, zone_id: v })} min={1} max={265} />
                        <PredictInput label="Hour" value={params.hour} onChange={(v) => setParams({ ...params, hour: v })} min={0} max={23} />
                        <PredictInput label="Demand" value={params.demand_level} onChange={(v) => setParams({ ...params, demand_level: v })} min={0} max={5} />
                        <PredictInput label="Supply" value={params.supply_level} onChange={(v) => setParams({ ...params, supply_level: v })} min={0} max={5} />
                        <PredictInput label="Day (0=Mon)" value={params.day_of_week} onChange={(v) => setParams({ ...params, day_of_week: v })} min={0} max={6} />
                    </div>

                    <button className="btn btn-primary" onClick={handlePredict} style={{ marginTop: '16px', width: '100%' }}>
                        {loading ? 'Predicting...' : '⚡ Predict Optimal Price'}
                    </button>

                    {/* Prediction Result */}
                    {result && !result.error && (
                        <motion.div
                            className="predict-result"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                        >
                            <div className="predict-result-main">
                                <span className="predict-label">Optimal Price</span>
                                <span className="predict-value primary">{result.optimal_price}x</span>
                            </div>
                            <div className="predict-result-sub">
                                <span>Q-Value: {result.q_value}</span>
                            </div>

                            {/* Q-values bar chart */}
                            {qValuesData.length > 0 && (
                                <div style={{ marginTop: '16px', height: '150px' }}>
                                    <ResponsiveContainer>
                                        <BarChart data={qValuesData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                            <XAxis dataKey="action" stroke="#64748b" fontSize={11} />
                                            <YAxis stroke="#64748b" fontSize={11} />
                                            <Tooltip contentStyle={tooltipStyle} />
                                            <Bar dataKey="q" radius={[4, 4, 0, 0]} name="Q-Value">
                                                {qValuesData.map((d, i) => (
                                                    <Cell key={i} fill={d.fill} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            )}
                        </motion.div>
                    )}
                </div>
            </div>

            {/* Theory Card */}
            <div className="glass-card theory-card">
                <h3 style={{ color: 'var(--accent-primary)', marginBottom: '12px', fontSize: '16px' }}>
                    🎓 How FQI Works
                </h3>
                <div className="theory-grid">
                    <div className="theory-item">
                        <strong>Fitted Q-Iteration</strong>
                        <p>Learns a Q-function Q(state, action) that estimates expected cumulative reward for each pricing action in each market state.</p>
                    </div>
                    <div className="theory-item">
                        <strong>Random Forest</strong>
                        <p>Uses an ensemble of 200 decision trees to approximate Q-values, providing robust generalization across the state space.</p>
                    </div>
                    <div className="theory-item">
                        <strong>Bellman Equation</strong>
                        <p>Q(s,a) = reward + γ · max Q(s',a') — iteratively refines value estimates through the fundamental RL update rule.</p>
                    </div>
                    <div className="theory-item">
                        <strong>Supply-Demand Pricing</strong>
                        <p>Agent learns: surge when demand exceeds supply, discount when oversupplied, balancing revenue with rider satisfaction.</p>
                    </div>
                </div>
            </div>
        </div>
    );
}


/* ═══════════════════════════════
   DISPATCH MODEL
   ═══════════════════════════════ */
function DispatchModel({ info }) {
    const [params, setParams] = useState({
        zone_id: 161, demand_level: 3, supply_level: 2, hour: 8,
    });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handlePredict = useCallback(async () => {
        setLoading(true);
        try {
            const data = await predictDispatch(params);
            setResult(data);
        } catch (e) {
            console.error(e);
        }
        setLoading(false);
    }, [params]);

    const actionIcons = {
        Stay: Minus, Move_North: ArrowUp, Move_South: ArrowDown,
        Move_East: ArrowRight, Move_West: ArrowLeft,
    };

    const actionDistData = info?.evaluation?.action_distribution
        ? Object.entries(info.evaluation.action_distribution).map(([action, count]) => ({
            action: action.replace('Move_', ''),
            count,
        }))
        : [];

    return (
        <div style={{ display: 'grid', gap: '20px' }}>
            <div className="model-panels">
                {/* Training Stats */}
                <div className="glass-card">
                    <div className="model-badge model-b">
                        <Car size={12} /> Training Results
                    </div>
                    {info?.evaluation ? (
                        <div className="model-stats-grid">
                            <ModelStat label="AI Reward" value={info.evaluation.ai_avg_reward?.toFixed(2)} color="success" icon={Award} />
                            <ModelStat label="Random Baseline" value={info.evaluation.random_avg_reward?.toFixed(2)} color="warning" icon={Target} />
                            <ModelStat label="vs Random" value={`+${info.evaluation.improvement_pct?.toFixed(1)}%`} color="success" icon={ArrowUpRight} />
                            <ModelStat label="vs Static" value={`+${info.evaluation.improvement_vs_stay_pct?.toFixed(1)}%`} color="success" icon={ArrowUpRight} />
                        </div>
                    ) : (
                        <p style={{ color: 'var(--text-muted)' }}>Loading...</p>
                    )}

                    {actionDistData.length > 0 && (
                        <div style={{ marginTop: '20px' }}>
                            <div className="card-title" style={{ marginBottom: '12px' }}>
                                <MapPin size={14} className="card-title-icon" /> Action Distribution
                            </div>
                            <div style={{ height: '200px' }}>
                                <ResponsiveContainer>
                                    <BarChart data={actionDistData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                        <XAxis dataKey="action" stroke="#64748b" fontSize={11} />
                                        <YAxis stroke="#64748b" fontSize={11} />
                                        <Tooltip contentStyle={tooltipStyle} />
                                        <Bar dataKey="count" radius={[4, 4, 0, 0]} name="Count">
                                            {actionDistData.map((_, i) => (
                                                <Cell key={i} fill={['#10b981', '#06b6d4', '#f59e0b', '#6366f1', '#ec4899'][i] || '#10b981'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}
                </div>

                {/* Interactive Prediction */}
                <div className="glass-card">
                    <div className="model-badge model-b">
                        <Zap size={12} /> Interactive Prediction
                    </div>
                    <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '16px' }}>
                        Set a driver's state to see the recommended dispatch action:
                    </p>

                    <div className="predict-form">
                        <PredictInput label="Zone ID" value={params.zone_id} onChange={(v) => setParams({ ...params, zone_id: v })} min={1} max={265} />
                        <PredictInput label="Demand" value={params.demand_level} onChange={(v) => setParams({ ...params, demand_level: v })} min={0} max={5} />
                        <PredictInput label="Supply" value={params.supply_level} onChange={(v) => setParams({ ...params, supply_level: v })} min={0} max={5} />
                        <PredictInput label="Hour" value={params.hour} onChange={(v) => setParams({ ...params, hour: v })} min={0} max={23} />
                    </div>

                    <button className="btn btn-success" onClick={handlePredict} style={{ marginTop: '16px', width: '100%' }}>
                        {loading ? 'Predicting...' : '🚗 Predict Dispatch Action'}
                    </button>

                    {result && !result.error && (
                        <motion.div
                            className="predict-result"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                        >
                            <div className="predict-result-main">
                                <span className="predict-label">Recommended Action</span>
                                <span className="predict-value success" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    {(() => {
                                        const Icon = actionIcons[result.optimal_action] || Minus;
                                        return <Icon size={24} />;
                                    })()}
                                    {result.optimal_action}
                                </span>
                            </div>

                            {/* Q-values for all actions */}
                            {result.q_values && (
                                <div style={{ marginTop: '12px' }}>
                                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>Q-Values per Action</div>
                                    {Object.entries(result.q_values).map(([action, q]) => {
                                        const isOptimal = action === result.optimal_action;
                                        return (
                                            <div key={action} style={{
                                                display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px',
                                                padding: '6px 10px', borderRadius: '6px',
                                                background: isOptimal ? 'rgba(16,185,129,0.1)' : 'transparent',
                                                border: isOptimal ? '1px solid rgba(16,185,129,0.3)' : '1px solid transparent',
                                            }}>
                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-secondary)', width: '90px' }}>
                                                    {action}
                                                </span>
                                                <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px' }}>
                                                    <div style={{
                                                        width: `${Math.max(5, (q / Math.max(1, Math.max(...Object.values(result.q_values)))) * 100)}%`,
                                                        height: '100%',
                                                        background: isOptimal ? '#10b981' : 'rgba(99,102,241,0.4)',
                                                        borderRadius: '2px',
                                                    }} />
                                                </div>
                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: isOptimal ? '#10b981' : 'var(--text-muted)', width: '50px', textAlign: 'right' }}>
                                                    {q}
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </motion.div>
                    )}
                </div>
            </div>

            {/* Theory Card */}
            <div className="glass-card theory-card">
                <h3 style={{ color: 'var(--accent-success)', marginBottom: '12px', fontSize: '16px' }}>
                    🎓 How SARSA(λ) Works
                </h3>
                <div className="theory-grid">
                    <div className="theory-item">
                        <strong>SARSA (On-Policy)</strong>
                        <p>State-Action-Reward-State-Action — learns from the actual actions taken, making it safer than off-policy methods like Q-learning.</p>
                    </div>
                    <div className="theory-item">
                        <strong>Eligibility Traces (λ)</strong>
                        <p>Credit assignment mechanism: recent decisions get more credit for outcomes. λ=0.8 balances immediate and long-term credit.</p>
                    </div>
                    <div className="theory-item">
                        <strong>Tile Coding</strong>
                        <p>Converts continuous states (zone, demand, supply, hour) into binary features using 8 overlapping grids with hashing for memory efficiency.</p>
                    </div>
                    <div className="theory-item">
                        <strong>ε-Greedy Policy</strong>
                        <p>Explores random actions with probability ε during training, gradually shifting to exploitation as the agent learns optimal routes.</p>
                    </div>
                </div>
            </div>
        </div>
    );
}


/* ── Shared Components ── */
function ModelStat({ label, value, color, icon: Icon }) {
    return (
        <div className="model-stat">
            <Icon size={16} style={{ color: `var(--accent-${color})` }} />
            <div>
                <div className="model-stat-label">{label}</div>
                <div className={`model-stat-value ${color}`}>{value}</div>
            </div>
        </div>
    );
}

function PredictInput({ label, value, onChange, min, max }) {
    return (
        <div className="predict-input">
            <label>{label}</label>
            <input
                type="number"
                className="input-number"
                value={value}
                onChange={(e) => onChange(Number(e.target.value))}
                min={min}
                max={max}
            />
        </div>
    );
}
