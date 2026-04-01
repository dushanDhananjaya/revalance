import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
    AreaChart, Area, BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
    RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts';
import {
    TrendingUp, DollarSign, Activity, MapPin, Clock,
    BarChart3, Target, Layers, ArrowUpRight,
} from 'lucide-react';
import { getAnalytics } from '../api';

const PIE_COLORS = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#a78bfa'];

export default function AnalyticsPage({ simState, history }) {
    const [activeTab, setActiveTab] = useState('overview');

    const hasData = history && history.length > 2;

    // Compute per-step deltas
    const stepData = hasData ? history.map((h, i) => {
        const prev = i > 0 ? history[i - 1] : h;
        return {
            step: h.step,
            hour: `${String(h.hour).padStart(2, '0')}:00`,
            stepRevenue: Math.max(0, Math.round(h.revenue - prev.revenue)),
            stepRides: Math.max(0, h.rides - prev.rides),
            stepUnserved: Math.max(0, h.unserved - prev.unserved),
            stepDemand: Math.max(0, h.demand - prev.demand),
            price: h.price,
            service: h.service,
            cumRevenue: h.revenue,
            cumRides: h.rides,
        };
    }) : [];

    // Zone-level analytics
    const zoneData = simState?.zones ? Object.entries(simState.zones)
        .map(([id, z]) => ({
            zone: `Z${id}`,
            demand: z.demand || 0,
            supply: z.supply || 0,
            rides: z.rides || 0,
            revenue: z.revenue || 0,
            price: z.price || 1.0,
        }))
        .sort((a, b) => b.revenue - a.revenue) : [];

    // Top 8 zones for pie chart
    const topZones = zoneData.slice(0, 8);

    // Hourly aggregation
    const hourlyMap = {};
    stepData.forEach((d) => {
        const hourKey = d.hour;
        if (!hourlyMap[hourKey]) hourlyMap[hourKey] = { hour: hourKey, rides: 0, revenue: 0, demand: 0, count: 0 };
        hourlyMap[hourKey].rides += d.stepRides;
        hourlyMap[hourKey].revenue += d.stepRevenue;
        hourlyMap[hourKey].demand += d.stepDemand;
        hourlyMap[hourKey].count += 1;
    });
    const hourlyData = Object.values(hourlyMap).sort((a, b) => a.hour.localeCompare(b.hour));

    const tabs = [
        { id: 'overview', label: 'Overview', icon: BarChart3 },
        { id: 'zones', label: 'Zone Analysis', icon: MapPin },
        { id: 'time', label: 'Time Analysis', icon: Clock },
    ];

    return (
        <div className="analytics-page">
            {/* Tab Switcher */}
            <div className="glass-card" style={{ padding: '12px 16px' }}>
                <div className="tab-bar">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                            onClick={() => setActiveTab(tab.id)}
                        >
                            <tab.icon size={14} />
                            <span>{tab.label}</span>
                        </button>
                    ))}
                </div>
            </div>

            {!hasData ? (
                <div className="glass-card" style={{ textAlign: 'center', padding: '60px 20px' }}>
                    <Activity size={48} style={{ color: 'var(--text-muted)', marginBottom: '16px' }} />
                    <h3 style={{ color: 'var(--text-secondary)', marginBottom: '8px' }}>No Simulation Data Yet</h3>
                    <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
                        Run a simulation from the Dashboard to see analytics here.
                    </p>
                </div>
            ) : (
                <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                >
                    {activeTab === 'overview' && (
                        <OverviewTab stepData={stepData} simState={simState} topZones={topZones} />
                    )}
                    {activeTab === 'zones' && (
                        <ZoneTab zoneData={zoneData} topZones={topZones} />
                    )}
                    {activeTab === 'time' && (
                        <TimeTab stepData={stepData} hourlyData={hourlyData} />
                    )}
                </motion.div>
            )}
        </div>
    );
}

/* ── Overview Tab ── */
function OverviewTab({ stepData, simState, topZones }) {
    const summary = simState ? {
        totalRevenue: simState.total_revenue || 0,
        totalRides: simState.total_rides || 0,
        serviceRate: simState.service_rate || 0,
        avgPrice: simState.avg_price_multiplier || 1.0,
        totalDemand: simState.total_demand || 0,
        unserved: simState.total_unserved || 0,
    } : {};

    return (
        <div style={{ display: 'grid', gap: '20px' }}>
            {/* Summary Cards */}
            <div className="metrics-grid">
                <SummaryCard label="Total Revenue" value={`$${summary.totalRevenue?.toLocaleString()}`} color="success" icon={DollarSign} />
                <SummaryCard label="Rides Matched" value={summary.totalRides?.toLocaleString()} color="primary" icon={Target} />
                <SummaryCard label="Service Rate" value={`${summary.serviceRate}%`} color="warning" icon={Activity} />
                <SummaryCard label="Avg Price" value={`${summary.avgPrice}x`} color="pink" icon={TrendingUp} />
            </div>

            {/* Revenue + Rides Chart */}
            <div className="charts-grid">
                <div className="glass-card">
                    <div className="card-header">
                        <span className="card-title"><DollarSign size={14} className="card-title-icon" /> Revenue Over Time</span>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <AreaChart data={stepData}>
                                <defs>
                                    <linearGradient id="aRevGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="step" stroke="#64748b" fontSize={11} />
                                <YAxis stroke="#64748b" fontSize={11} />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Area type="monotone" dataKey="cumRevenue" stroke="#10b981" fill="url(#aRevGrad)" strokeWidth={2} name="Revenue ($)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="glass-card">
                    <div className="card-header">
                        <span className="card-title"><Layers size={14} className="card-title-icon" /> Revenue by Zone (Top 8)</span>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <PieChart>
                                <Pie
                                    data={topZones}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={100}
                                    paddingAngle={3}
                                    dataKey="revenue"
                                    nameKey="zone"
                                    label={({ zone }) => zone}
                                >
                                    {topZones.map((_, i) => (
                                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip contentStyle={tooltipStyle} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    );
}

/* ── Zone Tab ── */
function ZoneTab({ zoneData, topZones }) {
    return (
        <div style={{ display: 'grid', gap: '20px' }}>
            {/* Zone Revenue Bar */}
            <div className="glass-card">
                <div className="card-header">
                    <span className="card-title"><MapPin size={14} className="card-title-icon" /> Revenue by Zone</span>
                </div>
                <div style={{ height: '350px' }}>
                    <ResponsiveContainer>
                        <BarChart data={zoneData.slice(0, 12)} layout="vertical"
                            margin={{ left: 50 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis type="number" stroke="#64748b" fontSize={11} />
                            <YAxis type="category" dataKey="zone" stroke="#64748b" fontSize={11} />
                            <Tooltip contentStyle={tooltipStyle} />
                            <Bar dataKey="revenue" radius={[0, 4, 4, 0]} name="Revenue ($)">
                                {zoneData.slice(0, 12).map((_, i) => (
                                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Demand vs Supply Comparison */}
            <div className="glass-card">
                <div className="card-header">
                    <span className="card-title"><Activity size={14} className="card-title-icon" /> Demand vs Supply per Zone</span>
                </div>
                <div style={{ height: '300px' }}>
                    <ResponsiveContainer>
                        <BarChart data={zoneData.slice(0, 15)}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="zone" stroke="#64748b" fontSize={10} />
                            <YAxis stroke="#64748b" fontSize={11} />
                            <Tooltip contentStyle={tooltipStyle} />
                            <Legend />
                            <Bar dataKey="demand" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Demand" />
                            <Bar dataKey="supply" fill="#10b981" radius={[4, 4, 0, 0]} name="Supply" />
                            <Bar dataKey="rides" fill="#6366f1" radius={[4, 4, 0, 0]} name="Rides" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Zone Table */}
            <div className="glass-card">
                <div className="card-header">
                    <span className="card-title"><BarChart3 size={14} className="card-title-icon" /> Zone Details</span>
                </div>
                <div className="zone-table-container">
                    <table className="zone-table">
                        <thead>
                            <tr>
                                <th>Zone</th>
                                <th>Demand</th>
                                <th>Supply</th>
                                <th>Rides</th>
                                <th>Revenue</th>
                                <th>Price</th>
                            </tr>
                        </thead>
                        <tbody>
                            {zoneData.map((z) => (
                                <tr key={z.zone}>
                                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{z.zone}</td>
                                    <td style={{ color: 'var(--accent-info)' }}>{z.demand}</td>
                                    <td style={{ color: 'var(--accent-success)' }}>{z.supply}</td>
                                    <td style={{ color: 'var(--accent-primary)' }}>{z.rides}</td>
                                    <td style={{ color: 'var(--accent-success)', fontWeight: 600 }}>${z.revenue.toFixed(2)}</td>
                                    <td>
                                        <span className={`zone-price ${z.price > 1.2 ? 'surge' : z.price < 1.0 ? 'discount' : 'normal'}`}>
                                            {z.price.toFixed(1)}x
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

/* ── Time Tab ── */
function TimeTab({ stepData, hourlyData }) {
    return (
        <div style={{ display: 'grid', gap: '20px' }}>
            {/* Hourly Performance */}
            <div className="glass-card">
                <div className="card-header">
                    <span className="card-title"><Clock size={14} className="card-title-icon" /> Hourly Performance</span>
                </div>
                <div style={{ height: '300px' }}>
                    <ResponsiveContainer>
                        <BarChart data={hourlyData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="hour" stroke="#64748b" fontSize={10} />
                            <YAxis stroke="#64748b" fontSize={11} />
                            <Tooltip contentStyle={tooltipStyle} />
                            <Legend />
                            <Bar dataKey="rides" fill="#6366f1" radius={[4, 4, 0, 0]} name="Rides" />
                            <Bar dataKey="revenue" fill="#10b981" radius={[4, 4, 0, 0]} name="Revenue" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Price Trend */}
            <div className="charts-grid">
                <div className="glass-card">
                    <div className="card-header">
                        <span className="card-title"><TrendingUp size={14} className="card-title-icon" /> Price Multiplier Trend</span>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <LineChart data={stepData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="step" stroke="#64748b" fontSize={11} />
                                <YAxis stroke="#64748b" fontSize={11} domain={[0.5, 2.5]} />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Line type="monotone" dataKey="price" stroke="#f59e0b" strokeWidth={2} dot={false} name="Price Mult" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="glass-card">
                    <div className="card-header">
                        <span className="card-title"><Activity size={14} className="card-title-icon" /> Service Rate Over Time</span>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer>
                            <AreaChart data={stepData}>
                                <defs>
                                    <linearGradient id="svcGrad2" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="step" stroke="#64748b" fontSize={11} />
                                <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} unit="%" />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Area type="monotone" dataKey="service" stroke="#8b5cf6" fill="url(#svcGrad2)" strokeWidth={2} name="Service %" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Step-level Rides */}
            <div className="glass-card">
                <div className="card-header">
                    <span className="card-title"><Target size={14} className="card-title-icon" /> Per-Step Rides Matched</span>
                </div>
                <div style={{ height: '250px' }}>
                    <ResponsiveContainer>
                        <AreaChart data={stepData}>
                            <defs>
                                <linearGradient id="ridesGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="step" stroke="#64748b" fontSize={11} />
                            <YAxis stroke="#64748b" fontSize={11} />
                            <Tooltip contentStyle={tooltipStyle} />
                            <Area type="monotone" dataKey="stepRides" stroke="#6366f1" fill="url(#ridesGrad)" strokeWidth={2} name="Rides per Step" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}


/* ── Helpers ── */
const tooltipStyle = {
    background: '#1e293b',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '8px',
    color: '#f1f5f9',
    fontSize: '12px',
};

function SummaryCard({ label, value, color, icon: Icon }) {
    return (
        <motion.div
            className="metric-card"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
        >
            <div className="metric-label"><Icon size={12} style={{ verticalAlign: '-1px' }} /> {label}</div>
            <div className={`metric-value ${color}`}>{value}</div>
        </motion.div>
    );
}
