import { useState } from 'react';
import { motion } from 'framer-motion';
import { LayoutDashboard, BarChart3, Brain, Zap, Menu, X, Sun, Moon } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const NAV_ITEMS = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'models', label: 'AI Models', icon: Brain },
];

export default function Navbar({ activePage, onNavigate, connected }) {
    const [mobileOpen, setMobileOpen] = useState(false);
    const { theme, toggle } = useTheme();

    return (
        <nav className="navbar">
            <div className="navbar-inner">
                {/* Brand */}
                <div className="navbar-brand" onClick={() => onNavigate('dashboard')}>
                    <div className="navbar-logo">
                        <Zap size={20} />
                    </div>
                    <div>
                        <div className="navbar-title">Revalance</div>
                        <div className="navbar-sub">RL Ride Optimization</div>
                    </div>
                </div>

                {/* Desktop Nav */}
                <div className="navbar-links">
                    {NAV_ITEMS.map((item) => (
                        <button
                            key={item.id}
                            className={`nav-link ${activePage === item.id ? 'active' : ''}`}
                            onClick={() => onNavigate(item.id)}
                            id={`nav-${item.id}`}
                        >
                            <item.icon size={16} />
                            <span>{item.label}</span>
                            {activePage === item.id && (
                                <motion.div
                                    className="nav-indicator"
                                    layoutId="navIndicator"
                                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                                />
                            )}
                        </button>
                    ))}
                </div>

                {/* Status + Theme Toggle */}
                <div className="navbar-right">
                    <div className={`conn-dot ${connected ? 'on' : 'off'}`} />
                    <span className="conn-label">{connected ? 'Live' : 'Offline'}</span>

                    {/* Theme Toggle */}
                    <motion.button
                        className="theme-toggle"
                        onClick={toggle}
                        id="btn-theme-toggle"
                        aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                        whileTap={{ scale: 0.9 }}
                        whileHover={{ scale: 1.08 }}
                    >
                        <motion.div
                            key={theme}
                            initial={{ rotate: -30, opacity: 0 }}
                            animate={{ rotate: 0, opacity: 1 }}
                            exit={{ rotate: 30, opacity: 0 }}
                            transition={{ duration: 0.25 }}
                        >
                            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
                        </motion.div>
                    </motion.button>
                </div>

                {/* Mobile Toggle */}
                <button className="mobile-toggle" onClick={() => setMobileOpen(!mobileOpen)}>
                    {mobileOpen ? <X size={20} /> : <Menu size={20} />}
                </button>
            </div>

            {/* Mobile Menu */}
            {mobileOpen && (
                <motion.div
                    className="mobile-menu"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    {NAV_ITEMS.map((item) => (
                        <button
                            key={item.id}
                            className={`mobile-nav-link ${activePage === item.id ? 'active' : ''}`}
                            onClick={() => { onNavigate(item.id); setMobileOpen(false); }}
                        >
                            <item.icon size={16} />
                            <span>{item.label}</span>
                        </button>
                    ))}
                </motion.div>
            )}
        </nav>
    );
}

