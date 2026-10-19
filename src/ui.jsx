import React from 'react';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';

// Plotly.js 3+ can crash in React 19 while unmounting; guard its purge method.
const originalPurge = Plotly.purge;
Plotly.purge = (gd) => {
    try {
        if (originalPurge) originalPurge(gd);
    } catch (e) {
        console.warn('Caught Plotly purge error on unmount:', e);
    }
};

export const Plot = createPlotlyComponent(Plotly);

// Plotly mutates the arrays inside `layout` (and may touch `data`) when the user zooms or pans.
// Every shared array passed to a plot goes through this, so the originals never change.
export const copy = (arr) => [...arr];

export const COLORS = {
    pos: '#f97316',      // a positive weight or signal: pushes the neuron up (orange, as on the slides)
    neg: '#3b82f6',      // a negative one: holds it back (blue)
    act: '#10b981',      // how much a neuron (or an input) is "on": from 0 to 1
    nice: '#f59e0b',
    quiet: '#a78bfa',
    go: '#10b981',
    muted: '#94a3b8',
};

// The fill of a circle whose value v goes from 0 (empty) to 1 (full green).
export const actFill = (v) => `rgba(16, 185, 129, ${0.06 + 0.84 * Math.min(1, Math.max(0, v))})`;

export function Metric({ value, label, className = '' }) {
    return (
        <div className={`metric-card ${className}`}>
            <div className="metric-value">{value}</div>
            <div className="metric-label" style={{ textTransform: 'none' }}>{label}</div>
        </div>
    );
}

export function Toggle({ checked, onChange, children, color = '#3b82f6' }) {
    return (
        <label className="toggle">
            <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} style={{ accentColor: color }} />
            {children}
        </label>
    );
}

// Tabs that switch what one panel shows. options: [{ key, label }].
export function ViewTabs({ options, value, onChange }) {
    return (
        <div className="view-tabs" role="tablist">
            {options.map((o) => (
                <button key={o.key} type="button" role="tab" aria-selected={value === o.key}
                    className={`view-tab ${value === o.key ? 'active' : ''}`} onClick={() => onChange(o.key)}>
                    {o.label}
                </button>
            ))}
        </div>
    );
}

// Base Plotly layout shared by the plots.
export const baseLayout = (pTheme) => ({
    autosize: true,
    showlegend: false,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: pTheme.fontColor, family: 'Inter, system-ui, sans-serif' },
    uirevision: 'keep',
});

export const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };
