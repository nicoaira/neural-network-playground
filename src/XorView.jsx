import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Play, RotateCcw } from 'lucide-react';
import { sigmoid, num, wnum, par } from './network';
import {
    XOR, CLOUD, PRESETS, ONE_START, xorForward, neuronLogic, outputLogic, networkLogic, range, lineInBox, halfPlaneInBox,
} from './xor';
import { Sym, Yhat } from './Sym';
import { Plot, COLORS, Metric, Toggle, ViewTabs, baseLayout, PLOT_CONFIG } from './ui';
import XorDiagram, { NEURON_COLORS } from './XorDiagram';

const BOX = [-0.5, 1.5];
const G = range(-0.5, 1.5, 161);
const GRID_C = range(-0.5, 1.5, 9);
const GRID_S = range(-0.5, 1.5, 41);
const MORPH_MS = 1600;
const TINT = { out: 'rgba(16, 185, 129, 0.24)' };
// The yes side of each hidden neuron is hatched (/ amber, \\ violet): where both say yes the hatches cross.
const HATCH = {
    1: { shape: '/', fgcolor: 'rgba(245, 158, 11, 0.75)', bgcolor: 'rgba(0, 0, 0, 0)', size: 11, solidity: 0.12 },
    2: { shape: '\\', fgcolor: 'rgba(167, 139, 250, 0.85)', bgcolor: 'rgba(0, 0, 0, 0)', size: 11, solidity: 0.12 },
};

const fmtPt = (pt) => `(${num(pt.x1, pt.x1 % 1 ? 2 : 0)}, ${num(pt.x2, pt.x2 % 1 ? 2 : 0)})`;

// The points as Plotly traces: y = 0 blue circles, y = 1 orange crosses (the four XOR points large, the
// cloud around them small), mistakes ringed in red, the highlighted XOR point ringed.
function pointTraces({ pts, xy, yhat, showMistakes, hl, pTheme }) {
    const pick = (f) => pts.map((pt, n) => ({ pt, n })).filter(({ pt, n }) => f(pt, n));
    const scatter = (sel, marker) => ({
        type: 'scatter', mode: 'markers', x: sel.map(({ n }) => xy[n][0]), y: sel.map(({ n }) => xy[n][1]),
        text: sel.map(({ pt, n }) => `x = ${fmtPt(pt)} · y = ${pt.y} · ŷ = ${yhat[n]}`), hovertemplate: '%{text}<extra></extra>', marker,
    });
    const out = [
        scatter(pick((pt) => !pt.big && pt.y === 0), { symbol: 'circle', size: 7, color: COLORS.neg, opacity: 0.55 }),
        scatter(pick((pt) => !pt.big && pt.y === 1), { symbol: 'x', size: 8, color: COLORS.pos, opacity: 0.6 }),
        scatter(pick((pt) => pt.big && pt.y === 0), { symbol: 'circle', size: 18, color: COLORS.neg, line: { color: 'white', width: 2 } }),
        scatter(pick((pt) => pt.big && pt.y === 1), { symbol: 'x', size: 19, color: COLORS.pos, line: { color: 'white', width: 1.5 } }),
    ];
    if (showMistakes) {
        const wrong = pick((pt, n) => yhat[n] !== pt.y);
        out.push({
            type: 'scatter', mode: 'markers', hoverinfo: 'skip', x: wrong.map(({ n }) => xy[n][0]), y: wrong.map(({ n }) => xy[n][1]),
            marker: { symbol: 'circle-open', size: wrong.map(({ pt }) => (pt.big ? 32 : 14)), color: '#ef4444', line: { width: 2.5 } },
        });
    }
    if (hl != null) {
        const n = pts.findIndex((pt) => pt.big && pt.i === hl);
        out.push({
            type: 'scatter', mode: 'markers', hoverinfo: 'skip', x: [xy[n][0]], y: [xy[n][1]],
            marker: { symbol: 'circle-open', size: 42, color: pTheme.fontColor, line: { width: 2 } },
        });
    }
    return out;
}

// Where a neuron's arrow and label go: at the fraction `at` of its line, towards its yes side.
function arrowAt(w1, w2, segm, box, at) {
    const L = Math.hypot(w1, w2);
    const span = box[0][1] - box[0][0];
    const mid = [segm[0][0] + at * (segm[1][0] - segm[0][0]), segm[0][1] + at * (segm[1][1] - segm[0][1])];
    const tip = [mid[0] + (0.1 * span * w1) / L, mid[1] + (0.1 * span * w2) / L];
    const lab = [tip[0] + (0.05 * span * w1) / L, tip[1] + (0.05 * span * w2) / L];
    return { mid, tip, lab };
}

// A neuron's line w1·u + w2·v + b = 0 in a box, the side where it says yes (a tint or a hatch), and an arrow
// towards that side, drawn at the fraction `at` of the line.
function neuronLine({ w1, w2, b, box, color, tint, hatch, label, width = 3, opacity = 1, at = 0.5 }) {
    const traces = [];
    const annotations = [];
    const poly = halfPlaneInBox(w1, w2, -b, 1, box[0], box[1]);
    if (poly.length > 2) {
        traces.push({
            type: 'scatter', mode: 'lines', x: poly.map((q) => q[0]), y: poly.map((q) => q[1]), fill: 'toself',
            fillcolor: hatch ? 'rgba(0, 0, 0, 0)' : tint, ...(hatch ? { fillpattern: hatch } : {}), line: { width: 0 }, hoverinfo: 'skip', opacity,
        });
    }
    const segm = lineInBox(w1, w2, -b, box[0], box[1]);
    if (segm.length === 2) {
        traces.push({
            type: 'scatter', mode: 'lines', x: segm.map((q) => q[0]), y: segm.map((q) => q[1]), hoverinfo: 'skip',
            line: { color, width }, opacity,
        });
        const { mid, tip, lab } = arrowAt(w1, w2, segm, box, at);
        annotations.push(
            { x: tip[0], y: tip[1], ax: mid[0], ay: mid[1], xref: 'x', yref: 'y', axref: 'x', ayref: 'y', text: '', showarrow: true, arrowhead: 2, arrowsize: 1.2, arrowwidth: 2.5, arrowcolor: color, opacity },
            { x: lab[0], y: lab[1], text: label, showarrow: false, font: { color, size: 13 }, opacity, bgcolor: 'rgba(0,0,0,0)' },
        );
    }
    return { traces, annotations };
}

// A square plane (the same scale on both axes); ticks at 0, 0.5 and 1 while the range is small.
const ticks = (r) => (r[1] - r[0] <= 3 ? { tickvals: [0, 0.5, 1] } : {});
const planeLayout = (pTheme, xr, yr, xt, yt) => ({
    ...baseLayout(pTheme),
    margin: { t: 10, r: 10, l: 50, b: 45 },
    xaxis: { title: { text: xt }, range: xr, gridcolor: pTheme.gridColor, zeroline: false, ...ticks(xr), constrain: 'domain' },
    yaxis: { title: { text: yt }, range: yr, gridcolor: pTheme.gridColor, zeroline: false, ...ticks(yr), scaleanchor: 'x', scaleratio: 1, constrain: 'domain' },
});

function ParamSlider({ label, value, min, max, step, onChange }) {
    return (
        <div className="slider-container in-row">
            <div className="slider-header">
                <span>{label}</span>
                <span className="in-val">{wnum(value)}</span>
            </div>
            <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
        </div>
    );
}

// ── 1 · One neuron is not enough ────────────────────────────────────────────
function OneNeuron({ pTheme }) {
    const [q, setQ] = useState(ONE_START);
    const set = (key) => (v) => setQ({ ...q, [key]: v });
    const pts = XOR.map((pt, i) => ({ ...pt, big: true, i }));
    const fs = pts.map((pt) => sigmoid(q.w1 * pt.x1 + q.w2 * pt.x2 + q.b));
    const yhat = fs.map((f) => (f >= 0.5 ? 1 : 0));
    const wrong = pts.filter((pt, n) => yhat[n] !== pt.y).length;
    const line = neuronLine({ w1: q.w1, w2: q.w2, b: q.b, box: [BOX, BOX], color: COLORS.go, tint: TINT.out, label: 'says yes' });
    return (
        <div className="plots-grid glass-panel">
            <div className="panel">
                <h3>One neuron: one straight line</h3>
                <Plot
                    data={[...line.traces, ...pointTraces({ pts, xy: pts.map((pt) => [pt.x1, pt.x2]), yhat, showMistakes: true, pTheme })]}
                    layout={{ ...planeLayout(pTheme, BOX, BOX, 'x₁', 'x₂'), annotations: line.annotations }}
                    config={PLOT_CONFIG} useResizeHandler style={{ width: '100%', height: '460px' }}
                />
            </div>
            <div className="panel">
                <div className="info-card">
                    One neuron is <strong>logistic regression</strong>: f = σ(<Sym b="w" j={1} />x₁ + <Sym b="w" j={2} />x₂ + b).
                    It says yes (f ≥ 0.5) on one side of the straight line <Sym b="w" j={1} />x₁ + <Sym b="w" j={2} />x₂ + b = 0, its
                    decision boundary. Move the line: can you put the two <span style={{ color: COLORS.pos }}>crosses</span> on its
                    green side and the two <span style={{ color: COLORS.neg }}>circles</span> on the other?
                </div>
                <ParamSlider label={<>weight <Sym b="w" j={1} /> (x₁)</>} value={q.w1} min={-10} max={10} step={0.5} onChange={set('w1')} />
                <ParamSlider label={<>weight <Sym b="w" j={2} /> (x₂)</>} value={q.w2} min={-10} max={10} step={0.5} onChange={set('w2')} />
                <ParamSlider label="bias b" value={q.b} min={-10} max={10} step={0.5} onChange={set('b')} />
                <div className="formula">boundary: {wnum(q.w1)} x₁ + {par(wnum(q.w2), q.w2)} x₂ + {par(wnum(q.b), q.b)} = 0</div>
                <div className="metrics-grid">
                    <Metric value={`${wrong} / 4`} label="mistakes" className={wrong ? 'decision no' : 'decision yes'} />
                    <Metric value="1 / 4" label="the fewest any straight line can make" />
                </div>
                <div className="info-card">
                    Every straight line gets at least one point wrong: the crosses sit on one diagonal and the circles on the other,
                    so any line that has both crosses on one side has a circle there too. XOR needs a boundary that is
                    <strong> not a straight line</strong>. The next tab builds one from two neurons.
                </div>
            </div>
        </div>
    );
}

// ── 2 · Two hidden neurons ──────────────────────────────────────────────────
function TwoNeurons({ pTheme }) {
    const [p, setP] = useState(PRESETS[0].p);
    const [act, setAct] = useState('sigmoid');
    const [show, setShow] = useState({ 1: true, 2: true, out: true });
    const [cloud, setCloud] = useState(false);
    const [focus, setFocus] = useState(null); // the neuron pointed at: 1, 2 or 3 (the output)
    const [sel, setSel] = useState(1); // the XOR point shown on the network
    const [hlRow, setHlRow] = useState(null);
    const [t, setT] = useState(1); // 0: the points where they are (x⃗); 1: where the hidden layer puts them (a⃗^[1])
    const raf = useRef(null);
    const set = (key) => (v) => setP({ ...p, [key]: v });

    useEffect(() => () => cancelAnimationFrame(raf.current), []);
    const playMorph = () => {
        cancelAnimationFrame(raf.current);
        const t0 = performance.now();
        const step = (now) => {
            const s = Math.min(1, (now - t0) / MORPH_MS);
            setT(Math.round(100 * (s < 0.5 ? 2 * s * s : 1 - 2 * (1 - s) ** 2)) / 100);
            if (s < 1) raf.current = requestAnimationFrame(step);
        };
        setT(0);
        raf.current = requestAnimationFrame(step);
    };

    const pts = useMemo(() => [...(cloud ? CLOUD.map((pt) => ({ ...pt, big: false })) : []), ...XOR.map((pt, i) => ({ ...pt, big: true, i }))], [cloud]);
    const fws = pts.map((pt) => xorForward(p, pt.x1, pt.x2, act));
    const yhat = fws.map((r) => r.yhat);
    const big = XOR.map((pt) => xorForward(p, pt.x1, pt.x2, act));
    const wrong = big.filter((r, i) => r.yhat !== XOR[i].y).length;
    const hl = hlRow ?? sel;
    const sig = act === 'sigmoid';

    const l1 = neuronLogic(p.w111, p.w112, p.b11);
    const l2 = neuronLogic(p.w121, p.w122, p.b12);
    const lo = outputLogic(p);
    const net = networkLogic(p, act);
    const combine = !sig ? null : lo.name === 'AND' ? 'overlap' : lo.name === 'OR' ? 'union' : null;

    // ── the input plane ──
    const fGrid = useMemo(() => G.map((y) => G.map((x) => xorForward(p, x, y, act).f)), [p, act]);
    // Each neuron's arrow and label go where they are farthest from the four points and from each other.
    const segs = [1, 2].map((j) => lineInBox(p[`w1${j}1`], p[`w1${j}2`], -p[`b1${j}`], BOX, BOX));
    let ats = [0.25, 0.75];
    if (segs.every((sg) => sg.length === 2)) {
        const lab = (j, a) => arrowAt(p[`w1${j}1`], p[`w1${j}2`], segs[j - 1], [BOX, BOX], a).lab;
        const clear = (q) => Math.min(...XOR.map((pt) => Math.hypot(q[0] - pt.x1, q[1] - pt.x2)));
        const AT = [0.15, 0.3, 0.5, 0.7, 0.85];
        let score = -1;
        for (const a1 of AT) {
            for (const a2 of AT) {
                const [u, v] = [lab(1, a1), lab(2, a2)];
                const sc = Math.min(clear(u), clear(v), Math.hypot(u[0] - v[0], u[1] - v[1]) / 2);
                if (sc > score) [score, ats] = [sc, [a1, a2]];
            }
        }
    }
    const plane = [];
    const ann = [];
    if (show.out) {
        plane.push({
            type: 'heatmap', x: G, y: G, z: fGrid.map((row) => row.map((f) => (f >= 0.5 ? 1 : null))), zsmooth: false, showscale: false,
            colorscale: [[0, TINT.out], [1, TINT.out]], hoverinfo: 'skip',
        });
    }
    [1, 2].forEach((j) => {
        if (!show[j]) return;
        const w = neuronLine({
            w1: p[`w1${j}1`], w2: p[`w1${j}2`], b: p[`b1${j}`], box: [BOX, BOX], color: NEURON_COLORS[j - 1], hatch: HATCH[j],
            label: `neuron ${j} says yes`, width: focus === j ? 4.5 : 2.5, opacity: focus && focus !== j && focus !== 3 ? 0.35 : 1,
            at: ats[j - 1],
        });
        plane.push(...w.traces);
        ann.push(...w.annotations);
    });
    if (show.out) {
        plane.push({
            type: 'contour', x: G, y: G, z: fGrid, autocontour: false, contours: { start: 0.5, end: 0.5, size: 1, coloring: 'lines' },
            colorscale: [[0, COLORS.go], [1, COLORS.go]], line: { width: focus === 3 ? 13 : 9 }, opacity: 0.85, showscale: false, hoverinfo: 'skip',
        });
    }
    plane.push(...pointTraces({ pts, xy: pts.map((pt) => [pt.x1, pt.x2]), yhat, showMistakes: true, hl, pTheme }));

    // ── what the output neuron sees: every point (and a grid) on its way from x⃗ to a⃗^[1] ──
    const move = (x1, x2) => {
        const r = xorForward(p, x1, x2, act);
        return [(1 - t) * x1 + t * r.a1, (1 - t) * x2 + t * r.a2];
    };
    const gridLine = (strong) => {
        const xs = [];
        const ys = [];
        GRID_C.filter((c) => (c === 0 || c === 1) === strong).forEach((c) => {
            for (const along of [true, false]) {
                GRID_S.forEach((s) => {
                    const [u, v] = move(along ? c : s, along ? s : c);
                    xs.push(u);
                    ys.push(v);
                });
                xs.push(null);
                ys.push(null);
            }
        });
        return {
            type: 'scatter', mode: 'lines', x: xs, y: ys, hoverinfo: 'skip',
            line: { color: strong ? COLORS.muted : pTheme.gridColor, width: strong ? 2 : 1 },
        };
    };
    const movedPts = pts.map((pt) => move(pt.x1, pt.x2));
    const grids = [gridLine(false), gridLine(true)];
    const allX = [...grids.flatMap((g) => g.x), ...movedPts.map((q) => q[0])].filter((v) => v != null);
    const allY = [...grids.flatMap((g) => g.y), ...movedPts.map((q) => q[1])].filter((v) => v != null);
    const pad = (lo2, hi2) => [lo2 - 0.06 * (hi2 - lo2 || 1), hi2 + 0.06 * (hi2 - lo2 || 1)];
    const hx = pad(Math.min(...allX), Math.max(...allX));
    const hy = pad(Math.min(...allY), Math.max(...allY));
    const hidden = [...grids];
    const hAnn = [];
    if (t === 1) {
        const o = neuronLine({ w1: p.w211, w2: p.w212, b: p.b21, box: [hx, hy], color: COLORS.go, tint: TINT.out, label: 'output says yes', width: focus === 3 ? 5 : 3.5 });
        hidden.push(...o.traces);
        hAnn.push(...o.annotations);
    }
    hidden.push(...pointTraces({ pts, xy: movedPts, yhat, showMistakes: true, hl, pTheme }));
    const hTitle = (k) => (t === 1 ? `a${'₁₂'[k]}: neuron ${k + 1}` : t === 0 ? `x${'₁₂'[k]}` : `from x${'₁₂'[k]} to a${'₁₂'[k]}`);

    const cards = [
        { n: 1, title: 'Neuron 1 (layer 1)', keys: ['w111', 'w112', 'b11'], logic: l1 },
        { n: 2, title: 'Neuron 2 (layer 1)', keys: ['w121', 'w122', 'b12'], logic: l2 },
        { n: 3, title: 'Output neuron (layer 2)', keys: ['w211', 'w212', 'b21'], logic: sig ? lo : null },
    ];
    const symOf = (key) => (key[0] === 'b' ? <Sym b="b" l={Number(key[1])} j={Number(key[2])} /> : <Sym b="w" l={Number(key[1])} j={Number(key[2])} k={Number(key[3])} />);
    const fromOf = (key) => (key[0] === 'b' ? 'bias' : key[1] === '1' ? `from x${'₁₂'[Number(key[3]) - 1]}` : `from neuron ${key[3]}`);
    const x = [XOR[sel].x1, XOR[sel].x2];

    return (
        <>
            <div className="play-grid glass-panel">
                <div className="panel">
                    <div className="panel-header">
                        <h3>The input plane: two lines, one region</h3>
                        <div className="toggle-row">
                            <Toggle checked={show[1]} onChange={(v) => setShow({ ...show, 1: v })} color={NEURON_COLORS[0]}>neuron 1</Toggle>
                            <Toggle checked={show[2]} onChange={(v) => setShow({ ...show, 2: v })} color={NEURON_COLORS[1]}>neuron 2</Toggle>
                            <Toggle checked={show.out} onChange={(v) => setShow({ ...show, out: v })} color={COLORS.go}>output</Toggle>
                            <Toggle checked={cloud} onChange={setCloud} color={COLORS.muted}>more points</Toggle>
                        </div>
                    </div>
                    <Plot data={plane} layout={{ ...planeLayout(pTheme, BOX, BOX, 'x₁', 'x₂'), annotations: ann }}
                        config={PLOT_CONFIG} useResizeHandler style={{ width: '100%', height: '500px' }} />
                    <div className="info-card">
                        {sig ? (
                            <>
                                Each hidden neuron draws <strong>one straight line</strong> and says yes on the hatched side, where its arrow points:{' '}
                                <span style={{ color: NEURON_COLORS[0] }}>neuron 1</span> asks “{l1.question}”,{' '}
                                <span style={{ color: NEURON_COLORS[1] }}>neuron 2</span> asks “{l2.question}”. The output neuron combines
                                their answers{lo.name ? <> with an <strong>{lo.name}</strong></> : null}
                                {combine ? <>: the <span style={{ color: COLORS.go }}>green region</span> is the <strong>{combine}</strong> of the two yes sides.</> : '.'}
                                {' '}Its boundary (the wide green line, along the two neuron lines) is <strong>not one straight line</strong>: no single
                                neuron could draw it.
                            </>
                        ) : (
                            <>
                                <strong>Without σ</strong>, a₁ and a₂ are straight-line functions of x₁ and x₂, and so is the output’s
                                z = <Sym b="w" l={2} j={1} k={1} />a₁ + <Sym b="w" l={2} j={1} k={2} />a₂ + b. The whole network is one
                                neuron again: the green region is cut by one straight line (or none), and the two hidden lines cannot
                                combine into a band. Switch σ back on.
                            </>
                        )}
                    </div>
                </div>
                <div className="panel">
                    <div className="panel-header">
                        <h3>The nine numbers</h3>
                    </div>
                    <div className="preset-row">
                        {PRESETS.map((pr) => (
                            <button key={pr.key} type="button" className={`chip ${Object.keys(pr.p).every((k) => pr.p[k] === p[k]) ? 'active' : ''}`}
                                onClick={() => setP(pr.p)}>{pr.label}</button>
                        ))}
                    </div>
                    <div className="act-row">
                        <span className="small muted">Activation of the hidden layer:</span>
                        <ViewTabs options={[{ key: 'sigmoid', label: 'σ (sigmoid)' }, { key: 'linear', label: 'none (a = z)' }]} value={act} onChange={setAct} />
                    </div>
                    {cards.map((cd) => (
                        <div key={cd.n} className={`neuron-card ${focus === cd.n ? 'focus' : ''}`} style={{ borderLeftColor: NEURON_COLORS[cd.n - 1] }}
                            onMouseEnter={() => setFocus(cd.n)} onMouseLeave={() => setFocus(null)}>
                            <div className="neuron-card-head">
                                <strong style={{ color: NEURON_COLORS[cd.n - 1] }}>{cd.title}</strong>
                                {cd.logic && <span className="small muted">{cd.n === 3 ? cd.logic.question : `“${cd.logic.question}”`}</span>}
                            </div>
                            <div className="slider-trio">
                                {cd.keys.map((key) => (
                                    <ParamSlider key={key} label={<>{symOf(key)} <span className="muted small">{fromOf(key)}</span></>} value={p[key]}
                                        min={key[0] === 'b' ? -30 : -20} max={key[0] === 'b' ? 30 : 20} step={0.5} onChange={set(key)} />
                                ))}
                            </div>
                        </div>
                    ))}
                    <div className="metrics-grid">
                        <Metric value={`${wrong} / 4`} label="mistakes on the four XOR points" className={wrong ? 'decision no' : 'decision yes'} />
                        <Metric value={net.name === 'XOR' ? 'XOR ✓' : net.name || '—'} label={<>the network answers: “{net.question}”</>}
                            className={net.name === 'XOR' ? 'decision yes' : ''} />
                    </div>
                </div>
            </div>

            <div className="controls glass-panel">
                <div className="panel">
                    <div className="panel-header">
                        <h3>What the output neuron sees</h3>
                        <div className="btn-row">
                            <button className="btn btn-secondary" type="button" onClick={playMorph}><Play size={16} /> Move the points</button>
                            <button className="btn btn-secondary" type="button" onClick={() => { cancelAnimationFrame(raf.current); setT(1); }}>
                                <RotateCcw size={16} /> End
                            </button>
                        </div>
                    </div>
                    <div className="slider-container in-row">
                        <div className="slider-header">
                            <span>each point: from its inputs <Sym b="x" vec /> to the hidden answers <Sym b="a" vec l={1} /> = (a₁, a₂)</span>
                            <span className="in-val">{num(t)}</span>
                        </div>
                        <input type="range" min={0} max={1} step={0.01} value={t} onChange={(e) => { cancelAnimationFrame(raf.current); setT(Number(e.target.value)); }} />
                    </div>
                    <Plot data={hidden} layout={{ ...planeLayout(pTheme, hx, hy, hTitle(0), hTitle(1)), annotations: hAnn, uirevision: `${act}` }}
                        config={PLOT_CONFIG} useResizeHandler style={{ width: '100%', height: '440px' }} />
                    <div className="info-card">
                        The hidden layer moves every point to (a₁, a₂): its two answers. The gray grid is the input plane, carried along.{' '}
                        {sig
                            ? <>It <strong>bends</strong>, because σ is not a straight line: that bending is the non-linearity. At the end, one
                                straight line, the output neuron’s, can separate the crosses from the circles.</>
                            : <>Without σ it only stretches and turns: straight lines stay straight, the points stay in their XOR pattern, and no
                                straight line can separate them.</>}
                    </div>
                </div>
                <div className="panel">
                    <h3>The network, for one point</h3>
                    <XorDiagram p={p} fw={big[sel]} x={x} act={act} focus={focus} onFocus={setFocus}
                        questions={[l1.question, l2.question, sig ? (lo.name ? `${lo.name} of the two answers` : lo.question) : 'no σ before it']} />
                    <table className="xor-table">
                        <thead>
                            <tr>
                                <th>x₁</th><th>x₂</th>
                                <th style={{ color: NEURON_COLORS[0] }}><Sym b="a" l={1} j={1} /></th>
                                <th style={{ color: NEURON_COLORS[1] }}><Sym b="a" l={1} j={2} /></th>
                                <th style={{ color: COLORS.go }}>f</th><th><Yhat /></th><th>y</th><th />
                            </tr>
                        </thead>
                        <tbody>
                            {XOR.map((pt, i) => {
                                const r = big[i];
                                return (
                                    <tr key={i} className={`${sel === i ? 'sel' : ''} ${hlRow === i ? 'hover' : ''}`}
                                        onMouseEnter={() => setHlRow(i)} onMouseLeave={() => setHlRow(null)} onClick={() => setSel(i)}>
                                        <td>{pt.x1}</td><td>{pt.x2}</td>
                                        <td>{num(r.a1)}</td><td>{num(r.a2)}</td><td>{num(r.f)}</td>
                                        <td>{r.yhat}</td><td>{pt.y}</td>
                                        <td className={r.yhat === pt.y ? 'ok' : 'bad'}>{r.yhat === pt.y ? '✓' : '✗'}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                    <div className="hint">Click a row to show that point on the network; point at a row to find it on both plots.</div>
                </div>
            </div>
        </>
    );
}

// Step 3: the XOR problem. One neuron cannot solve it; two hidden neurons and σ can.
export default function XorView({ pTheme }) {
    const [view, setView] = useState('two');
    return (
        <>
            <div className="glass-panel xor-intro">
                <div className="xor-intro-text">
                    <h3>XOR: one or the other, but not both</h3>
                    <div>
                        Two inputs, each 0 or 1, and y = 1 when <strong>exactly one</strong> of them is 1. Think of a corridor lamp with a
                        switch at each end: it is on when the two switches are in different positions.
                    </div>
                    <ViewTabs options={[{ key: 'one', label: '1 · One neuron is not enough' }, { key: 'two', label: '2 · Two hidden neurons' }]}
                        value={view} onChange={setView} />
                </div>
                <table className="xor-table truth">
                    <thead><tr><th>x₁</th><th>x₂</th><th>y</th></tr></thead>
                    <tbody>
                        {XOR.map((pt) => (
                            <tr key={`${pt.x1}${pt.x2}`}>
                                <td>{pt.x1}</td><td>{pt.x2}</td>
                                <td style={{ color: pt.y ? COLORS.pos : COLORS.neg, fontWeight: 700 }}>{pt.y} {pt.y ? '✕' : '●'}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            {view === 'one' ? <OneNeuron pTheme={pTheme} /> : <TwoNeurons pTheme={pTheme} />}
        </>
    );
}
