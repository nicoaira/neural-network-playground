import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Play, Pause, StepForward, Shuffle, Plus, Minus } from 'lucide-react';
import { DATASETS, RANGE, makeData, Net, contour } from './builder';
import { Sym, Yhat } from './Sym';
import { COLORS, Toggle, ViewTabs } from './ui';

const N = 50;                         // the maps are computed on an N × N grid of the plane
const MAX_LAYERS = 4;
const MAX_NEURONS = 8;
const LRS = [0.003, 0.01, 0.03, 0.1, 0.3];
const FAST_STEPS = 6;                 // gradient descent steps per frame (1 in slow motion)
const HISTORY = 400;                  // points kept on the cost curve

// ── The drawing of the network (CSS pixels) ──
const DW = 660;
const DH = 500;
const BOX = 40;
const BOX_OUT = 64;

function layoutOf(sizes) {
    const L = sizes.length;
    return sizes.map((n, c) => {
        const x = 70 + (c * (DW - 190)) / (L - 1);
        const s = c === L - 1 ? BOX_OUT : BOX;
        const sp = Math.min(54, 380 / Math.max(n, 1));
        return Array.from({ length: n }, (_, j) => ({ x, y: 270 + (j - (n - 1) / 2) * sp, s }));
    });
}

// ── Colors of the maps ──
const RGB = { pos: [249, 115, 22], neg: [59, 130, 246], act: [16, 185, 129] };
const mix = (bg, c, t) => bg.map((v, i) => Math.round(v + (c[i] - v) * t));
function mapPixels(map, kind, dark) {
    const bg = dark ? [15, 23, 42] : [248, 250, 252];
    const px = new Uint8ClampedArray(4 * map.length);
    let big = 0;
    if (kind !== 'output') for (const v of map) big = Math.max(big, Math.abs(v));
    for (let i = 0; i < map.length; i++) {
        let rgb;
        if (kind === 'output') {                       // f: blue (0) … background (0.5) … orange (1)
            const t = 2 * map[i] - 1;
            rgb = mix(bg, t > 0 ? RGB.pos : RGB.neg, 0.85 * Math.abs(t) ** 0.8);
        } else if (kind === 'input') {                 // x₁ or x₂: blue (negative) … orange (positive)
            const t = map[i] / RANGE;
            rgb = mix(bg, t > 0 ? RGB.pos : RGB.neg, 0.75 * Math.abs(t));
        } else {                                       // a hidden neuron: brighter green = more active
            rgb = mix(bg, RGB.act, big > 0 ? 0.9 * (map[i] / big) : 0);
        }
        px.set([...rgb, 255], 4 * i);
    }
    return new ImageData(px, N, N);
}
const kindOf = (l, L) => (l === 0 ? 'input' : l === L - 1 ? 'output' : 'hidden');

// Draw an N × N map into a rectangle of a canvas, smoothly scaled up.
function drawMap(ctx, scratch, map, kind, dark, x, y, w, h) {
    scratch.getContext('2d').putImageData(mapPixels(map, kind, dark), 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(scratch, x, y, w, h);
}

function useCanvas(width, height) {
    const ref = useRef(null);
    const ctx = () => {
        const c = ref.current;
        if (!c) return null;
        const dpr = window.devicePixelRatio || 1;
        if (c.width !== width * dpr) {
            c.width = width * dpr;
            c.height = height * dpr;
        }
        const g = c.getContext('2d');
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        return g;
    };
    return [ref, ctx];
}

const distToSeg = (px, py, x1, y1, x2, y2) => {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
};

// ── The plane: the map of the output (or of the neuron pointed at), the boundary and the points ──
const PM = { l: 36, t: 10, size: 360 };
const PW = PM.l + PM.size + 12;
const PH = PM.t + PM.size + 34;
const toPx = (x1, x2) => [PM.l + ((x1 + RANGE) / (2 * RANGE)) * PM.size, PM.t + ((RANGE - x2) / (2 * RANGE)) * PM.size];

export default function BuilderView({ dark }) {
    const [dataKey, setDataKey] = useState('moons');
    const [noise, setNoise] = useState(0.1);
    const [hidden, setHidden] = useState([4]);
    const [lr, setLr] = useState(0.03);
    const [slow, setSlow] = useState(false);
    const [playing, setPlaying] = useState(false);
    const [showTest, setShowTest] = useState(false);
    const [showMistakes, setShowMistakes] = useState(true);
    const [hover, setHover] = useState(null); // { type: 'n', l, j } a neuron | { type: 'w', l, j, k } a weight
    const [tip, setTip] = useState(null);
    const [seed, setSeed] = useState(2);    // the random start of the weights (2: the first one that fits the moons with 4 neurons)
    const [stats, setStats] = useState(null); // { epoch, train, test }
    const [history, setHistory] = useState([]);

    const sizes = useMemo(() => [2, ...hidden, 1], [hidden]);
    const data = useMemo(() => makeData(dataKey, noise), [dataKey, noise]);
    const net = useRef(null);
    const maps = useRef(null);
    const epoch = useRef(0);
    const hist = useRef([]);
    const lrRef = useRef(lr);
    lrRef.current = lr;
    const scratch = useMemo(() => {
        const c = document.createElement('canvas');
        c.width = N;
        c.height = N;
        return c;
    }, []);
    const [netRef, netCtx] = useCanvas(DW, DH);
    const [planeRef, planeCtx] = useCanvas(PW, PH);

    const measure = () => {
        const tr = net.current.evaluate(data.train);
        const te = net.current.evaluate(data.test);
        setStats({ epoch: epoch.current, train: tr, test: te });
        return { tr, te };
    };
    const record = () => {
        const { tr, te } = measure();
        hist.current.push({ epoch: epoch.current, Jtr: tr.J, Jte: te.J });
        if (hist.current.length > HISTORY) hist.current = hist.current.filter((_, i) => i % 2 === 0 || i === hist.current.length - 1);
        setHistory([...hist.current]);
    };

    // A new network (new random weights) whenever the pattern, the layers or the random start change.
    useEffect(() => {
        setPlaying(false);
        net.current = new Net(sizes, seed);
        epoch.current = 0;
        hist.current = [];
        maps.current = net.current.grid(N);
        record();
        draw();
    }, [sizes, seed, data]); // eslint-disable-line react-hooks/exhaustive-deps

    // Training: some gradient descent steps every frame, then everything is redrawn.
    useEffect(() => {
        if (!playing) return undefined;
        let raf;
        let last = 0;
        const loop = (now) => {
            const steps = slow ? 1 : FAST_STEPS;
            for (let s = 0; s < steps; s++) net.current.step(data.train, lrRef.current);
            epoch.current += steps;
            maps.current = net.current.grid(N);
            drawRef.current();
            if (now - last > 120) {
                last = now;
                record();
            }
            raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
    }, [playing, slow, data]); // eslint-disable-line react-hooks/exhaustive-deps

    const oneStep = () => {
        setPlaying(false);
        net.current.step(data.train, lrRef.current);
        epoch.current += 1;
        maps.current = net.current.grid(N);
        record();
        draw();
    };

    // ── drawing ──
    const lay = useMemo(() => layoutOf(sizes), [sizes]);
    const draw = () => {
        if (!net.current || !maps.current) return;
        drawNetwork();
        drawPlane();
    };
    const drawRef = useRef(draw);
    drawRef.current = draw;                 // the training loop always draws with the latest hover, toggles and theme
    useEffect(draw); // after every render (hover, theme, toggles…)

    function drawNetwork() {
        const g = netCtx();
        if (!g) return;
        const { W } = net.current;
        const L = sizes.length;
        g.clearRect(0, 0, DW, DH);
        const big = Math.max(1e-9, ...W.map((w) => Math.max(...Array.from(w, Math.abs))));
        for (let l = 0; l < L - 1; l++) {
            const nOut = sizes[l + 1];
            lay[l].forEach((a, k) => lay[l + 1].forEach((b, j) => {
                const w = W[l][k * nOut + j];
                const t = Math.abs(w) / big;
                const on = hover?.type === 'w' && hover.l === l + 1 && hover.j === j + 1 && hover.k === k + 1;
                g.strokeStyle = w > 0 ? COLORS.pos : COLORS.neg;
                g.globalAlpha = on ? 1 : 0.18 + 0.82 * t;
                g.lineWidth = on ? 5 : 0.6 + 3.4 * t;
                g.beginPath();
                g.moveTo(a.x + a.s / 2, a.y);
                g.lineTo(b.x - b.s / 2, b.y);
                g.stroke();
            }));
        }
        g.globalAlpha = 1;
        lay.forEach((col, l) => col.forEach((p, j) => {
            drawMap(g, scratch, maps.current[l][j], kindOf(l, L), dark, p.x - p.s / 2, p.y - p.s / 2, p.s, p.s);
            const on = hover?.type === 'n' && hover.l === l && hover.j === j + 1;
            g.strokeStyle = on ? '#facc15' : dark ? '#94a3b8' : '#64748b';
            g.lineWidth = on ? 3 : 1.2;
            g.strokeRect(p.x - p.s / 2, p.y - p.s / 2, p.s, p.s);
        }));
    }

    function drawPlane() {
        const g = planeCtx();
        if (!g) return;
        const L = sizes.length;
        const shown = hover?.type === 'n' ? hover : { l: L - 1, j: 1 };
        g.clearRect(0, 0, PW, PH);
        drawMap(g, scratch, maps.current[shown.l][shown.j - 1], kindOf(shown.l, L), dark, PM.l, PM.t, PM.size, PM.size);
        // the decision boundary: where f = 0.5
        const cell = PM.size / N;
        g.strokeStyle = dark ? '#f8fafc' : '#0f172a';
        g.lineWidth = 2;
        g.beginPath();
        for (const [x0, y0, x1, y1] of contour(maps.current[L - 1][0], N)) {
            g.moveTo(PM.l + (x0 + 0.5) * cell, PM.t + (y0 + 0.5) * cell);
            g.lineTo(PM.l + (x1 + 0.5) * cell, PM.t + (y1 + 0.5) * cell);
        }
        g.stroke();
        // axes
        g.strokeStyle = dark ? '#475569' : '#cbd5e1';
        g.lineWidth = 1;
        g.strokeRect(PM.l, PM.t, PM.size, PM.size);
        g.fillStyle = dark ? '#cbd5e1' : '#334155';
        g.font = '12px Inter, system-ui, sans-serif';
        g.textAlign = 'center';
        for (const v of [-1, 0, 1]) {
            const [x] = toPx(v, 0);
            g.fillText(String(v).replace('-', '−'), x, PM.t + PM.size + 15);
            const [, y] = toPx(0, v);
            g.textAlign = 'right';
            g.fillText(String(v).replace('-', '−'), PM.l - 6, y + 4);
            g.textAlign = 'center';
        }
        g.font = 'italic 14px "STIX Two Text", "Times New Roman", serif';
        g.fillText('x₁', PM.l + PM.size / 2, PM.t + PM.size + 30);
        g.save();
        g.translate(12, PM.t + PM.size / 2);
        g.rotate(-Math.PI / 2);
        g.fillText('x₂', 0, 0);
        g.restore();
        // the points
        const pts = [...data.train.map((p) => ({ ...p, test: false })), ...(showTest ? data.test.map((p) => ({ ...p, test: true })) : [])];
        for (const p of pts) {
            const [x, y] = toPx(p.x1, p.x2);
            if (x < PM.l || x > PM.l + PM.size || y < PM.t || y > PM.t + PM.size) continue;
            const color = p.y ? COLORS.pos : COLORS.neg;
            if (p.y) {                              // an orange cross (a white outline for the training points)
                for (const pass of p.test ? [color] : ['#ffffff', color]) {
                    g.strokeStyle = pass;
                    g.lineWidth = pass === '#ffffff' ? 4.2 : 2;
                    g.beginPath();
                    g.moveTo(x - 4, y - 4); g.lineTo(x + 4, y + 4);
                    g.moveTo(x + 4, y - 4); g.lineTo(x - 4, y + 4);
                    g.stroke();
                }
            } else {
                g.beginPath();
                g.arc(x, y, 4, 0, 2 * Math.PI);
                if (p.test) {
                    g.strokeStyle = color;
                    g.lineWidth = 2;
                    g.stroke();
                } else {
                    g.fillStyle = color;
                    g.fill();
                    g.strokeStyle = '#ffffff';
                    g.lineWidth = 1;
                    g.stroke();
                }
            }
            if (showMistakes) {
                const f = net.current.forward(p.x1, p.x2);
                if ((f >= 0.5 ? 1 : 0) !== p.y) {
                    g.strokeStyle = '#ef4444';
                    g.lineWidth = 1.8;
                    g.beginPath();
                    g.arc(x, y, 8, 0, 2 * Math.PI);
                    g.stroke();
                }
            }
        }
    }

    // ── pointing at the network ──
    const onNetMove = (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const px = ((e.clientX - r.left) * DW) / r.width;
        const py = ((e.clientY - r.top) * DH) / r.height;
        let found = null;
        lay.forEach((col, l) => col.forEach((p, j) => {
            if (Math.abs(px - p.x) <= p.s / 2 + 2 && Math.abs(py - p.y) <= p.s / 2 + 2) found = { type: 'n', l, j: j + 1 };
        }));
        if (!found) {
            let best = 6;
            for (let l = 0; l < sizes.length - 1; l++) {
                lay[l].forEach((a, k) => lay[l + 1].forEach((b, j) => {
                    const d = distToSeg(px, py, a.x + a.s / 2, a.y, b.x - b.s / 2, b.y);
                    if (d < best) {
                        best = d;
                        found = { type: 'w', l: l + 1, j: j + 1, k: k + 1 };
                    }
                }));
            }
        }
        setHover(found);
        setTip(found ? { x: e.clientX - r.left, y: e.clientY - r.top } : null);
    };

    const L = sizes.length;
    const tipInfo = () => {
        if (!hover) return null;
        if (hover.type === 'w') {
            const w = net.current.W[hover.l - 1][(hover.k - 1) * sizes[hover.l] + hover.j - 1];
            const from = hover.l === 1 ? <Sym b="x" j={hover.k} /> : <>neuron {hover.k} of layer {hover.l - 1}</>;
            const to = hover.l === L - 1 ? 'the output neuron' : `neuron ${hover.j} of layer ${hover.l}`;
            return (
                <>
                    <div className="tip-title"><Sym b="w" l={hover.l} j={hover.j} k={hover.k} /> = {w.toFixed(2).replace('-', '−')}</div>
                    <div>the weight from {from} into {to}</div>
                </>
            );
        }
        if (hover.l === 0) {
            return (
                <>
                    <div className="tip-title">input <Sym b="x" j={hover.j} /></div>
                    <div>{hover.j === 1 ? 'grows from left to right' : 'grows from bottom to top'} (orange: positive, blue: negative)</div>
                </>
            );
        }
        if (hover.l === L - 1) {
            return (
                <>
                    <div className="tip-title">the output neuron, f = <Sym b="a" l={hover.l} j={1} /></div>
                    <div>the chance of orange, from 0 (blue) to 1 (orange); <Yhat /> = 1 where f ≥ 0.5</div>
                </>
            );
        }
        return (
            <>
                <div className="tip-title"><Sym b="a" l={hover.l} j={hover.j} />: neuron {hover.j} of layer {hover.l}</div>
                <div>its activation over the whole plane: brighter green = more active. It is shown large on the right.</div>
            </>
        );
    };

    const setNeurons = (i, delta) => setHidden(hidden.map((n, k) => (k === i ? Math.min(MAX_NEURONS, Math.max(1, n + delta)) : n)));
    const setLayers = (delta) => setHidden(delta > 0 ? [...hidden, hidden[hidden.length - 1] ?? 4].slice(0, MAX_LAYERS) : hidden.slice(0, -1));
    const shownName = hover?.type === 'n' && hover.l > 0 && hover.l < L - 1
        ? <>neuron {hover.j} of layer {hover.l}: <Sym b="a" l={hover.l} j={hover.j} /></>
        : hover?.type === 'n' && hover.l === 0 ? <>input <Sym b="x" j={hover.j} /></> : <>the answer of the network, f (the line: f = 0.5)</>;

    // ── the cost curve ──
    const CW = 380;
    const CH = 130;
    const maxE = Math.max(10, ...history.map((h) => h.epoch));
    const maxJ = Math.max(0.8, ...history.map((h) => Math.max(h.Jtr, h.Jte)));
    const cx = (e) => 34 + (e / maxE) * (CW - 44);
    const cy = (J) => 10 + (1 - Math.min(J, maxJ) / maxJ) * (CH - 36);
    const line = (key) => history.map((h) => `${cx(h.epoch).toFixed(1)},${cy(h[key]).toFixed(1)}`).join(' ');
    const pct = (v) => `${(100 * v).toFixed(1)} %`;

    return (
        <>
            <div className="glass-panel builder-intro">
                <h3>Build your own network</h3>
                <div>
                    Choose a pattern, design the network (how many hidden layers, how many neurons in each) and press <strong>Train</strong>.
                    The network learns from the training points by gradient descent, here and now; the test points are only used to check it.
                    Start small: with no hidden layer the network is one neuron, and its boundary is a straight line.
                </div>
            </div>

            <div className="glass-panel builder-controls">
                <div className="ctl">
                    <span className="ctl-lab">Pattern</span>
                    <div className="preset-row">
                        {DATASETS.map((d) => (
                            <button key={d.key} type="button" className={`chip ${dataKey === d.key ? 'active' : ''}`} onClick={() => setDataKey(d.key)}>
                                {d.name}
                            </button>
                        ))}
                    </div>
                </div>
                <div className="ctl ctl-slider">
                    <span className="ctl-lab">Noise <span className="in-val">{noise.toFixed(2)}</span></span>
                    <input type="range" min={0} max={0.3} step={0.05} value={noise} onChange={(e) => setNoise(Number(e.target.value))} aria-label="noise" />
                </div>
                <div className="ctl">
                    <span className="ctl-lab">Learning rate α</span>
                    <ViewTabs options={LRS.map((v) => ({ key: v, label: String(v) }))} value={lr} onChange={setLr} />
                </div>
                <div className="ctl ctl-buttons">
                    <button className="btn" type="button" onClick={() => setPlaying(!playing)}>
                        {playing ? <Pause size={18} /> : <Play size={18} />} {playing ? 'Pause' : 'Train'}
                    </button>
                    <button className="btn btn-secondary" type="button" onClick={oneStep}><StepForward size={16} /> One step</button>
                    <button className="btn btn-secondary" type="button" onClick={() => setSeed(seed + 1)}><Shuffle size={16} /> New random start</button>
                    <Toggle checked={slow} onChange={setSlow}>Slow motion</Toggle>
                </div>
            </div>

            <div className="glass-panel builder-grid">
                <div className="panel">
                    <div className="panel-header">
                        <h3>The network</h3>
                        <div className="layer-ctl">
                            <span className="muted small">hidden layers</span>
                            <button type="button" className="round-btn" onClick={() => setLayers(-1)} disabled={hidden.length === 0} aria-label="remove a hidden layer"><Minus size={14} /></button>
                            <strong>{hidden.length}</strong>
                            <button type="button" className="round-btn" onClick={() => setLayers(1)} disabled={hidden.length >= MAX_LAYERS} aria-label="add a hidden layer"><Plus size={14} /></button>
                        </div>
                    </div>
                    <div className="net-scroll">
                        <div className="builder-net" style={{ width: DW, height: DH }}>
                            <canvas ref={netRef} style={{ width: DW, height: DH }} onMouseMove={onNetMove}
                                onMouseLeave={() => { setHover(null); setTip(null); }} />
                            {lay.map((col, l) => (
                                <div key={`h${l}`} className="layer-head" style={{ left: col[0]?.x ?? 0 }}>
                                    <div><span className="n-layer strong">layer [{l}]</span></div>
                                    {l > 0 && l < L - 1 ? (
                                        <div className="neuron-ctl">
                                            <button type="button" className="round-btn" onClick={() => setNeurons(l - 1, -1)} disabled={hidden[l - 1] <= 1} aria-label="remove a neuron"><Minus size={12} /></button>
                                            <span>{hidden[l - 1]}</span>
                                            <button type="button" className="round-btn" onClick={() => setNeurons(l - 1, 1)} disabled={hidden[l - 1] >= MAX_NEURONS} aria-label="add a neuron"><Plus size={12} /></button>
                                        </div>
                                    ) : <div className="muted small">{l === 0 ? 'inputs' : 'output'}</div>}
                                </div>
                            ))}
                            {lay[0].map((p, j) => (
                                <div key={`x${j}`} className="node-lab" style={{ left: p.x - p.s / 2 - 8, top: p.y }}><Sym b="x" j={j + 1} /></div>
                            ))}
                            <div className="node-lab right-of" style={{ left: lay[L - 1][0].x + BOX_OUT / 2 + 8, top: lay[L - 1][0].y }}>
                                f
                            </div>
                            {tip && hover && (
                                <div className="tooltip" style={{ left: tip.x + (tip.x > DW - 300 ? -16 : 16), top: tip.y + 16, transform: tip.x > DW - 300 ? 'translateX(-100%)' : 'none' }}>
                                    {tipInfo()}
                                </div>
                            )}
                        </div>
                    </div>
                    <div className="info-card">
                        Each square shows what that neuron computes over the whole plane: <span style={{ color: COLORS.go }}>brighter green</span> where it
                        is more active. The output square is the answer of the network: <span style={{ color: COLORS.pos }}>orange</span> where it says
                        “orange”, <span style={{ color: COLORS.neg }}>blue</span> where it says “blue”. Lines: <span style={{ color: COLORS.pos }}>orange</span>{' '}
                        weights are positive, <span style={{ color: COLORS.neg }}>blue</span> negative, thicker = larger. Point at a square to see it large, and at a line to read its weight.
                    </div>
                </div>

                <div className="panel">
                    <div className="panel-header">
                        <h3>The plane</h3>
                        <div className="toggle-row">
                            <Toggle checked={showTest} onChange={setShowTest} color={COLORS.muted}>test points</Toggle>
                            <Toggle checked={showMistakes} onChange={setShowMistakes} color="#ef4444">mistakes</Toggle>
                        </div>
                    </div>
                    <div className="plane-caption">{shownName}</div>
                    <canvas ref={planeRef} className="builder-plane" style={{ width: PW, height: PH }} />
                    <div className="metrics-grid builder-metrics">
                        <div className="metric-card"><div className="metric-value">{stats?.epoch ?? 0}</div><div className="metric-label" style={{ textTransform: 'none' }}>epochs</div></div>
                        <div className="metric-card"><div className="metric-value">{stats ? stats.train.J.toFixed(3) : '—'}</div><div className="metric-label" style={{ textTransform: 'none' }}>cost J (training points)</div></div>
                        <div className="metric-card"><div className="metric-value">{stats ? pct(stats.train.accuracy) : '—'}</div><div className="metric-label" style={{ textTransform: 'none' }}>right: training points</div></div>
                        <div className="metric-card"><div className="metric-value">{stats ? pct(stats.test.accuracy) : '—'}</div><div className="metric-label" style={{ textTransform: 'none' }}>right: test points</div></div>
                    </div>
                    <svg className="cost-curve" viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label="the cost J while training">
                        <line x1={34} y1={CH - 26} x2={CW - 10} y2={CH - 26} className="axis" />
                        <line x1={34} y1={10} x2={34} y2={CH - 26} className="axis" />
                        <text x={30} y={14} className="tick" textAnchor="end">{maxJ.toFixed(1)}</text>
                        <text x={30} y={CH - 26} className="tick" textAnchor="end">0</text>
                        <text x={CW - 10} y={CH - 12} className="tick" textAnchor="end">{maxE} epochs</text>
                        <text x={40} y={CH - 12} className="tick">cost J</text>
                        <polyline points={line('Jte')} className="curve test" />
                        <polyline points={line('Jtr')} className="curve train" />
                    </svg>
                    <div className="curve-legend small">
                        <span className="lg train" /> training points <span className="lg test" /> test points
                    </div>
                </div>
            </div>

            <div className="glass-panel builder-hints">
                <div className="info-card">
                    <strong>Things to try.</strong> With no hidden layer, the boundary is one straight line: fine for nothing here. <em>Two moons</em>:
                    one hidden layer of 4–8 neurons. <em>A circle inside a ring</em>: how few neurons can close the circle? <em>Two spirals</em>: two or three
                    layers of 8 neurons, and some patience. Press <strong>New random start</strong> to train the same network from other random
                    weights: small networks do not always end in the same place. Raise the noise and compare the training and the test points.
                </div>
            </div>
        </>
    );
}
