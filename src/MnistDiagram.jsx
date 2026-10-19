import React, { useEffect, useMemo, useRef, useState } from 'react';
import { MODEL, SIZE, N_IN, H1, H2, N_OUT, paint, weightsOf } from './mnist';
import { num } from './network';
import { Sym } from './Sym';
import { COLORS, actFill } from './ui';

// ── Geometry, in viewBox units (labels are HTML on top, sized with container units) ──
const VW = 1200;
const VH = 640;
const CELL = 6;
const IMG = { x: 24, y: 236, size: SIZE * CELL };
const N_SHOWN = 6;                       // inputs shown at each end of the column: x1 … x6 and x779 … x784
const IN_X = 330;
const R_IN = 13;
const FIRST = Array.from({ length: N_SHOWN }, (_, k) => k);
const LAST = Array.from({ length: N_SHOWN }, (_, k) => N_IN - N_SHOWN + k);
const inY = (slot) => (slot < N_SHOWN ? 70 + 40 * slot : 370 + 40 * (slot - N_SHOWN));
const PROBE_Y = 320;                     // a pixel pointed at gets its input here, between the dots
const L1 = Array.from({ length: H1 }, (_, j) => ({ x: 590, y: 50 + (j * 540) / (H1 - 1) }));
const L2 = Array.from({ length: H2 }, (_, j) => ({ x: 830, y: 50 + (j * 540) / (H2 - 1) }));
const OUT = Array.from({ length: N_OUT }, (_, d) => ({ x: 1035, y: 62 + (d * 516) / (N_OUT - 1) }));
const R_H = 12;
const R_O = 16;
const pct = (x, y) => ({ left: `${(100 * x) / VW}%`, top: `${(100 * y) / VH}%` });
const cellCenter = (i) => ({ x: IMG.x + ((i % SIZE) + 0.5) * CELL, y: IMG.y + (Math.floor(i / SIZE) + 0.5) * CELL });
const sub = (i) => <Sym b="x" j={i + 1} />;

// A line drawn by its signal (input × weight): orange pushes the next neuron up, blue holds it back,
// brighter and wider when larger (relative to the largest signal between the same two layers).
function signalLines(pairs, keyOf, dimmed) {
    const big = Math.max(1e-9, ...pairs.map((q) => Math.abs(q.s)));
    return pairs.map((q) => {
        const t = Math.abs(q.s) / big;
        const off = t < 0.02;
        return (
            <line key={keyOf(q)} x1={q.a.x} y1={q.a.y} x2={q.b.x} y2={q.b.y} strokeLinecap="round"
                strokeWidth={off ? 0.6 : 0.5 + 2.6 * t}
                style={{
                    stroke: off ? 'var(--edge-off)' : q.s > 0 ? COLORS.pos : COLORS.neg,
                    strokeOpacity: (off ? 0.5 : 0.08 + 0.92 * t ** 0.8) * (dimmed(q) ? 0.12 : 1),
                }} />
        );
    });
}

// A 28 × 28 image (a digit, or the weights of a neuron) in a canvas, scaled up without blur.
export function TinyImage({ values, mode = 'gray', dark = true, size = 56, className = '' }) {
    const ref = useRef(null);
    useEffect(() => paint(ref.current, values, mode, dark), [values, mode, dark]);
    return <canvas ref={ref} width={SIZE} height={SIZE} className={`tiny-img ${className}`} style={{ width: size, height: size }} />;
}

// What a part of the network is, for the tooltip. hov: { type: 'x', i } | { type: 'h1' | 'h2', j } | { type: 'out', d }.
function Info({ hov, fw, dark }) {
    if (hov.type === 'x') {
        const r = Math.floor(hov.i / SIZE) + 1;
        const c = (hov.i % SIZE) + 1;
        return (
            <>
                <div className="tip-title">{sub(hov.i)} = {num(fw.x[hov.i])}</div>
                <div>the pixel in row {r}, column {c}: input number 28 × ({r} − 1) + {c} = {hov.i + 1}</div>
                <div className="tip-note">0 is black, 1 is white. It goes into all {H1} neurons of layer 1.</div>
            </>
        );
    }
    if (hov.type === 'h1') {
        return (
            <>
                <div className="tip-title"><Sym b="a" l={1} j={hov.j + 1} /> = {num(fw.a1[hov.j])}</div>
                <div>neuron {hov.j + 1} of layer 1: its 784 weights, one per pixel</div>
                <div className="tip-img">
                    <TinyImage values={weightsOf(hov.j)} mode="signed" dark={dark} size={112} />
                    <div className="tip-note">
                        <span style={{ color: COLORS.pos }}>Orange</span> pixels push it up when they are white,{' '}
                        <span style={{ color: COLORS.neg }}>blue</span> ones hold it back.
                    </div>
                </div>
            </>
        );
    }
    if (hov.type === 'h2') {
        const ws = MODEL.W3[hov.j];
        const up = ws.indexOf(Math.max(...ws));
        const down = ws.indexOf(Math.min(...ws));
        return (
            <>
                <div className="tip-title"><Sym b="a" l={2} j={hov.j + 1} /> = {num(fw.a2[hov.j])}</div>
                <div>neuron {hov.j + 1} of layer 2: it combines the 16 neurons of layer 1</div>
                <div className="tip-note">
                    When it is active it pushes most towards the digit {up} (w = {num(ws[up])}) and most against the
                    digit {down} (w = {num(ws[down])}).
                </div>
            </>
        );
    }
    return (
        <>
            <div className="tip-title">digit {hov.d}: {(100 * fw.p[hov.d]).toFixed(1)} %</div>
            <div>neuron {hov.d + 1} of layer 3, <Sym b="a" l={3} j={hov.d + 1} /> = {num(fw.p[hov.d], 3)}</div>
            <div className="tip-note">The 10 outputs are chances that add up to 1 (softmax); the largest is the answer.</div>
        </>
    );
}

// The digits network: the image (as one block, or expanded into its 784 inputs), 16 + 16 hidden neurons
// and 10 outputs, lit by their activations. hov / onHover: the part pointed at, shared with the page.
export default function MnistDiagram({ fw, expanded, hov, onHover, dark }) {
    const wrap = useRef(null);
    const [tip, setTip] = useState(null);
    const imageURL = useMemo(() => {
        const c = document.createElement('canvas');
        c.width = SIZE;
        c.height = SIZE;
        paint(c, fw.x, 'gray');
        return c.toDataURL();
    }, [fw.x]);

    const where = (e) => {
        const r = wrap.current.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height };
    };
    const show = (item, e) => {
        setTip({ ...where(e) });
        onHover(item);
    };
    const hide = () => {
        setTip(null);
        onHover(null);
    };
    const hit = (item) => ({ onMouseEnter: (e) => show(item, e), onMouseMove: (e) => setTip(where(e)), onMouseLeave: hide });
    const onImage = (e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const c = Math.min(SIZE - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * SIZE)));
        const rr = Math.min(SIZE - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * SIZE)));
        show({ type: 'x', i: rr * SIZE + c }, e);
    };

    // The inputs drawn as neurons: the first and the last ones, and the pixel pointed at (if it is not one of them).
    const probe = hov?.type === 'x' && !FIRST.includes(hov.i) && !LAST.includes(hov.i) ? hov.i : null;
    const inputs = [...FIRST, ...LAST].map((i, slot) => ({ i, x: IN_X, y: inY(slot) }));
    if (probe != null) inputs.push({ i: probe, x: IN_X, y: PROBE_Y, probe: true });

    // What is lit when something is pointed at: the neuron and its lines; the rest fades.
    const focusOn = (layer, j) => hov && ((layer === 'h1' && hov.type === 'h1' && hov.j === j) || (layer === 'h2' && hov.type === 'h2' && hov.j === j)
        || (layer === 'out' && hov.type === 'out' && hov.d === j));
    const anyNeuron = hov && hov.type !== 'x';
    const fade = (layer, j) => (anyNeuron && !focusOn(layer, j) ? 0.25 : 1);
    const lineDim = (from, j, to, k) => anyNeuron && !focusOn(from, j) && !focusOn(to, k);

    const inLines = expanded ? signalLines(
        inputs.flatMap((p) => L1.map((b, j) => ({ a: { x: p.x + R_IN, y: p.y }, b, s: fw.x[p.i] * MODEL.W1[p.i][j], i: p.i, j }))),
        (q) => `in${q.i}-${q.j}`, (q) => (hov?.type === 'x' ? hov.i !== q.i : lineDim('x', q.i, 'h1', q.j)),
    ) : null;
    const l12 = signalLines(
        L1.flatMap((a, j) => L2.map((b, k) => ({ a, b, s: fw.a1[j] * MODEL.W2[j][k], j, k }))),
        (q) => `h${q.j}-${q.k}`, (q) => lineDim('h1', q.j, 'h2', q.k),
    );
    const l23 = signalLines(
        L2.flatMap((a, j) => OUT.map((b, d) => ({ a, b, s: fw.a2[j] * MODEL.W3[j][d], j, k: d }))),
        (q) => `o${q.j}-${q.k}`, (q) => lineDim('h2', q.j, 'out', q.k),
    );
    const scale = MODEL.scale;
    const hiCell = hov?.type === 'x' ? cellCenter(hov.i) : null;

    const tipStyle = tip && {
        left: tip.w - tip.x > 340 ? tip.x + 16 : tip.x - 16,
        top: tip.y < tip.h * 0.55 ? tip.y + 16 : tip.y - 16,
        transform: `translate(${tip.w - tip.x > 340 ? '0' : '-100%'}, ${tip.y < tip.h * 0.55 ? '0' : '-100%'})`,
    };

    return (
        <div className="net-scroll">
            <div className="net mnist-net" ref={wrap}>
                <svg viewBox={`0 0 ${VW} ${VH}`} className="net-svg" role="img"
                    aria-label="The digits network: a 28 by 28 image, two hidden layers of 16 neurons and 10 outputs">
                    <g className="net-part" style={{ opacity: expanded ? 0 : 1, pointerEvents: expanded ? 'none' : 'auto' }}>
                        {L1.map((b, j) => (
                            <line key={j} x1={IMG.x + IMG.size} y1={IMG.y + IMG.size / 2} x2={b.x - R_H} y2={b.y}
                                strokeWidth={0.6 + 2.2 * Math.min(1, fw.a1[j] / scale[0][j])}
                                style={{ stroke: COLORS.act, strokeOpacity: (0.12 + 0.75 * Math.min(1, fw.a1[j] / scale[0][j])) * fade('h1', j) }} />
                        ))}
                    </g>
                    <g className="net-part" style={{ opacity: expanded ? 1 : 0, pointerEvents: expanded ? 'auto' : 'none' }}>
                        {inLines}
                        {[...FIRST.map((i, s) => [i, s]), ...LAST.map((i, s) => [i, N_SHOWN + s])].map(([i, s]) => {
                            const c = cellCenter(i);
                            return (
                                <path key={i} d={`M ${c.x} ${c.y} C ${c.x + 70} ${c.y}, ${IN_X - 90} ${inY(s)}, ${IN_X - R_IN} ${inY(s)}`}
                                    className="pixel-link" />
                            );
                        })}
                        {(probe == null ? [304, 312, 320] : [289, 295, 301, 339, 345, 351]).map((y) => (
                            <circle key={y} cx={IN_X} cy={y} r={probe == null ? 2.4 : 1.9} className="dot" />
                        ))}
                        {probe != null && (() => {
                            const c = cellCenter(probe);
                            return <path d={`M ${c.x} ${c.y} C ${c.x + 70} ${c.y}, ${IN_X - 90} ${PROBE_Y}, ${IN_X - R_IN} ${PROBE_Y}`} className="pixel-link hot" />;
                        })()}
                        {inputs.map((p) => {
                            const v = fw.x[p.i];
                            return (
                                <g key={`n${p.i}`} className="net-node" {...hit({ type: 'x', i: p.i })}>
                                    {hov?.type === 'x' && hov.i === p.i && <circle cx={p.x} cy={p.y} r={R_IN + 6} className="glow-fill" />}
                                    <circle cx={p.x} cy={p.y} r={R_IN} strokeWidth={1.8} style={{ fill: actFill(v), stroke: 'var(--node-edge)' }} />
                                </g>
                            );
                        })}
                    </g>
                    {l12}
                    {l23}

                    <image href={imageURL} x={IMG.x} y={IMG.y} width={IMG.size} height={IMG.size} className="pixelated" />
                    <rect x={IMG.x} y={IMG.y} width={IMG.size} height={IMG.size} className="img-frame"
                        onMouseEnter={onImage} onMouseMove={onImage} onMouseLeave={hide} />
                    {hiCell && <rect x={hiCell.x - CELL / 2 - 1} y={hiCell.y - CELL / 2 - 1} width={CELL + 2} height={CELL + 2} className="pixel-hl" />}

                    {[[L1, 'h1', scale[0], fw.a1], [L2, 'h2', scale[1], fw.a2]].map(([layer, name, sc, acts]) => layer.map((c, j) => (
                        <g key={`${name}${j}`} className="net-part net-node" style={{ opacity: fade(name, j) }} {...hit({ type: name, j })}>
                            {focusOn(name, j) && <circle cx={c.x} cy={c.y} r={R_H + 6} className="glow-fill" />}
                            <circle cx={c.x} cy={c.y} r={R_H} fill={actFill(acts[j] / sc[j])} stroke={COLORS.act} strokeWidth={2} />
                        </g>
                    )))}
                    {OUT.map((c, d) => (
                        <g key={`o${d}`} className="net-part net-node" style={{ opacity: fade('out', d) }} {...hit({ type: 'out', d })}>
                            {(d === fw.best || focusOn('out', d)) && <circle cx={c.x} cy={c.y} r={R_O + 6} className={d === fw.best ? 'best-ring' : 'glow-fill'} />}
                            <circle cx={c.x} cy={c.y} r={R_O} fill={actFill(fw.p[d])} stroke={COLORS.act} strokeWidth={2.4} />
                            <text x={c.x} y={c.y + 5} textAnchor="middle" className="net-val" style={{ fill: fw.p[d] > 0.6 ? '#fff' : 'var(--text-color)' }}>{d}</text>
                        </g>
                    ))}
                </svg>

                {/* ── labels ── */}
                <div className="net-lab center muted" style={pct(IMG.x + IMG.size / 2, IMG.y - 22)}>28 × 28 pixels</div>
                {expanded && (
                    <>
                        <div className="net-lab center small-lab" style={pct(IMG.x + IMG.size / 2, IMG.y - 48)}>
                            row 1 → {sub(0)} … {sub(27)}
                        </div>
                        <div className="net-lab center small-lab" style={pct(IMG.x + IMG.size / 2, IMG.y + IMG.size + 22)}>
                            row 28 → {sub(756)} … {sub(783)}
                        </div>
                        {inputs.map((p) => (
                            <div key={p.i} className={`net-lab left ${p.probe ? 'probe-lab' : ''}`} style={pct(p.x + R_IN + 6, p.y)}>{sub(p.i)}</div>
                        ))}
                        {probe == null && (
                            <div className="net-lab center small-lab muted" style={pct(IN_X, 341)}>{N_IN - 2 * N_SHOWN} more</div>
                        )}
                    </>
                )}
                {!expanded && (
                    <div className="net-lab center small-lab muted" style={pct(380, 330)}>784 × 16 weights</div>
                )}
                {[[expanded ? IN_X : IMG.x + IMG.size / 2, expanded ? '784 inputs' : 'input', 0], [L1[0].x, `${H1} neurons`, 1],
                    [L2[0].x, `${H2} neurons`, 2], [OUT[0].x, '10 outputs', 3]].map(([x, name, l]) => (
                    <div key={l} className="net-lab center" style={pct(x, 16)}>
                        <span className="n-layer strong">layer [{l}]</span> <span className="muted">{name}</span>
                    </div>
                ))}
                {OUT.map((c, d) => (
                    <div key={d} className={`net-lab left out-bar ${d === fw.best ? 'best' : ''}`} style={pct(c.x + R_O + 10, c.y)}>
                        <span className="bar"><span style={{ width: `${100 * fw.p[d]}%` }} /></span>
                        <span className="pc">{(100 * fw.p[d]).toFixed(fw.p[d] > 0.995 ? 0 : 1)} %</span>
                    </div>
                ))}

                {tip && hov && (
                    <div className="tooltip" style={tipStyle}>
                        <Info hov={hov} fw={fw} dark={dark} />
                    </div>
                )}
            </div>
        </div>
    );
}
