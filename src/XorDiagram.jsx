import React, { useRef, useState } from 'react';
import { num, wnum, par } from './network';
import { Sym, Yhat } from './Sym';
import { COLORS, actFill } from './ui';

// The 2-2-1 network of the XOR tab, every part hoverable. Neuron 1 and neuron 2 have the colors of their
// lines on the plots (amber, violet); the output neuron is green.
const VW = 690;
const VH = 300;
const IN = [{ x: 95, y: 95 }, { x: 95, y: 215 }];
const HID = [{ x: 330, y: 80 }, { x: 330, y: 220 }];
const OUT = { x: 535, y: 150 };
const R_IN = 22;
const R_H = 32;
const R_O = 36;
export const NEURON_COLORS = [COLORS.nice, COLORS.quiet, COLORS.go];

const seg = (a, ra, b, rb) => {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const L = Math.hypot(dx, dy);
    return { x1: a.x + (dx / L) * ra, y1: a.y + (dy / L) * ra, x2: b.x - (dx / L) * rb, y2: b.y - (dy / L) * rb };
};
const EDGES = [
    ...HID.flatMap((h, j) => IN.map((p, k) => ({ l: 1, j: j + 1, k: k + 1, ...seg(p, R_IN, h, R_H) }))),
    ...HID.map((h, k) => ({ l: 2, j: 1, k: k + 1, ...seg(h, R_H, OUT, R_O) })),
];
const pct = (x, y) => ({ left: `${(100 * x) / VW}%`, top: `${(100 * y) / VH}%` });

// p: the nine parameters; fw: the forward pass of the selected point; x: its inputs [x1, x2];
// focus: the neuron pointed at elsewhere (1, 2 or 3 for the output); onFocus: report the neuron pointed at here;
// questions: what each neuron answers, in words.
export default function XorDiagram({ p, fw, x, act, focus, onFocus, questions }) {
    const wrap = useRef(null);
    const [tip, setTip] = useState(null);
    const where = (e) => {
        const r = wrap.current.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width };
    };
    const hit = (info, neuron) => ({
        onMouseEnter: (e) => { setTip({ info, ...where(e) }); onFocus?.(neuron); },
        onMouseMove: (e) => { const q = where(e); setTip((t) => (t ? { ...t, ...q } : t)); },
        onMouseLeave: () => { setTip(null); onFocus?.(null); },
    });
    const dim = (n) => (focus && focus !== n ? 0.3 : 1);
    const wOf = (e) => p[`w${e.l}${e.j}${e.k}`];
    const srcOf = (e) => (e.l === 1 ? x[e.k - 1] : [fw.a1, fw.a2][e.k - 1]);
    const zs = [fw.z1, fw.z2, fw.z];
    const as = [fw.a1, fw.a2, fw.f];
    const neurons = [...HID.map((c, j) => ({ c, r: R_H, l: 1, j: j + 1, n: j + 1 })), { c: OUT, r: R_O, l: 2, j: 1, n: 3 }];
    const inSym = (l, k) => (l === 1 ? <Sym b="x" j={k} /> : <Sym b="a" l={1} j={k} />);

    return (
        <div className="net xor-net" ref={wrap}>
            <svg viewBox={`0 0 ${VW} ${VH}`} className="net-svg" role="img" aria-label="The XOR network: two inputs, two hidden neurons, one output neuron">
                {EDGES.map((e) => {
                    const w = wOf(e);
                    const width = Math.abs(w) < 0.05 ? 1.2 : 1.2 + 4 * Math.min(1, Math.abs(w) / 12);
                    const target = e.l === 1 ? e.j : 3;
                    return (
                        <g key={`${e.l}${e.j}${e.k}`} className="net-part" style={{ opacity: dim(target) }}>
                            <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} strokeWidth={width} strokeLinecap="round"
                                style={{ stroke: Math.abs(w) < 0.05 ? 'var(--edge-off)' : w > 0 ? COLORS.pos : COLORS.neg }} />
                            <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} className="hit" strokeWidth={14}
                                {...hit(
                                    <>
                                        <div className="tip-title"><Sym b="w" l={e.l} j={e.j} k={e.k} /> = {wnum(w)}</div>
                                        <div>weight from {inSym(e.l, e.k)} into {e.l === 1 ? `neuron ${e.j}` : 'the output neuron'}</div>
                                        <div className="tip-math">signal: {par(wnum(w), w)} × {num(srcOf(e))} = {num(w * srcOf(e))}</div>
                                    </>, target,
                                )} />
                        </g>
                    );
                })}
                {IN.map((c, k) => (
                    <g key={k} className="net-node">
                        <circle cx={c.x} cy={c.y} r={R_IN} strokeWidth={2} style={{ fill: actFill(x[k]), stroke: 'var(--node-edge)' }} />
                        <text x={c.x} y={c.y + 5} textAnchor="middle" className="net-val" style={{ fill: x[k] > 0.62 ? '#fff' : 'var(--text-color)' }}>
                            {num(x[k], 0)}
                        </text>
                    </g>
                ))}
                {neurons.map(({ c, r, l, j, n }) => {
                    const bias = p[`b${l}${j}`];
                    const ws = l === 1 ? [p[`w1${j}1`], p[`w1${j}2`]] : [p.w211, p.w212];
                    const a = as[n - 1];
                    const sig = l === 2 || act === 'sigmoid';
                    return (
                        <g key={n} className="net-part net-node" style={{ opacity: dim(n) }}
                            {...hit(
                                <>
                                    <div className="tip-title">{l === 1 ? `Neuron ${j} of layer 1` : 'The output neuron'}</div>
                                    <div className="tip-math">
                                        <Sym b="z" l={l} j={j} /> = {wnum(ws[0])} {inSym(l, 1)} + {par(wnum(ws[1]), ws[1])} {inSym(l, 2)} + {par(wnum(bias), bias)} = {num(zs[n - 1])}
                                    </div>
                                    <div className="tip-math">
                                        <Sym b="a" l={l} j={j} /> = {sig ? <>σ({num(zs[n - 1])})</> : <Sym b="z" l={l} j={j} />} = {num(a)}
                                    </div>
                                    <div className="tip-note">
                                        {l === 1 ? <>Its line on the input plane: {wnum(ws[0])} x₁ + {par(wnum(ws[1]), ws[1])} x₂ + {par(wnum(bias), bias)} = 0.</>
                                            : <>f, the answer of the network: <Yhat /> = {fw.yhat}.</>}
                                    </div>
                                </>, n,
                            )}>
                            <circle cx={c.x} cy={c.y} r={r} fill={sig ? actFill(Math.min(1, Math.max(0, a))) : 'var(--solid-bg)'}
                                stroke={NEURON_COLORS[n - 1]} strokeWidth={focus === n ? 5 : 3} />
                            <line x1={c.x} y1={c.y - r} x2={c.x} y2={c.y + r} stroke={NEURON_COLORS[n - 1]} strokeWidth={2} />
                            <text x={c.x - r * 0.45} y={c.y + r * 0.2} textAnchor="middle" className="net-glyph" style={{ fill: 'var(--text-color)', fontSize: r * 0.55 }}>Σ</text>
                            <text x={c.x + r * 0.45} y={c.y + r * 0.2} textAnchor="middle" className="net-glyph" style={{ fill: 'var(--text-color)', fontSize: r * 0.55 }}>
                                {sig ? 'σ' : '—'}
                            </text>
                        </g>
                    );
                })}
                <line x1={OUT.x + R_O + 3} y1={OUT.y} x2={OUT.x + R_O + 30} y2={OUT.y} stroke={COLORS.act} strokeWidth={3} />
                <path d={`M ${OUT.x + R_O + 42} ${OUT.y} L ${OUT.x + R_O + 28} ${OUT.y - 7} L ${OUT.x + R_O + 28} ${OUT.y + 7} Z`} fill={COLORS.act} />
            </svg>

            {IN.map((c, k) => (
                <div key={k} className="net-lab right" style={pct(c.x - R_IN - 8, c.y)}><Sym b="x" j={k + 1} /></div>
            ))}
            {HID.map((c, j) => (
                <React.Fragment key={j}>
                    <div className="net-lab center xor-q" style={{ ...pct(c.x, j === 0 ? c.y - R_H - 30 : c.y + R_H + 30), color: NEURON_COLORS[j], opacity: dim(j + 1) }}>
                        <div className="strong">neuron {j + 1}</div>
                        <div className="small-q">{questions[j]}</div>
                    </div>
                    <div className="net-lab left pill" style={{ ...pct(c.x + R_H + 6, c.y + (j === 0 ? -26 : 26)), opacity: dim(j + 1) }}>
                        <Sym b="a" l={1} j={j + 1} /> = {num(as[j])}
                    </div>
                </React.Fragment>
            ))}
            <div className="net-lab center xor-q" style={{ ...pct(OUT.x, OUT.y - R_O - 30), color: NEURON_COLORS[2], opacity: dim(3) }}>
                <div className="strong">output</div>
                <div className="small-q">{questions[2]}</div>
            </div>
            <div className="net-lab left out" style={pct(OUT.x + R_O + 46, OUT.y)}>
                <div>f = {num(fw.f)}</div>
                <div><Yhat /> = {fw.yhat}</div>
            </div>

            {tip && (
                <div className="tooltip" style={{
                    left: tip.w - tip.x > 330 ? tip.x + 16 : tip.x - 16, top: tip.y + 16,
                    transform: tip.w - tip.x > 330 ? 'none' : 'translateX(-100%)',
                }}>
                    {tip.info}
                </div>
            )}
        </div>
    );
}
