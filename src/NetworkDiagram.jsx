import React, { useRef, useState } from 'react';
import { INPUTS, HIDDEN, OUTPUT, W1, B1, W2, B2, towel, num, wnum, par } from './network';
import { Sym, Yhat } from './Sym';
import { COLORS, actFill } from './ui';

// ── Geometry, in viewBox units. The HTML labels on top are placed in % of the viewBox and sized with
// container units (cqw), so they scale with the drawing.
const VW = 1000;
const VH = 500;
const IN = INPUTS.map((_, k) => ({ x: 250, y: 64 + 58 * k }));
const HID = [{ x: 570, y: 162 }, { x: 570, y: 372 }];
const OUT = { x: 810, y: 267 };
const R_IN = 18;
const R_H = 46;
const R_O = 50;
const ARROW_END = OUT.x + R_O + 38;

// A connection from circle a (radius ra) to circle b (radius rb), from edge to edge.
const seg = (a, ra, b, rb) => {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const L = Math.hypot(dx, dy);
    return { x1: a.x + (dx / L) * ra, y1: a.y + (dy / L) * ra, x2: b.x - (dx / L) * rb, y2: b.y - (dy / L) * rb };
};

// Every connection, with its weight. frac: where its name is written (fraction of the way from its input).
const EDGES = [
    ...HID.flatMap((h, j) => IN.map((p, k) => ({
        item: { type: 'w', l: 1, j: j + 1, k: k + 1 }, w: W1[k][j], frac: 0.3, ...seg(p, R_IN, h, R_H),
    }))),
    ...HID.map((h, k) => ({ item: { type: 'w', l: 2, j: 1, k: k + 1 }, w: W2[k][0], frac: 0.5, ...seg(h, R_H, OUT, R_O) })),
].map((e) => ({ ...e, key: keyOf(e.item) }));

const NEURONS = [
    ...HID.map((c, j) => ({ c, r: R_H, l: 1, j: j + 1 })),
    { c: OUT, r: R_O, l: 2, j: 1 },
];

// Every part of the network is an "item": { type, l, j, k } with
//   w: the weight w^[l]_{j,k} (from input k into neuron j of layer l), x: the input x_k,
//   z: the Σ half of neuron j of layer l (its weighted sum), a: its σ half (its activation), b: its bias.
// Its key names it in the drawing: "w1:2:8", "x:3", "z1:2", "a2:1", "b1:1". An example number i is ignored.
export function keyOf(it) {
    if (it.type === 'w') return `w${it.l}:${it.j}:${it.k}`;
    if (it.type === 'x') return `x:${it.k}`;
    return `${it.type}${it.l}:${it.j}`;
}

// What to light up when an item is pointed at: the item itself (strong) and what it is connected to (rel).
export function lights(it) {
    if (!it) return null;
    const strong = new Set([keyOf(it)]);
    const rel = new Set();
    const neuron = (l, j) => [`z${l}:${j}`, `a${l}:${j}`];
    const add = (...keys) => keys.flat().forEach((key) => rel.add(key));
    if (it.type === 'w') {
        add(it.l === 1 ? `x:${it.k}` : neuron(1, it.k), neuron(it.l, it.j));
    } else if (it.type === 'x') {
        add(`w1:1:${it.k}`, `w1:2:${it.k}`);
    } else if (it.type === 'z') {
        const n = it.l === 1 ? INPUTS.length : HIDDEN.length;
        for (let k = 1; k <= n; k++) add(`w${it.l}:${it.j}:${k}`, it.l === 1 ? `x:${k}` : neuron(1, k));
        add(`b${it.l}:${it.j}`, `a${it.l}:${it.j}`);
    } else if (it.type === 'a') {
        add(`z${it.l}:${it.j}`, it.l === 1 ? `w2:1:${it.j}` : 'out');
    } else if (it.type === 'b') {
        add(neuron(it.l, it.j));
    }
    return { strong, rel };
}

const OPACITY = { normal: 1, strong: 1, rel: 1, dim: 0.28, faint: 0.1 };
const RANK = { strong: 4, rel: 3, normal: 2, dim: 1, faint: 0 };
const best = (...states) => states.reduce((a, b) => (RANK[b] > RANK[a] ? b : a));

const pct = (x, y) => ({ left: `${(100 * x) / VW}%`, top: `${(100 * y) / VH}%` });
const half = ({ x, y }, r, side) => `M ${x} ${y - r} A ${r} ${r} 0 0 ${side === 'L' ? 0 : 1} ${x} ${y + r} Z`;
const xy = (e) => ({ x1: e.x1, y1: e.y1, x2: e.x2, y2: e.y2 });

// The tooltip: the name of the part, its value, and how it is computed.
export function ItemInfo({ item, fw, raw }) {
    const { type, l, j, k } = item;
    if (type === 'x') {
        const def = INPUTS[k - 1];
        return (
            <>
                <div className="tip-title"><Sym b="x" j={k} /> = {num(fw.x[k - 1])}</div>
                <div>input {k}: <strong>{def.name}</strong>{raw && <>, {def.show(raw[k - 1])}</>}</div>
                {raw && <div className="tip-math">{def.how(raw[k - 1])}</div>}
                <div className="tip-note">It goes into both hidden neurons.</div>
            </>
        );
    }
    if (type === 'w') {
        const w = l === 1 ? W1[k - 1][j - 1] : W2[k - 1][0];
        const src = l === 1 ? fw.x[k - 1] : fw.a1[k - 1];
        return (
            <>
                <div className="tip-title"><Sym b="w" l={l} j={j} k={k} /> = {wnum(w)}</div>
                <div>
                    weight from {l === 1 ? <><Sym b="x" j={k} /> ({INPUTS[k - 1].name})</> : <><Sym b="a" l={1} j={k} /> ({HIDDEN[k - 1].name})</>}
                    {' '}into {l === 1 ? <>neuron {j} of layer 1 ({HIDDEN[j - 1].name})</> : <>the output neuron ({OUTPUT})</>}
                </div>
                <div className="tip-math">signal: {par(wnum(w), w)} × {num(src)} = {num(w * src)}</div>
                <div className="tip-note">
                    {w === 0 ? 'Weight 0: this input does not matter for this neuron.'
                        : w > 0 ? 'Positive: more of this input pushes the neuron up.' : 'Negative: more of this input holds the neuron back.'}
                </div>
            </>
        );
    }
    const z = l === 1 ? fw.z1[j - 1] : fw.z2;
    const a = l === 1 ? fw.a1[j - 1] : fw.a2;
    const b = l === 1 ? B1[j - 1] : B2[0];
    const inVec = l === 1 ? <Sym b="x" vec /> : <Sym b="a" vec l={1} />;
    if (type === 'z') {
        return (
            <>
                <div className="tip-title"><Sym b="z" l={l} j={j} /> = {num(z)}</div>
                <div><strong>Σ</strong>: the weighted sum of neuron {j} of layer {l}, plus its bias</div>
                <div className="tip-math">
                    <Sym b="w" vec l={l} j={j} /> · {inVec} + <Sym b="b" l={l} j={j} /> = {num(z - b)} + {par(wnum(b), b)} = {num(z)}
                </div>
            </>
        );
    }
    if (type === 'a') {
        return (
            <>
                <div className="tip-title"><Sym b="a" l={l} j={j} /> = {num(a)}</div>
                <div><strong>σ</strong>: the sigmoid squeezes <Sym b="z" l={l} j={j} /> into a number from 0 to 1</div>
                <div className="tip-math">σ({num(z)}) = 1 / (1 + e<sup>{num(-z)}</sup>) = {num(a)}</div>
                <div className="tip-note">
                    {l === 1 ? <>The answer to “{HIDDEN[j - 1].name}”: {a >= 0.5 ? 'rather yes' : 'rather no'}.</>
                        : <>f, the chance of going. <Yhat /> = {fw.yhat} ({fw.yhat ? 'go' : 'stay home'}), because f {fw.yhat ? '≥' : '<'} 0.5.</>}
                </div>
            </>
        );
    }
    return (
        <>
            <div className="tip-title"><Sym b="b" l={l} j={j} /> = {wnum(b)}</div>
            <div>the bias of neuron {j} of layer {l} ({l === 1 ? HIDDEN[j - 1].name : OUTPUT})</div>
            <div className="tip-note">It is added to the weighted sum, whatever the inputs are.</div>
        </>
    );
}

// The beach network, every part of it named and hoverable.
//   fw: the forward pass (network.forward); raw: what you know about the afternoon (for the tooltips).
//   mode: lines drawn by their 'signal' w × x, or by their 'weight' w.
//   hover / onHover: the item pointed at, when it is shared with other panels (else kept here).
//   lit: what to light up, instead of lights(hover). focus: the keys of the part being studied (the rest fades).
export default function NetworkDiagram({ fw, raw, mode = 'signal', hover, onHover, lit: litProp, focus = null }) {
    const wrap = useRef(null);
    const [local, setLocal] = useState(null); // { item, x, y, w, h }: what the mouse is on, and where
    const active = hover !== undefined ? hover : local?.item ?? null;
    const lit = litProp !== undefined ? litProp : lights(active);
    const state = (key) => {
        if (lit?.strong.has(key)) return 'strong';
        if (lit?.rel.has(key)) return 'rel';
        if (focus && !focus.has(key)) return 'faint';
        return lit ? 'dim' : 'normal';
    };

    const where = (e) => {
        const r = wrap.current.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height };
    };
    const enter = (item) => (e) => {
        setLocal({ item, ...where(e) });
        onHover?.(item);
    };
    const move = (e) => {
        const p = where(e);
        setLocal((s) => (s ? { ...s, ...p } : s));
    };
    const leave = () => {
        setLocal(null);
        onHover?.(null);
    };
    const hit = (item) => ({ onMouseEnter: enter(item), onMouseMove: move, onMouseLeave: leave, onClick: enter(item) });

    const srcOf = (it) => (it.l === 1 ? fw.x[it.k - 1] : fw.a1[it.k - 1]);
    const edges = [...EDGES].sort((a, b) => RANK[state(a.key)] - RANK[state(b.key)]);
    const strongEdges = EDGES.filter((e) => state(e.key) === 'strong');
    const outState = best(state('out'), state('a2:1'));

    // The tooltip goes where there is room for it: right of the pointer, else left, else centered; below, else above.
    const tip = local && (() => {
        const room = 350;
        const side = local.w - local.x > room ? 'right' : local.x > room ? 'left' : 'center';
        const below = local.y < local.h * 0.55;
        return {
            left: side === 'right' ? local.x + 18 : side === 'left' ? local.x - 18 : local.w / 2,
            top: local.y + (below ? 18 : -18),
            transform: `translate(${{ right: '0', left: '-100%', center: '-50%' }[side]}, ${below ? '0' : '-100%'})`,
        };
    })();

    return (
        <div className="net-scroll">
        <div className="net" ref={wrap}>
            <svg viewBox={`0 0 ${VW} ${VH}`} className="net-svg" role="img"
                aria-label="The beach network: eight inputs, two hidden neurons (will it be nice? will it be quiet?) and one output neuron (go to the beach?)">
                {edges.map((e) => {
                    const st = state(e.key);
                    const s = e.w * srcOf(e.item);
                    const off = mode === 'weight' ? e.w === 0 : Math.abs(s) < 0.005;
                    const sign = mode === 'weight' ? e.w : s;
                    const width = off ? 1.2 : 1 + 1.35 * Math.min(Math.abs(e.w), 3);
                    const alpha = off || mode === 'weight' ? 1 : 0.2 + 0.8 * Math.min(1, Math.abs(s) / 3);
                    return (
                        <g key={e.key} className="net-part" style={{ opacity: OPACITY[st] }}>
                            {st === 'strong' && <line {...xy(e)} className="glow" strokeWidth={width + 10} />}
                            <line {...xy(e)} strokeWidth={width} strokeLinecap="round"
                                style={{ stroke: off ? 'var(--edge-off)' : sign > 0 ? COLORS.pos : COLORS.neg, strokeOpacity: alpha }} />
                            <line {...xy(e)} className="hit" strokeWidth={14} {...hit(e.item)} />
                        </g>
                    );
                })}

                <g className="net-part" style={{ opacity: OPACITY[outState] }}>
                    <line x1={OUT.x + R_O + 3} y1={OUT.y} x2={ARROW_END - 12} y2={OUT.y} stroke={COLORS.act} strokeWidth={3} />
                    <path d={`M ${ARROW_END} ${OUT.y} L ${ARROW_END - 15} ${OUT.y - 7} L ${ARROW_END - 15} ${OUT.y + 7} Z`} fill={COLORS.act} />
                </g>

                {IN.map((p, k) => {
                    const st = state(`x:${k + 1}`);
                    const v = fw.x[k];
                    const t = k === 7 && raw ? towel(raw[7]) : null;
                    return (
                        <g key={k} className="net-part net-node" style={{ opacity: OPACITY[st] }} {...hit({ type: 'x', k: k + 1 })}>
                            {st === 'strong' && <circle cx={p.x} cy={p.y} r={R_IN + 7} className="glow-fill" />}
                            <circle cx={p.x} cy={p.y} r={R_IN} strokeWidth={2}
                                style={{ fill: t ? t.color : actFill(v), stroke: 'var(--node-edge)' }} />
                            <text x={p.x} y={p.y + 5} textAnchor="middle" className="net-val"
                                style={{ fill: t ? '#1e293b' : v > 0.62 ? '#fff' : 'var(--text-color)' }}>{num(v)}</text>
                        </g>
                    );
                })}

                {NEURONS.map(({ c, r, l, j }) => {
                    const a = l === 1 ? fw.a1[j - 1] : fw.a2;
                    const sz = state(`z${l}:${j}`);
                    const sa = state(`a${l}:${j}`);
                    const ink = a > 0.6 ? '#fff' : 'var(--text-color)';
                    return (
                        <g key={`${l}${j}`} className="net-part net-node" style={{ opacity: OPACITY[best(sz, sa)] }}>
                            <circle cx={c.x} cy={c.y} r={r} fill={actFill(a)} stroke={COLORS.act} strokeWidth={2.6} />
                            {sz === 'strong' && <path d={half(c, r, 'L')} className="half-hl" />}
                            {sa === 'strong' && <path d={half(c, r, 'R')} className="half-hl" />}
                            <line x1={c.x} y1={c.y - r} x2={c.x} y2={c.y + r} stroke={COLORS.act} strokeWidth={2} />
                            <text x={c.x - r * 0.45} y={c.y + r * 0.2} textAnchor="middle" className="net-glyph" style={{ fill: ink, fontSize: r * 0.55 }}>Σ</text>
                            <text x={c.x + r * 0.45} y={c.y + r * 0.2} textAnchor="middle" className="net-glyph" style={{ fill: ink, fontSize: r * 0.55 }}>σ</text>
                            <path d={half(c, r, 'L')} className="hit-fill" {...hit({ type: 'z', l, j })} />
                            <path d={half(c, r, 'R')} className="hit-fill" {...hit({ type: 'a', l, j })} />
                        </g>
                    );
                })}
            </svg>

            {/* ── labels ── */}
            {[[IN[0].x, 0, 'input'], [HID[0].x, 1, 'hidden'], [OUT.x, 2, 'output']].map(([x, l, name]) => (
                <div key={l} className="net-lab center" style={pct(x, 22)}>
                    <span className="n-layer strong">layer [{l}]</span> <span className="muted">{name}</span>
                </div>
            ))}
            {IN.map((p, k) => (
                <div key={k} className="net-lab right net-part" style={{ ...pct(p.x - R_IN - 8, p.y), opacity: OPACITY[state(`x:${k + 1}`)] }}>
                    <span className="muted">{INPUTS[k].name}</span> <Sym b="x" j={k + 1} />
                </div>
            ))}
            {HID.map((c, j) => {
                const st = best(state(`z1:${j + 1}`), state(`a1:${j + 1}`));
                return (
                    <React.Fragment key={j}>
                        <div className="net-lab center question net-part" style={{ ...pct(c.x, j === 0 ? c.y - R_H - 18 : c.y + R_H + 20), opacity: OPACITY[st] }}>
                            {HIDDEN[j].name}
                        </div>
                        <div className="net-lab left pill hot net-part" style={{ ...pct(c.x + R_H + 8, c.y + (j === 0 ? -36 : 36)), opacity: OPACITY[state(`a1:${j + 1}`)] }}
                            {...hit({ type: 'a', l: 1, j: j + 1 })}>
                            <Sym b="a" l={1} j={j + 1} /> = {num(fw.a1[j])}
                        </div>
                        <div className="net-lab center pill bias hot net-part" style={{ ...pct(c.x, j === 0 ? c.y + R_H + 21 : c.y - R_H - 21), opacity: OPACITY[state(`b1:${j + 1}`)] }}
                            {...hit({ type: 'b', l: 1, j: j + 1 })}>
                            <Sym b="b" l={1} j={j + 1} /> = {wnum(B1[j])}
                        </div>
                    </React.Fragment>
                );
            })}
            <div className="net-lab center question net-part" style={{ ...pct(OUT.x, OUT.y - R_O - 18), opacity: OPACITY[outState] }}>{OUTPUT}</div>
            <div className="net-lab center pill bias hot net-part" style={{ ...pct(OUT.x, OUT.y + R_O + 21), opacity: OPACITY[state('b2:1')] }}
                {...hit({ type: 'b', l: 2, j: 1 })}>
                <Sym b="b" l={2} j={1} /> = {wnum(B2[0])}
            </div>
            <div className="net-lab left out hot net-part" style={{ ...pct(ARROW_END + 6, OUT.y), opacity: OPACITY[outState] }} {...hit({ type: 'a', l: 2, j: 1 })}>
                <div>f = {num(fw.a2)}</div>
                <div><Yhat /> = {fw.yhat}</div>
                <div className={`verdict ${fw.yhat ? 'yes' : 'no'}`}>{fw.yhat ? 'go!' : 'stay home'}</div>
            </div>
            {strongEdges.map((e) => {
                const { l, j, k } = e.item;
                return (
                    <div key={e.key} className="net-lab center edge-lab" style={pct(e.x1 + (e.x2 - e.x1) * e.frac, e.y1 + (e.y2 - e.y1) * e.frac)}>
                        <Sym b="w" l={l} j={j} k={k} /> = {wnum(e.w)}
                    </div>
                );
            })}

            {local && (
                <div className="tooltip" style={tip}>
                    <ItemInfo item={local.item} fw={fw} raw={raw} />
                </div>
            )}
        </div>
        </div>
    );
}
