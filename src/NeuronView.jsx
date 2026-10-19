import React, { useEffect, useMemo, useState } from 'react';
import { Play, Pause, StepForward, ChevronsRight } from 'lucide-react';
import { INPUTS, HIDDEN, OUTPUT, layerOf, sigmoid, num, wnum, par } from './network';
import { Sym, Yhat } from './Sym';
import { Plot, COLORS, baseLayout, PLOT_CONFIG } from './ui';
import NetworkDiagram, { lights } from './NetworkDiagram';

const NEURONS = [
    { l: 1, j: 1, label: 'Layer 1, neuron 1', name: HIDDEN[0].name },
    { l: 1, j: 2, label: 'Layer 1, neuron 2', name: HIDDEN[1].name },
    { l: 2, j: 1, label: 'Layer 2, neuron 1', name: OUTPUT },
];
const STEP_MS = 650;
const ZS = Array.from({ length: 161 }, (_, i) => -10 + i / 8);

// The input vector of a layer, as a symbol: x⃗ for layer 1, a⃗^[1] for layer 2; and one of its numbers.
const InVec = ({ l }) => (l === 1 ? <Sym b="x" vec /> : <Sym b="a" vec l={1} />);
const InOne = ({ l, k }) => (l === 1 ? <Sym b="x" j={k} /> : <Sym b="a" l={1} j={k} />);

export default function NeuronView({ fw, pTheme }) {
    const [sel, setSel] = useState(0);
    const [hover, setHover] = useState(null);
    const [n, setN] = useState(null); // terms added so far, while stepping (null: all of them)
    const [playing, setPlaying] = useState(false);
    const { l, j, name } = NEURONS[sel];
    const d = layerOf(fw, l);
    const ws = d.W.map((row) => row[j - 1]);
    const b = d.b[j - 1];
    const len = d.ins.length;
    const shown = n ?? len;
    const complete = shown === len;
    const prods = d.ins.map((v, k) => v * ws[k]);
    const dot = prods.slice(0, shown).reduce((s, p) => s + p, 0);
    const z = d.z[j - 1];
    const a = sigmoid(z);
    const names = l === 1 ? INPUTS : HIDDEN;

    useEffect(() => {
        if (!playing) return undefined;
        if (shown >= len) {
            setPlaying(false);
            return undefined;
        }
        const id = setTimeout(() => setN(shown + 1), STEP_MS);
        return () => clearTimeout(id);
    }, [playing, shown, len]);

    const choose = (i) => {
        setSel(i);
        setN(null);
        setPlaying(false);
        setHover(null);
    };
    const play = () => {
        if (playing) return setPlaying(false);
        if (complete) setN(0);
        return setPlaying(true);
    };
    const next = () => {
        setPlaying(false);
        setN(complete ? 1 : shown + 1);
    };

    // The network shows only this neuron's computation; the terms not added yet fade out.
    const focus = useMemo(() => {
        const s = new Set([`z${l}:${j}`, `a${l}:${j}`]);
        for (let k = 1; k <= shown; k++) {
            s.add(`w${l}:${j}:${k}`);
            if (l === 1) s.add(`x:${k}`);
            else s.add(`z1:${k}`).add(`a1:${k}`);
        }
        if (complete) s.add(`b${l}:${j}`);
        if (l === 2) s.add('out');
        return s;
    }, [l, j, shown, complete]);
    const current = !complete && shown > 0 ? shown : null; // the term just added
    const lit = hover ? lights(hover)
        : current ? { strong: new Set([`w${l}:${j}:${current}`]), rel: new Set([l === 1 ? `x:${current}` : `a1:${current}`, `z${l}:${j}`]) }
            : null;

    // Is column k (input k and its weight) the one pointed at, here or on the network?
    const colOn = (k) => (hover
        ? (hover.type === 'w' && hover.l === l && hover.k === k)
        || (l === 1 && hover.type === 'x' && hover.k === k)
        || (l === 2 && hover.type === 'a' && hover.l === 1 && hover.j === k)
        : current === k);
    const col = (k) => ({
        className: `dg-cell ${colOn(k) ? 'on' : ''} ${k > shown ? 'later' : ''}`,
        onMouseEnter: () => setHover({ type: 'w', l, j, k }),
        onMouseLeave: () => setHover(null),
    });
    const tint = (p) => {
        if (Math.abs(p) < 0.005) return undefined;
        const alpha = 0.14 + 0.5 * Math.min(1, Math.abs(p) / 3);
        return { background: p > 0 ? `rgba(249, 115, 22, ${alpha})` : `rgba(59, 130, 246, ${alpha})` };
    };
    const zlo = Math.min(-8, z - 1);
    const zhi = Math.max(8, z + 1);

    return (
        <div className="plots-grid glass-panel neuron-grid">
            <div className="panel">
                <div className="chip-row">
                    {NEURONS.map((nn, i) => (
                        <button key={nn.label} type="button" className={`chip ${sel === i ? 'active' : ''}`} onClick={() => choose(i)}>
                            {nn.label}: <em>{nn.name}</em>
                        </button>
                    ))}
                </div>
                <NetworkDiagram fw={fw} hover={hover} onHover={setHover} lit={lit} focus={focus} />
                <div className="info-card">
                    Every neuron does the same two things: <strong>Σ</strong> (left half) multiplies each of its inputs by its
                    weight, adds the products and the bias, and gives <Sym b="z" l={l} j={j} />. Then <strong>σ</strong> (right
                    half) turns <Sym b="z" l={l} j={j} /> into the activation <Sym b="a" l={l} j={j} />, a number from 0 to 1.
                    Point at a column on the right, or at a line here: both light up together.
                </div>
            </div>

            <div className="panel steps">
                <div className="formula big">
                    <Sym b="z" l={l} j={j} /> = <Sym b="w" vec l={l} j={j} /> · <InVec l={l} /> + <Sym b="b" l={l} j={j} />
                    <span className="sep" />
                    <Sym b="a" l={l} j={j} /> = σ(<Sym b="z" l={l} j={j} />)
                </div>

                <div className="btn-row">
                    <button className="btn" type="button" onClick={play}>
                        {playing ? <Pause size={18} /> : <Play size={18} />}
                        {playing ? 'Pause' : 'Add one term at a time'}
                    </button>
                    <button className="btn btn-secondary" type="button" onClick={next}><StepForward size={16} /> Next term</button>
                    <button className="btn btn-secondary" type="button" onClick={() => { setPlaying(false); setN(null); }} disabled={complete && n === null}>
                        <ChevronsRight size={16} /> All terms
                    </button>
                </div>

                <div className="step">
                    <div className="step-title"><span className="step-n">1</span> Multiply each input by its weight</div>
                    <div className="dot-grid" style={{ gridTemplateColumns: `auto repeat(${len}, minmax(0, 1fr))` }}>
                        <span className="dg-lab" />
                        {names.map((nm, k) => (
                            <span key={k} {...col(k + 1)} className={`${col(k + 1).className} dg-head`} title={nm.name}>
                                <span className="n-input strong">{k + 1}</span>
                                <span className="dg-name">{nm.short}</span>
                            </span>
                        ))}
                        <span className="dg-lab"><Sym b="w" vec l={l} j={j} /></span>
                        {ws.map((w, k) => <span key={k} {...col(k + 1)}>{wnum(w)}</span>)}
                        <span className="dg-lab"><InVec l={l} /></span>
                        {d.ins.map((v, k) => <span key={k} {...col(k + 1)}>{num(v)}</span>)}
                        <span className="dg-lab">product</span>
                        {prods.map((p, k) => (
                            <span key={k} {...col(k + 1)} style={k < shown ? tint(p) : undefined}>{k < shown ? num(p) : '·'}</span>
                        ))}
                    </div>
                    <div className="hint">
                        Column k pairs the weight <Sym b="w" l={l} j={j} k="k" /> with the input <InOne l={l} k="k" /> it multiplies:
                        same position, same k.
                    </div>
                </div>

                <div className="step">
                    <div className="step-title"><span className="step-n">2</span> Add the products: the dot product</div>
                    <div className="formula wrap">
                        <Sym b="w" vec l={l} j={j} /> · <InVec l={l} /> ={' '}
                        {ws.map((w, k) => (
                            <span key={k} className={`term ${colOn(k + 1) ? 'on' : ''} ${k >= shown ? 'later' : ''}`}
                                onMouseEnter={() => setHover({ type: 'w', l, j, k: k + 1 })} onMouseLeave={() => setHover(null)}>
                                {k ? ' + ' : ''}{par(wnum(w), w)} × {num(d.ins[k])}
                            </span>
                        ))}
                    </div>
                    <div className="formula">
                        = {prods.map((p, k) => (
                            <span key={k} className={`term ${colOn(k + 1) ? 'on' : ''} ${k >= shown ? 'later' : ''}`}>
                                {k ? ' + ' : ''}{k < shown ? par(num(p), p) : '…'}
                            </span>
                        ))} = <strong>{num(dot)}</strong>{!complete && <span className="muted"> (so far)</span>}
                    </div>
                </div>

                <div className={`step ${complete ? '' : 'later'}`}>
                    <div className="step-title"><span className="step-n">3</span> Add the bias</div>
                    <div className="formula">
                        <Sym b="z" l={l} j={j} /> = {complete ? num(dot) : '?'} + <span
                            className={`term ${hover?.type === 'b' ? 'on' : ''}`}
                            onMouseEnter={() => setHover({ type: 'b', l, j })} onMouseLeave={() => setHover(null)}>
                            {par(wnum(b), b)}
                        </span> = <strong>{complete ? num(z) : '?'}</strong>
                    </div>
                </div>

                <div className={`step ${complete ? '' : 'later'}`}>
                    <div className="step-title"><span className="step-n">4</span> The activation function: the sigmoid σ</div>
                    <div className="formula">
                        <Sym b="a" l={l} j={j} /> = σ({complete ? num(z) : '?'}) = 1 / (1 + e<sup>{complete ? num(-z) : '?'}</sup>) ={' '}
                        <strong>{complete ? num(a) : '?'}</strong>
                    </div>
                    <Plot
                        data={[
                            { x: ZS, y: ZS.map(sigmoid), mode: 'lines', line: { color: COLORS.go, width: 3 }, hovertemplate: 'z = %{x:.2f}<br>σ(z) = %{y:.3f}<extra></extra>' },
                            ...(complete ? [{ x: [z], y: [a], mode: 'markers', hoverinfo: 'skip', marker: { size: 13, color: COLORS.pos, line: { color: 'white', width: 2 } } }] : []),
                        ]}
                        layout={{
                            ...baseLayout(pTheme),
                            margin: { t: 10, r: 15, l: 50, b: 45 },
                            xaxis: { title: { text: 'z' }, range: [zlo, zhi], gridcolor: pTheme.gridColor, zeroline: false },
                            yaxis: { title: { text: 'a = σ(z)' }, range: [-0.05, 1.05], gridcolor: pTheme.gridColor, zeroline: false, tickvals: [0, 0.5, 1] },
                            shapes: complete ? [
                                { type: 'line', x0: z, x1: z, y0: -0.05, y1: a, line: { color: COLORS.pos, width: 1.5, dash: 'dot' } },
                                { type: 'line', x0: zlo, x1: z, y0: a, y1: a, line: { color: COLORS.pos, width: 1.5, dash: 'dot' } },
                            ] : [],
                        }}
                        config={PLOT_CONFIG}
                        useResizeHandler
                        style={{ width: '100%', height: '230px' }}
                    />
                    <div className="hint">
                        {l === 1
                            ? <>σ squeezes any z into a number from 0 to 1: the neuron’s answer to “{name}”. Large positive z → close to 1, large negative z → close to 0, z = 0 → 0.5.</>
                            : <>This is f, the output of the network: the chance of going. <Yhat /> = {fw.yhat}, because f {fw.yhat ? '≥' : '<'} 0.5: {fw.yhat ? 'go!' : 'stay home'}.</>}
                    </div>
                </div>
            </div>
        </div>
    );
}
