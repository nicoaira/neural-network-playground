import React, { useMemo, useState } from 'react';
import { INPUTS, HIDDEN, PRESETS, TOWELS, toX, forward, sameRaw, num, hhmm } from './network';
import { Sym, Yhat } from './Sym';
import { Plot, COLORS, Metric, ViewTabs, baseLayout, PLOT_CONFIG } from './ui';
import NetworkDiagram from './NetworkDiagram';

// The inputs that can be swept on the "what if" plot (the others are yes/no or a color).
const SWEEPABLE = INPUTS.map((d, k) => k).filter((k) => INPUTS[k].kind === 'range' || INPUTS[k].kind === 'time');

export function PresetBar({ raw, setRaw }) {
    return (
        <div className="preset-row">
            {PRESETS.map((p) => (
                <button key={p.key} type="button" className={`chip ${sameRaw(p.raw, raw) ? 'active' : ''}`}
                    onClick={() => setRaw([...p.raw])} title={p.detail}>
                    {p.label}
                </button>
            ))}
        </div>
    );
}

function InputControl({ k, def, value, x, onChange }) {
    const head = (
        <div className="slider-header">
            <span><Sym b="x" j={k + 1} /> {def.name}</span>
            <span className="in-val">{def.show(value)} <span className="in-x">→ {num(x)}</span></span>
        </div>
    );
    if (def.kind === 'bool') {
        return (
            <div className="slider-container in-row">
                {head}
                <div className="seg">
                    {[false, true].map((v) => (
                        <button key={String(v)} type="button" className={value === v ? 'active' : ''} onClick={() => onChange(v)}>
                            {v ? 'yes' : 'no'}
                        </button>
                    ))}
                </div>
            </div>
        );
    }
    if (def.kind === 'towel') {
        return (
            <div className="slider-container in-row">
                {head}
                <div className="seg">
                    {TOWELS.map((t) => (
                        <button key={t.name} type="button" className={value === t.name ? 'active' : ''} onClick={() => onChange(t.name)}>
                            <span className="swatch" style={{ background: t.color }} /> {t.name}
                        </button>
                    ))}
                </div>
            </div>
        );
    }
    return (
        <div className="slider-container in-row">
            {head}
            <input type="range" min={def.min} max={def.max} step={def.step} value={value} aria-label={def.name}
                onChange={(e) => onChange(Number(e.target.value))} />
        </div>
    );
}

export default function Playground({ raw, setRaw, pTheme }) {
    const [mode, setMode] = useState('signal');
    const [sweepK, setSweepK] = useState(1); // the input on the "what if" plot (temperature)
    const x = useMemo(() => toX(raw), [raw]);
    const fw = useMemo(() => forward(x), [x]);
    const setOne = (k) => (v) => setRaw(raw.map((r, i) => (i === k ? v : r)));

    // Sweep one input from its minimum to its maximum, the other seven as they are.
    const sweep = useMemo(() => {
        const def = INPUTS[sweepK];
        const n = 241;
        const vs = Array.from({ length: n }, (_, i) => def.min + ((def.max - def.min) * i) / (n - 1));
        const rows = vs.map((v) => forward(toX(raw.map((r, i) => (i === sweepK ? v : r)))));
        return { def, vs, a1: rows.map((r) => r.a1[0]), a2: rows.map((r) => r.a1[1]), f: rows.map((r) => r.a2) };
    }, [raw, sweepK]);

    const now = raw[sweepK];
    const isTime = sweep.def.kind === 'time';
    const plot = [
        { x: sweep.vs, y: sweep.a1, mode: 'lines', name: 'a₁ nice?', line: { color: COLORS.nice, width: 2.5 }, hovertemplate: 'nice? %{y:.2f}<extra></extra>' },
        { x: sweep.vs, y: sweep.a2, mode: 'lines', name: 'a₂ quiet?', line: { color: COLORS.quiet, width: 2.5 }, hovertemplate: 'quiet? %{y:.2f}<extra></extra>' },
        { x: sweep.vs, y: sweep.f, mode: 'lines', name: 'f go?', line: { color: COLORS.go, width: 4 }, hovertemplate: 'go? f = %{y:.2f}<extra></extra>' },
        {
            x: [now, now, now], y: [fw.a1[0], fw.a1[1], fw.a2], mode: 'markers', hoverinfo: 'skip', showlegend: false,
            marker: { size: 11, color: [COLORS.nice, COLORS.quiet, COLORS.go], line: { color: 'white', width: 1.5 } },
        },
    ];

    return (
        <>
            <div className="play-grid glass-panel">
                <div className="panel">
                    <div className="panel-header">
                        <h3>The beach network</h3>
                        <ViewTabs options={[{ key: 'signal', label: 'Lines: signals w × x' }, { key: 'weight', label: 'Lines: weights w' }]}
                            value={mode} onChange={setMode} />
                    </div>
                    <NetworkDiagram fw={fw} raw={raw} mode={mode} />
                    <div className="info-card">
                        <strong>Point at any part</strong> of the network: a line (a weight), an input, the <strong>Σ</strong> half of a
                        neuron (its weighted sum z), the <strong>σ</strong> half (its activation a) or a bias b.{' '}
                        <span style={{ color: COLORS.pos }}>Orange</span> pushes the next neuron up,{' '}
                        <span style={{ color: COLORS.neg }}>blue</span> holds it back; a wider line is a larger weight.
                        {mode === 'signal' ? ' A brighter line carries a larger signal w × x: a line stays gray when its input is 0.'
                            : ' Gray lines have weight 0.'}{' '}
                        The circles fill with green as their value goes from 0 to 1.
                    </div>
                </div>
                <div className="panel">
                    <div className="panel-header"><h3>The afternoon</h3></div>
                    <PresetBar raw={raw} setRaw={setRaw} />
                    <div className="inputs">
                        {INPUTS.map((def, k) => (
                            <InputControl key={def.name} k={k} def={def} value={raw[k]} x={x[k]} onChange={setOne(k)} />
                        ))}
                    </div>
                </div>
            </div>

            <div className="controls glass-panel">
                <div className="panel">
                    <h3>The answer</h3>
                    <div className="metrics-grid">
                        <Metric value={num(fw.a1[0])} label={<><Sym b="a" l={1} j={1} /> · {HIDDEN[0].name}</>} />
                        <Metric value={num(fw.a1[1])} label={<><Sym b="a" l={1} j={2} /> · {HIDDEN[1].name}</>} />
                        <Metric value={num(fw.a2)} label={<>f = <Sym b="a" l={2} j={1} /> · the chance of going</>} />
                        <Metric className={fw.yhat ? 'decision yes' : 'decision no'} value={fw.yhat ? 'go! (1)' : 'stay home (0)'}
                            label={<><Yhat />: 1 when f ≥ 0.5</>} />
                    </div>
                    <div className="info-card">
                        The output neuron computes f = σ(6 <Sym b="a" l={1} j={1} /> + 6 <Sym b="a" l={1} j={2} /> − 9).
                        One high answer gives at most 6 − 9 = −3, so f stays low: you only go when the afternoon is
                        <strong> nice and quiet</strong>.
                    </div>
                </div>
                <div className="panel">
                    <div className="panel-header">
                        <h3>What if only one input changes?</h3>
                        <select className="select" value={sweepK} onChange={(e) => setSweepK(Number(e.target.value))} aria-label="input to change">
                            {SWEEPABLE.map((k) => <option key={k} value={k}>{`x${'₁₂₃₄₅₆₇₈'[k]} ${INPUTS[k].name}`}</option>)}
                        </select>
                    </div>
                    <Plot
                        data={plot}
                        layout={{
                            ...baseLayout(pTheme),
                            showlegend: true,
                            legend: { orientation: 'h', x: 0, y: 1.12 },
                            margin: { t: 30, r: 15, l: 50, b: 50 },
                            xaxis: {
                                title: { text: isTime ? 'arrival time' : `${sweep.def.name}${sweep.def.unit ? ` (${sweep.def.unit})` : ''}` },
                                gridcolor: pTheme.gridColor, zeroline: false,
                                ...(isTime ? { tickvals: [10, 12, 14, 16, 18, 20], ticktext: [10, 12, 14, 16, 18, 20].map(hhmm) } : {}),
                            },
                            yaxis: { title: { text: 'value (0 to 1)' }, range: [-0.03, 1.03], gridcolor: pTheme.gridColor, zeroline: false },
                            shapes: [
                                { type: 'line', x0: sweep.def.min, x1: sweep.def.max, y0: 0.5, y1: 0.5, line: { color: COLORS.muted, width: 1.5, dash: 'dash' } },
                                { type: 'line', x0: now, x1: now, y0: 0, y1: 1, line: { color: pTheme.fontColor, width: 1, dash: 'dot' } },
                            ],
                        }}
                        config={PLOT_CONFIG}
                        useResizeHandler
                        style={{ width: '100%', height: '320px' }}
                    />
                    <div className="info-card">
                        The other seven inputs stay as they are; the dotted line is the afternoon above. Look for inputs that push the
                        two hidden neurons in opposite directions: the temperature makes the afternoon nicer but the beach busier.
                    </div>
                </div>
            </div>
        </>
    );
}
