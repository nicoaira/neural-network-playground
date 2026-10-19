import React, { useMemo, useState } from 'react';
import { INPUTS, HIDDEN, OUTPUT, PRESETS, W1, W2, toX, forward, layerOf, num, wnum, par } from './network';
import { Sym, Yhat } from './Sym';
import { ViewTabs } from './ui';
import NetworkDiagram, { keyOf } from './NetworkDiagram';

// The examples of "m examples at once": afternoons 1, 2 and 3 of the slides, and the one of Step 1.
const BATCH = [...PRESETS.slice(1).map((p) => ({ label: p.label.toLowerCase(), detail: p.detail, raw: p.raw })), { label: 'yours', detail: 'the afternoon of Step 1' }];

// Which matrix a cell is in decides what it is related to: in layer 1 the inputs are x, in layer 2 they are a^[1].
const isIn = (c, L) => (L === 1 ? c.type === 'x' : c.type === 'a' && c.l === 1);
const inIdx = (c) => (c.type === 'x' ? c.k : c.j);
const sameI = (a, b) => a.i === b.i;

// How a cell c looks while h is pointed at: 'hl' (it is h), 'rel' (it takes part in h) or ''.
function cellState(h, c, L) {
    if (!h) return '';
    if (keyOf(h) === keyOf(c) && (c.type === 'w' || sameI(h, c))) return 'hl';
    if (h.type === 'w') {
        if (h.l !== L) return '';
        if (isIn(c, L) && inIdx(c) === h.k) return 'rel';
        return c.type === 'z' && c.l === L && c.j === h.j ? 'rel' : '';
    }
    if (isIn(h, L)) {
        if (c.type === 'w' && c.l === L && c.k === inIdx(h)) return 'rel';
        return c.type === 'z' && c.l === L && sameI(h, c) ? 'rel' : '';
    }
    if (h.l !== L) return '';
    if (h.type === 'z') {
        if (isIn(c, L) && sameI(h, c)) return 'rel';
        if (c.type === 'w' && c.l === L && c.j === h.j) return 'rel';
        return (c.type === 'b' || (c.type === 'a' && c.l === L)) && c.j === h.j && sameI(h, c) ? 'rel' : '';
    }
    return c.type === 'z' && c.l === L && c.j === h.j && sameI(h, c) ? 'rel' : '';
}

// A matrix (or a row vector) between square brackets, with optional row and column labels.
function Mx({ name, shape, cells, rowLabels, colLabels, cw = '3.5em', st, setHover, onRowClick, rowMark, after }) {
    const cols = `repeat(${cells[0].length}, ${cw})`;
    const side = (labels, cls) => (
        <div className={`mx-side ${cls}`}>
            {colLabels && <span className="mx-clab" />}
            <div className="mx-side-rows">
                {labels.map((lab, r) => (
                    <span key={r} className={`mx-rlab ${onRowClick && cls === 'left' ? 'click' : ''} ${rowMark === r && cls === 'left' ? 'mark' : ''}`}
                        onClick={onRowClick && cls === 'left' ? () => onRowClick(r) : undefined}>{lab}</span>
                ))}
            </div>
        </div>
    );
    return (
        <div className="mx">
            <div className="mx-body">
                {rowLabels && side(rowLabels, 'left')}
                <div className="mx-main">
                    {colLabels && (
                        <div className="mx-clabels" style={{ gridTemplateColumns: cols }}>
                            {colLabels.map((c, i) => <span key={i} className="mx-clab">{c}</span>)}
                        </div>
                    )}
                    <div className="mx-br">
                        <div className="mx-cells" style={{ gridTemplateColumns: cols }}>
                            {cells.flatMap((row, r) => row.map((c, ci) => (
                                <span key={`${r}-${ci}`} tabIndex={0} className={`mx-cell ${st(c.item)}`}
                                    onMouseEnter={() => setHover(c.item)} onMouseLeave={() => setHover(null)}
                                    onFocus={() => setHover(c.item)} onBlur={() => setHover(null)}>
                                    {c.text}
                                </span>
                            )))}
                        </div>
                    </div>
                </div>
                {after && side(after, 'right')}
            </div>
            <div className="mx-name">{name}<span className="mx-shape">{shape}</span></div>
        </div>
    );
}

// z_j written out: every input times its weight, plus the bias. emph: the term to stress ({ k } or { b }).
function ZLine({ L, j, i, d, emph = {} }) {
    return (
        <div className="formula wrap">
            <Sym b="z" l={L} j={j} i={i} /> ={' '}
            {d.ins.map((v, k) => (
                <span key={k} className={`term ${emph.k === k + 1 ? 'on' : ''}`}>
                    {k ? ' + ' : ''}{num(v)} × {par(wnum(d.W[k][j - 1]), d.W[k][j - 1])}
                </span>
            ))}
            {' + '}<span className={`term ${emph.b ? 'on' : ''}`}>{par(wnum(d.b[j - 1]), d.b[j - 1])}</span> = <strong>{num(d.z[j - 1])}</strong>
        </div>
    );
}

function Explain({ hover, L, batch, exOf, pick }) {
    const inName = (k) => (L === 1 ? INPUTS[k - 1].name : HIDDEN[k - 1].name);
    const outName = (j) => (L === 1 ? HIDDEN[j - 1].name : OUTPUT);
    const InOne = ({ k, i }) => (L === 1 ? <Sym b="x" j={k} i={i} /> : <Sym b="a" l={1} j={k} i={i} />);
    const Wm = () => <Sym b="W" mat l={L} />;
    const ofLayer = hover && (hover.type === 'w' || hover.type === 'z' || hover.type === 'b' ? hover.l === L : hover.type === 'x' ? L === 1 : hover.l === L || isIn(hover, L));
    if (!hover) {
        return (
            <div className="explain muted">
                Point at any number of the matrices above, or at a part of the network: the same part lights up in both.
                {batch ? ' Every row of the first matrix is one example (one afternoon); they all go through the same weights.' : ''}
            </div>
        );
    }
    if (!ofLayer) {
        return <div className="explain muted">That part of the network belongs to layer {L === 1 ? 2 : 1}: switch to it above.</div>;
    }
    const i = batch ? hover.i ?? pick : undefined;
    const d = exOf(i);
    if (hover.type === 'w') {
        const { j, k } = hover;
        const w = (L === 1 ? W1 : W2)[k - 1][j - 1];
        return (
            <div className="explain">
                <div>
                    <strong><Sym b="w" l={L} j={j} k={k} /> = {wnum(w)}</strong> is in <strong>row <span className="n-input">{k}</span></strong> of{' '}
                    <Wm /> (the input it comes from: {inName(k)}) and <strong>column <span className="n-unit">{j}</span></strong> (the neuron
                    it goes into: {outName(j)}). Its name says the neuron first and the input second; the matrix says the row (input) first.
                </div>
                <ZLine L={L} j={j} i={i} d={d} emph={{ k }} />
            </div>
        );
    }
    if (isIn(hover, L)) {
        const k = inIdx(hover);
        const ncol = L === 1 ? 2 : 1;
        return (
            <div className="explain">
                <div>
                    <strong><InOne k={k} i={i} /> = {num(d.ins[k - 1])}</strong> ({inName(k)}) multiplies <strong>row <span className="n-input">{k}</span></strong> of{' '}
                    <Wm />: {ncol === 2 ? 'one product for each neuron of the layer, the two lines that leave this input.' : 'its weight into the output neuron.'}
                </div>
                {Array.from({ length: ncol }, (_, j) => <ZLine key={j} L={L} j={j + 1} i={i} d={d} emph={{ k }} />)}
            </div>
        );
    }
    const { j } = hover;
    if (hover.type === 'z') {
        return (
            <div className="explain">
                <div>
                    <strong>Row times column</strong>: the row {L === 1 ? <Sym b="x" vec i={i} /> : <Sym b="a" vec l={1} i={i} />}
                    {batch && <> (row {i} of the first matrix)</>} times column <span className="n-unit">{j}</span> of <Wm />, which is{' '}
                    <Sym b="w" vec l={L} j={j} />: multiply in pairs and add (the dot product), then add the bias.
                </div>
                <ZLine L={L} j={j} i={i} d={d} />
            </div>
        );
    }
    if (hover.type === 'b') {
        return (
            <div className="explain">
                <div>
                    <strong><Sym b="b" l={L} j={j} /> = {wnum(d.b[j - 1])}</strong>, the bias of neuron {j} ({outName(j)}), is added after the products.
                    {batch && <> <Sym b="B" mat l={L} /> repeats <Sym b="b" vec l={L} /> in every row: the same biases for every example.</>}
                </div>
                <ZLine L={L} j={j} i={i} d={d} emph={{ b: true }} />
            </div>
        );
    }
    const z = d.z[j - 1];
    const a = d.a[j - 1];
    return (
        <div className="explain">
            <div>
                σ is applied to each number of {batch ? <Sym b="Z" mat l={L} /> : <Sym b="z" vec l={L} />}, one by one:
                {L === 1 ? <> the answer to “{outName(j)}”.</> : <> f, the chance of going; <Yhat /> = 1 when it is 0.5 or more.</>}
            </div>
            <div className="formula">
                <Sym b="a" l={L} j={j} i={i} /> = σ({num(z)}) = 1 / (1 + e<sup>{num(-z)}</sup>) = <strong>{num(a)}</strong>
                {L === 2 && <> → <Yhat /> = {a >= 0.5 ? 1 : 0} ({a >= 0.5 ? 'go' : 'stay home'})</>}
            </div>
        </div>
    );
}

export default function MatrixView({ fw: fwYours, raw }) {
    const [L, setL] = useState(1);
    const [batch, setBatch] = useState(false);
    const [hover, setHover] = useState(null);
    const [pick, setPick] = useState(BATCH.length); // the example shown on the network (m examples): yours
    const fws = useMemo(() => [...BATCH.slice(0, -1).map((e) => forward(toX(e.raw))), fwYours], [fwYours]);
    const m = fws.length;
    const shownI = batch ? hover?.i ?? pick : m;
    const exOf = (i) => layerOf(i ? fws[i - 1] : fwYours, L);
    const rows = batch ? fws.map((_, i) => i + 1) : [undefined];
    const st = (item) => cellState(hover, item, L);
    const onDiagram = (it) => setHover(it && batch && it.type !== 'w' ? { ...it, i: shownI } : it);

    const W = L === 1 ? W1 : W2;
    const nIn = W.length;
    const nOut = W[0].length;
    const inCells = rows.map((i) => exOf(i).ins.map((v, k) => ({
        item: L === 1 ? { type: 'x', k: k + 1, i } : { type: 'a', l: 1, j: k + 1, i }, text: num(v),
    })));
    const wCells = W.map((row, k) => row.map((w, j) => ({ item: { type: 'w', l: L, j: j + 1, k: k + 1 }, text: wnum(w) })));
    const perRow = (type, values) => rows.map((i) => values(exOf(i)).map((v, j) => ({
        item: { type, l: L, j: j + 1, i }, text: type === 'b' ? wnum(v) : num(v),
    })));
    const bCells = perRow('b', (d) => d.b);
    const zCells = perRow('z', (d) => d.z);
    const aCells = perRow('a', (d) => d.a);

    const mm = batch ? 'm' : '1';
    const shape = (r, c) => `${r} × ${c}`;
    const unitLabels = Array.from({ length: nOut }, (_, j) => <span className="n-unit strong">{j + 1}</span>);
    const inputLabels = Array.from({ length: nIn }, (_, k) => <span className="n-input strong">{k + 1}</span>);
    const exLabels = batch && BATCH.map((e, r) => <><span className="n-ex strong">({r + 1})</span> {e.label}</>);
    const yhats = L === 2 && batch && fws.map((f) => <span className={`verdict ${f.yhat ? 'yes' : 'no'}`}>← {f.yhat ? 'yes' : 'no'}</span>);

    // The part of the network this matrix product computes; the rest fades out.
    const focus = useMemo(() => {
        const s = new Set();
        if (L === 1) {
            for (let k = 1; k <= 8; k++) s.add(`x:${k}`).add(`w1:1:${k}`).add(`w1:2:${k}`);
            [1, 2].forEach((j) => s.add(`z1:${j}`).add(`a1:${j}`).add(`b1:${j}`));
        } else {
            [1, 2].forEach((k) => s.add(`z1:${k}`).add(`a1:${k}`).add(`w2:1:${k}`));
            s.add('z2:1').add('a2:1').add('b2:1').add('out');
        }
        return s;
    }, [L]);

    const inName = L === 1 ? (batch ? <Sym b="X" mat /> : <Sym b="x" vec />) : (batch ? <Sym b="A" mat l={1} /> : <Sym b="a" vec l={1} />);
    const bName = batch ? <Sym b="B" mat l={L} /> : <Sym b="b" vec l={L} />;
    const zName = batch ? <Sym b="Z" mat l={L} /> : <Sym b="z" vec l={L} />;
    const aName = batch ? <Sym b="A" mat l={L} /> : <Sym b="a" vec l={L} />;

    return (
        <>
            <div className="glass-panel matrix-panel">
                <div className="panel-header">
                    <ViewTabs options={[{ key: 1, label: 'Layer 1: inputs → hidden layer' }, { key: 2, label: 'Layer 2: hidden layer → output' }]}
                        value={L} onChange={(v) => { setL(v); setHover(null); }} />
                    <ViewTabs options={[{ key: 'one', label: 'One example' }, { key: 'many', label: `m = ${m} examples at once` }]}
                        value={batch ? 'many' : 'one'} onChange={(v) => { setBatch(v === 'many'); setHover(null); }} />
                </div>
                <div className="formula big center">
                    {zName} = {inName} <Sym b="W" mat l={L} /> + {bName}
                    <span className="sep" />
                    {aName} = σ({zName})
                    <span className="sep" />
                    <span className="muted">
                        ({mm} × {nIn})({nIn} × {nOut}) → ({mm} × {nOut})
                    </span>
                </div>

                <div className="eq-row">
                    <Mx name={inName} shape={shape(batch ? m : 1, nIn)} cells={inCells} colLabels={inputLabels}
                        rowLabels={exLabels || undefined} onRowClick={batch ? (r) => setPick(r + 1) : undefined} rowMark={batch ? shownI - 1 : undefined}
                        cw={L === 1 ? '3.3em' : '3.6em'} st={st} setHover={setHover} />
                    <Mx name={<Sym b="W" mat l={L} />} shape={shape(nIn, nOut)} cells={wCells} cw="3.6em"
                        colLabels={Array.from({ length: nOut }, (_, j) => <Sym b="w" vec l={L} j={j + 1} />)}
                        rowLabels={Array.from({ length: nIn }, (_, k) => (
                            <><span className="n-input strong">{k + 1}</span> <span className="muted">{L === 1 ? INPUTS[k].short : HIDDEN[k].short}</span></>
                        ))}
                        st={st} setHover={setHover} />
                    <span className="op">+</span>
                    <Mx name={bName} shape={shape(batch ? m : 1, nOut)} cells={bCells} colLabels={unitLabels} cw="3.6em" st={st} setHover={setHover} />
                    <span className="op">=</span>
                    <Mx name={zName} shape={shape(batch ? m : 1, nOut)} cells={zCells} colLabels={unitLabels} cw="3.6em" st={st} setHover={setHover} />
                    <span className="op sigma">σ →</span>
                    <Mx name={aName} shape={shape(batch ? m : 1, nOut)} cells={aCells} colLabels={unitLabels} cw="3.6em" st={st} setHover={setHover}
                        after={yhats || undefined} />
                </div>

            </div>

            <div className="plots-grid glass-panel matrix-net">
                <div className="panel">
                    <div className="panel-header">
                        <h3>The same numbers in the network</h3>
                        {batch && (
                            <span className="muted small">
                                showing example <span className="n-ex strong">({shownI})</span>: {BATCH[shownI - 1].label}, {BATCH[shownI - 1].detail}.
                                Point at a row to see another one; click its label to keep it.
                            </span>
                        )}
                    </div>
                    <NetworkDiagram fw={fws[shownI - 1]} raw={shownI === m ? raw : BATCH[shownI - 1].raw} hover={hover} onHover={onDiagram} focus={focus} />
                </div>
                <div className="panel">
                    <h3>What you are pointing at</h3>
                    <Explain hover={hover} L={L} batch={batch} exOf={exOf} pick={shownI} />
                    <h3>How to read <Sym b="W" mat l={L} /></h3>
                    <div className="info-card guide">
                        {L === 1 ? (
                            <ul>
                                <li><strong>One column per neuron.</strong> Column <span className="n-unit">j</span> is <Sym b="w" vec l={1} j="j" />: the 8 weights of neuron j (1: nice?, 2: quiet?).</li>
                                <li><strong>One row per input.</strong> Row <span className="n-input">k</span> holds the two weights that leave input k, one for each neuron.</li>
                                <li>Row <span className="n-input">k</span>, column <span className="n-unit">j</span>: <Sym b="w" l={1} j="j" k="k" />, from input k into neuron j.</li>
                                <li><strong>Row × column.</strong> <Sym b="z" l={1} j="j" /> = <Sym b="x" vec /> times column j: multiply in pairs and add, the dot product of the previous tab.</li>
                                <li><strong>Shapes.</strong> (1 × 8)(8 × 2) → (1 × 2): the inner 8s must match; the outer numbers give the shape of the result.</li>
                                <li><strong>m examples.</strong> <Sym b="X" mat /> has one row per example: (m × 8)(8 × 2) → (m × 2), every afternoon at once, through the same <Sym b="W" mat l={1} />.</li>
                            </ul>
                        ) : (
                            <ul>
                                <li><Sym b="W" mat l={2} /> has <strong>2 rows</strong>, one per input of layer 2 (the two hidden neurons), and <strong>1 column</strong>, one per neuron (the output).</li>
                                <li>Its inputs are not <Sym b="x" vec />: they are the activations of layer 1, <Sym b="a" vec l={1} />.</li>
                                <li><strong>Shapes.</strong> (1 × 2)(2 × 1) → (1 × 1): one number, f, the chance of going.</li>
                                <li>In general, layer after layer: <Sym b="a" vec l="l" /> = σ(<Sym b="a" vec l="l−1" /> <Sym b="W" mat l="l" /> + <Sym b="b" vec l="l" />), starting from <Sym b="a" vec l={0} /> = <Sym b="x" vec />.</li>
                            </ul>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}
