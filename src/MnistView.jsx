import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Eraser, Shuffle, Maximize2, Minimize2 } from 'lucide-react';
import { MODEL, SAMPLES, H1, H2, N_IN, forward, weightsOf } from './mnist';
import { num } from './network';
import { Sym } from './Sym';
import { COLORS, ViewTabs } from './ui';
import DrawPad from './DrawPad';
import MnistDiagram, { TinyImage } from './MnistDiagram';

const WEIGHTS = Array.from({ length: H1 }, (_, j) => weightsOf(j));
const FIRST = SAMPLES.findIndex((s) => s.label === 3);
const pct = (v) => `${(100 * v).toFixed(1)} %`;

// Step 4: a network that reads handwritten digits. Draw one, or pick a test image, and watch every neuron.
export default function MnistView({ dark }) {
    const pad = useRef(null);
    const [x, setX] = useState(SAMPLES[FIRST].pixels);
    const [sample, setSample] = useState(FIRST); // the test image shown; null while you draw
    const [expanded, setExpanded] = useState(true);
    const [hov, setHov] = useState(null);
    const [tiles, setTiles] = useState('w'); // the layer-1 images: 'w' (the weights) or 'xw' (your drawing × the weights)
    const fw = useMemo(() => forward(x), [x]);
    const blank = !x.some((v) => v > 0);

    useEffect(() => pad.current?.show(SAMPLES[FIRST].pixels), []);
    const pick = (k) => {
        setSample(k);
        setX(SAMPLES[k].pixels);
        pad.current.show(SAMPLES[k].pixels);
    };
    const another = () => {
        let k = Math.floor(Math.random() * SAMPLES.length);
        if (k === sample) k = (k + 1) % SAMPLES.length;
        pick(k);
    };
    const clear = () => {
        pad.current.clear();
        setSample(null);
        setX(new Float32Array(N_IN));
    };
    const tileValues = useMemo(() => (tiles === 'w' ? WEIGHTS : WEIGHTS.map((w) => w.map((v, i) => v * x[i]))), [tiles, x]);

    return (
        <>
            <div className="glass-panel mnist-intro">
                <h3>Reading handwritten digits</h3>
                <div>
                    A picture of a digit is <strong>28 × 28 = 784 pixels</strong>, each a number from 0 (black) to 1 (white). The network
                    reads them as 784 inputs, row by row, passes them through two hidden layers of {H1} and {H2} neurons, and gives one
                    chance for each digit from 0 to 9. It learned its {(784 * H1 + H1 + H1 * H2 + H2 + H2 * 10 + 10).toLocaleString()} weights
                    and biases from 55 000 example images (the MNIST dataset), and gets {pct(MODEL.accuracy.test)} of 10 000 new test images right.
                </div>
            </div>

            <div className="glass-panel mnist-grid">
                <div className="panel draw-panel">
                    <div className="panel-header">
                        <h3>Draw a digit</h3>
                        <div className="btn-row">
                            <button className="btn btn-secondary btn-small" type="button" onClick={clear}><Eraser size={16} /> Clear</button>
                            <button className="btn btn-secondary btn-small" type="button" onClick={another}><Shuffle size={16} /> Example</button>
                        </div>
                    </div>
                    <DrawPad ref={pad} onInput={setX} onStart={() => setSample(null)} />
                    <div className="answer">
                        {blank ? <span className="muted">Draw a digit in the black square, as large as you can.</span> : (
                            <>
                                <span className="muted">The network says</span>
                                <span className="answer-digit">{fw.best}</span>
                                <span className="answer-pc">{pct(fw.p[fw.best])}</span>
                                {sample != null && (
                                    <span className={`small ${SAMPLES[sample].label === fw.best ? 'ok' : 'bad'}`}>
                                        (the image is a {SAMPLES[sample].label})
                                    </span>
                                )}
                            </>
                        )}
                    </div>
                    <div className="small muted">Test images the network never saw while learning:</div>
                    <div className="samples-grid">
                        {SAMPLES.map((s, k) => (
                            <button key={k} type="button" className={`sample-btn ${sample === k ? 'active' : ''}`} onClick={() => pick(k)}
                                title={`a ${s.label} (test image ${s.index})`}>
                                <TinyImage values={s.pixels} size={26} />
                            </button>
                        ))}
                    </div>
                </div>

                <div className="panel">
                    <div className="panel-header">
                        <h3>The network, live</h3>
                        <button className="btn btn-secondary btn-small" type="button" onClick={() => setExpanded(!expanded)}>
                            {expanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                            {expanded ? 'Show the image as one block' : 'Expand the image into its 784 inputs'}
                        </button>
                    </div>
                    <MnistDiagram fw={fw} expanded={expanded} hov={hov} onHover={setHov} dark={dark} />
                    <div className="info-card">
                        {expanded ? (
                            <>
                                The image is laid out <strong>row by row</strong>: row 1 gives the inputs <Sym b="x" j={1} /> to{' '}
                                <Sym b="x" j={28} />, row 2 gives <Sym b="x" j={29} /> to <Sym b="x" j={56} />, … and row 28 ends with{' '}
                                <Sym b="x" j={784} />. Only the first and the last inputs are drawn (the corners, always black); point at any
                                pixel to see its input and its lines.{' '}
                            </>
                        ) : 'All 784 pixels go into every neuron of layer 1: 784 × 16 weights. '}
                        Brighter neurons are <strong>more active</strong>. Each line shows its signal, activation × weight:{' '}
                        <span style={{ color: COLORS.pos }}>orange</span> pushes the next neuron up,{' '}
                        <span style={{ color: COLORS.neg }}>blue</span> holds it back. Point at a neuron to see its value and its lines.
                    </div>
                </div>
            </div>

            <div className="glass-panel gallery-panel">
                <div className="panel-header">
                    <h3>What each neuron of layer 1 looks at</h3>
                    <ViewTabs options={[{ key: 'w', label: 'its weights' }, { key: 'xw', label: 'your drawing × its weights' }]} value={tiles} onChange={setTiles} />
                </div>
                <div className="gallery">
                    {tileValues.map((vals, j) => {
                        const lit = Math.min(1, fw.a1[j] / MODEL.scale[0][j]);
                        return (
                            <div key={j} className={`w-tile ${hov?.type === 'h1' && hov.j === j ? 'on' : ''}`}
                                onMouseEnter={() => setHov({ type: 'h1', j })} onMouseLeave={() => setHov(null)}>
                                <TinyImage values={vals} mode="signed" dark={dark} size={76} />
                                <div className="w-cap"><Sym b="a" l={1} j={j + 1} /> = {num(fw.a1[j], 1)}</div>
                                <div className="w-bar"><span style={{ width: `${100 * lit}%` }} /></div>
                            </div>
                        );
                    })}
                </div>
                <div className="info-card">
                    {tiles === 'w' ? (
                        <>
                            A neuron of layer 1 has <strong>one weight per pixel</strong>, so its 784 weights can be drawn as a 28 × 28 image:{' '}
                            <span style={{ color: COLORS.pos }}>orange</span> where a white pixel pushes it up,{' '}
                            <span style={{ color: COLORS.neg }}>blue</span> where it holds it back. Nobody drew these patterns: they come out of
                            training. Point at one to find its neuron in the network.
                        </>
                    ) : (
                        <>
                            Each pixel of your drawing times its weight: only the white pixels count. A neuron is more active when there is more{' '}
                            <span style={{ color: COLORS.pos }}>orange</span> than <span style={{ color: COLORS.neg }}>blue</span> (plus its bias).
                            The bar under each image shows how active it is now.
                        </>
                    )}
                </div>
            </div>
        </>
    );
}
