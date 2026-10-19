import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Eraser, Shuffle, Maximize2, Minimize2, Play, Pause, RotateCcw } from 'lucide-react';
import { MODEL, SAMPLES, H1, H2, N_IN, forward, weightsOf, loadHistory } from './mnist';
import { num } from './network';
import { Sym } from './Sym';
import { COLORS, Toggle, ViewTabs, Plot, baseLayout, PLOT_CONFIG } from './ui';
import DrawPad from './DrawPad';
import MnistDiagram, { TinyImage } from './MnistDiagram';

const FIRST = SAMPLES.findIndex((s) => s.label === 3);
const pct = (v) => `${(100 * v).toFixed(1)} %`;
const fmtEpoch = (e) => (Number.isInteger(e) ? String(e) : String(Number(e.toFixed(3))));
const EPOCH_MS = 450; // how long each epoch lasts on screen while training

// Training: the cost J and the accuracy, epoch after epoch, with the controls to start, pause and move along it.
function TrainingPanel({ history, snap, playing, loading, error, onTrain, onPause, onSeek, onTrained, pTheme }) {
    const last = history ? history.length - 1 : 0;
    const upTo = history ? history.slice(0, (snap ?? last) + 1) : [];
    const cur = history && snap != null ? history[snap] : null;
    const epochs = history ? history[last].epoch : MODEL.training.epochs;
    return (
        <div className="glass-panel train-panel">
            <div className="panel train-left">
                <h3>Training</h3>
                <div className="info-card">
                    Training starts from <strong>random weights</strong>: the network answers at random, about 1 time in 10. After each
                    pass over the 60 000 training images (an <strong>epoch</strong>), gradient descent has changed every weight a little:
                    the cost J goes down and the accuracy on the test images goes up. Watch the images of layer 1 below take shape, and
                    the answer for your drawing change.
                </div>
                <div className="btn-row">
                    {playing ? (
                        <button className="btn" type="button" onClick={onPause}><Pause size={18} /> Pause</button>
                    ) : (
                        <button className="btn" type="button" onClick={onTrain} disabled={loading}>
                            <Play size={18} /> {loading ? 'Starting…' : cur && snap < last ? 'Continue' : 'Train the network'}
                        </button>
                    )}
                    <button className="btn btn-secondary" type="button" onClick={onTrained} disabled={snap == null}>
                        <RotateCcw size={16} /> The trained network
                    </button>
                </div>
                {error && <div className="small bad-text">{error}</div>}
                {history && (
                    <div className="slider-container in-row">
                        <div className="slider-header">
                            <span>epoch</span>
                            <span className="in-val">{fmtEpoch((cur ?? history[last]).epoch)} of {epochs}</span>
                        </div>
                        <input type="range" min={0} max={last} step={1} value={snap ?? last} onChange={(e) => onSeek(Number(e.target.value))}
                            aria-label="epoch" />
                    </div>
                )}
                <div className="metrics-grid">
                    <div className="metric-card">
                        <div className="metric-value">{cur ? cur.J.toFixed(3) : history ? history[last].J.toFixed(3) : '—'}</div>
                        <div className="metric-label" style={{ textTransform: 'none' }}>cost J on the training images</div>
                    </div>
                    <div className="metric-card">
                        <div className="metric-value">{pct(cur ? cur.accuracy : MODEL.accuracy.test)}</div>
                        <div className="metric-label" style={{ textTransform: 'none' }}>right on the 10 000 test images</div>
                    </div>
                </div>
            </div>
            <div className="panel">
                {history ? (
                    <Plot
                        data={[
                            { x: upTo.map((h) => h.epoch), y: upTo.map((h) => h.J), name: 'cost J', mode: 'lines+markers', line: { color: COLORS.quiet, width: 3 }, marker: { size: 5 }, hovertemplate: 'epoch %{x}<br>J = %{y:.3f}<extra></extra>' },
                            { x: upTo.map((h) => h.epoch), y: upTo.map((h) => 100 * h.accuracy), name: 'accuracy (%)', yaxis: 'y2', mode: 'lines+markers', line: { color: COLORS.go, width: 3 }, marker: { size: 5 }, hovertemplate: 'epoch %{x}<br>%{y:.1f} % right<extra></extra>' },
                        ]}
                        layout={{
                            ...baseLayout(pTheme),
                            showlegend: true,
                            legend: { orientation: 'h', x: 0, y: 1.14 },
                            margin: { t: 30, r: 55, l: 55, b: 45 },
                            xaxis: { title: { text: 'epoch' }, range: [-0.5, epochs + 0.5], gridcolor: pTheme.gridColor, zeroline: false },
                            yaxis: { title: { text: 'cost J' }, range: [0, 2.4], gridcolor: pTheme.gridColor, zeroline: false },
                            yaxis2: { title: { text: 'accuracy (%)' }, range: [0, 100], overlaying: 'y', side: 'right', showgrid: false, zeroline: false },
                        }}
                        config={PLOT_CONFIG} useResizeHandler style={{ width: '100%', height: '300px' }}
                    />
                ) : (
                    <div className="train-empty muted">Press <strong>Train the network</strong> to watch it learn, from random weights to the trained network.</div>
                )}
            </div>
        </div>
    );
}

// Step 4: a network that reads handwritten digits. Draw one, or pick a test image, and watch every neuron.
export default function MnistView({ dark, pTheme }) {
    const pad = useRef(null);
    const [x, setX] = useState(SAMPLES[FIRST].pixels);
    const [sample, setSample] = useState(FIRST); // the test image shown; null while you draw
    const [expanded, setExpanded] = useState(true);
    const [hov, setHov] = useState(null);
    const [tiles, setTiles] = useState('w'); // the layer-1 images: 'w' (the weights) or 'xw' (your drawing × the weights)
    const [center, setCenter] = useState(false); // resize and center the drawing as in MNIST before it goes in
    useEffect(() => {
        if (sample == null) pad.current?.refresh();
    }, [center]); // eslint-disable-line react-hooks/exhaustive-deps
    // Training: the network at each moment of its training (null until it is needed), and the moment shown
    // (null: the trained network).
    const [history, setHistory] = useState(null);
    const [snap, setSnap] = useState(null);
    const [playing, setPlaying] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const net = history && snap != null ? history[snap] : MODEL;
    useEffect(() => {
        if (!playing || !history) return undefined;
        if (snap >= history.length - 1) {
            setPlaying(false);
            return undefined;
        }
        const gap = history[snap + 1].epoch - history[snap].epoch;
        const id = setTimeout(() => setSnap(snap + 1), snap === 0 ? 1000 : Math.max(250, EPOCH_MS * gap));
        return () => clearTimeout(id);
    }, [playing, snap, history]);
    const train = async () => {
        setError(null);
        setLoading(true);
        try {
            const h = history ?? await loadHistory();
            setHistory(h);
            if (snap == null || snap >= h.length - 1) setSnap(0);
            setPlaying(true);
        } catch (e) {
            setError(`The training could not start (${e.message}).`);
        } finally {
            setLoading(false);
        }
    };

    const fw = useMemo(() => forward(x, net), [x, net]);
    const weights = useMemo(() => Array.from({ length: H1 }, (_, j) => weightsOf(j, net)), [net]);
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
    const tileValues = useMemo(() => (tiles === 'w' ? weights : weights.map((w) => w.map((v, i) => v * x[i]))), [tiles, x, weights]);

    return (
        <>
            <div className="glass-panel mnist-intro">
                <h3>Reading handwritten digits</h3>
                <div>
                    A picture of a digit is <strong>28 × 28 = 784 pixels</strong>, each a number from 0 (black) to 1 (white). The network
                    reads them as 784 inputs, row by row, passes them through two hidden layers of {H1} and {H2} neurons, and gives one
                    chance for each digit from 0 to 9. It learned its {(784 * H1 + H1 + H1 * H2 + H2 + H2 * 10 + 10).toLocaleString()} weights
                    and biases from the 60 000 training images of the MNIST dataset, and gets {pct(MODEL.accuracy.test)} of 10 000 new test
                    images right.
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
                    <DrawPad ref={pad} onInput={setX} onStart={() => setSample(null)} center={center}
                        onPixel={(i) => setHov(i == null || center ? null : { type: 'x', i })} />
                    <Toggle checked={center} onChange={setCenter} color={COLORS.go}>Resize and center the digit, as in MNIST</Toggle>
                    <div className="hint">
                        {center
                            ? 'Your drawing is resized to fit 20 × 20 pixels and moved to the center, as the MNIST images were: small or off-center digits are recognized better, but the image no longer matches where you draw.'
                            : 'Each small square of the grid is one of the 28 × 28 pixels, exactly where you draw it: the bottom-right corner is x₇₈₄. Draw large and in the middle, as in the MNIST images.'}
                    </div>
                    <div className="answer">
                        {blank ? <span className="muted">Draw a digit in the black square.</span> : (
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
                        {snap != null && history && (
                            <span className="train-badge">epoch {fmtEpoch(history[snap].epoch)} · {pct(history[snap].accuracy)} right</span>
                        )}
                        <button className="btn btn-secondary btn-small" type="button" onClick={() => setExpanded(!expanded)}>
                            {expanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                            {expanded ? 'Show the image as one block' : 'Expand the image into its 784 inputs'}
                        </button>
                    </div>
                    <MnistDiagram fw={fw} net={net} expanded={expanded} hov={hov} onHover={setHov} dark={dark} />
                    <div className="info-card">
                        {expanded ? (
                            <>
                                The image is laid out <strong>row by row</strong>: row 1 gives the inputs <Sym b="x" j={1} /> to{' '}
                                <Sym b="x" j={28} />, row 2 gives <Sym b="x" j={29} /> to <Sym b="x" j={56} />, … and row 28 ends with{' '}
                                <Sym b="x" j={784} />. Only the first and the last inputs are drawn (the top-left and bottom-right corners of the
                                image); point at any pixel, here or on the drawing, to see its input and its lines.{' '}
                            </>
                        ) : 'All 784 pixels go into every neuron of layer 1: 784 × 16 weights. '}
                        Brighter neurons are <strong>more active</strong>. Each line shows its signal, activation × weight:{' '}
                        <span style={{ color: COLORS.pos }}>orange</span> pushes the next neuron up,{' '}
                        <span style={{ color: COLORS.neg }}>blue</span> holds it back. Point at a neuron to see its value and its lines.
                    </div>
                </div>
            </div>

            <TrainingPanel history={history} snap={snap} playing={playing} loading={loading} error={error} pTheme={pTheme}
                onTrain={train} onPause={() => setPlaying(false)}
                onSeek={(k) => { setPlaying(false); setSnap(k); }}
                onTrained={() => { setPlaying(false); setSnap(null); }} />

            <div className="glass-panel gallery-panel">
                <div className="panel-header">
                    <h3>What each neuron of layer 1 looks at</h3>
                    <ViewTabs options={[{ key: 'w', label: 'its weights' }, { key: 'xw', label: 'your drawing × its weights' }]} value={tiles} onChange={setTiles} />
                </div>
                <div className="gallery">
                    {tileValues.map((vals, j) => {
                        const lit = Math.min(1, fw.a1[j] / net.scale[0][j]);
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
