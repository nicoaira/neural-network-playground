// Step 4: a small network that reads handwritten digits (MNIST), trained by training/train_mnist.py.
// 784 inputs (the 28 × 28 pixels, row by row) → 16 → 16 → 10 outputs (one per digit).
import MODEL from './mnist/model.json';

export { MODEL };
export const SIZE = 28;
export const N_IN = SIZE * SIZE;
export const [, H1, H2, N_OUT] = MODEL.layers;

// The 28 × 28 test images that come with the model: { label, pixels: Float32Array(784) in [0, 1] }.
export const SAMPLES = MODEL.samples.map((s) => {
    const bytes = Uint8Array.from(atob(s.pixels), (c) => c.charCodeAt(0));
    return { label: s.label, index: s.index, pixels: Float32Array.from(bytes, (v) => v / 255) };
});

// Forward propagation through a network (the trained one, or one moment of its training), layer after layer
// (z = x W + b). Hidden layers: a = max(0, z). Output: softmax.
export function forward(x, net = MODEL) {
    const layer = (inp, W, b, hidden) => b.map((bj, j) => {
        let z = bj;
        for (let k = 0; k < inp.length; k++) if (inp[k] !== 0) z += inp[k] * W[k][j];
        return hidden ? Math.max(0, z) : z;
    });
    const a1 = layer(x, net.W1, net.b1, true);
    const a2 = layer(a1, net.W2, net.b2, true);
    const z3 = layer(a2, net.W3, net.b3, false);
    const m = Math.max(...z3);
    const e = z3.map((z) => Math.exp(z - m));
    const sum = e.reduce((s, v) => s + v, 0);
    const p = e.map((v) => v / sum);
    const best = p.indexOf(Math.max(...p));
    return { x, a1, a2, z3, p, best };
}

// A drawing (gray levels in [0, 1], w × h, w and h multiples of 28) → the 784 inputs, exactly where it was drawn:
// each pixel is the average of its square of the drawing (10 × 10 for a 280 × 280 drawing).
export function downsample(gray, w, h) {
    const out = new Float32Array(N_IN);
    const sx = w / SIZE;
    const sy = h / SIZE;
    for (let y = 0; y < h; y++) {
        const r = Math.floor(y / sy);
        for (let x = 0; x < w; x++) out[r * SIZE + Math.floor(x / sx)] += gray[y * w + x];
    }
    for (let i = 0; i < N_IN; i++) out[i] /= sx * sy;
    return out;
}

// A drawing (gray levels in [0, 1], w × h) → the 784 inputs, prepared the way MNIST was: the digit is scaled to fit
// a 20 × 20 box (keeping its proportions), then placed in the 28 × 28 image with its center of mass at the center.
export function preprocess(gray, w, h) {
    let x0 = w;
    let x1 = -1;
    let y0 = h;
    let y1 = -1;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (gray[y * w + x] > 0.1) {
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
        }
    }
    const out = new Float32Array(N_IN);
    if (x1 < 0) return out;
    const bw = x1 - x0 + 1;
    const bh = y1 - y0 + 1;
    const s = 20 / Math.max(bw, bh);
    const tw = Math.max(1, Math.round(bw * s));
    const th = Math.max(1, Math.round(bh * s));
    // Each small pixel is the average of the drawing pixels it covers.
    const box = new Float32Array(tw * th);
    for (let ty = 0; ty < th; ty++) {
        const sy0 = y0 + Math.floor(ty / s);
        const sy1 = Math.max(sy0 + 1, Math.min(y1 + 1, y0 + Math.floor((ty + 1) / s)));
        for (let tx = 0; tx < tw; tx++) {
            const sx0 = x0 + Math.floor(tx / s);
            const sx1 = Math.max(sx0 + 1, Math.min(x1 + 1, x0 + Math.floor((tx + 1) / s)));
            let sum = 0;
            for (let y = sy0; y < sy1; y++) for (let x = sx0; x < sx1; x++) sum += gray[y * w + x];
            box[ty * tw + tx] = sum / ((sy1 - sy0) * (sx1 - sx0));
        }
    }
    let m = 0;
    let cx = 0;
    let cy = 0;
    for (let ty = 0; ty < th; ty++) {
        for (let tx = 0; tx < tw; tx++) {
            const v = box[ty * tw + tx];
            m += v;
            cx += v * (tx + 0.5);
            cy += v * (ty + 0.5);
        }
    }
    const clamp = (v, hi) => Math.min(hi, Math.max(0, v));
    const ox = clamp(Math.round(SIZE / 2 - cx / m), SIZE - tw);
    const oy = clamp(Math.round(SIZE / 2 - cy / m), SIZE - th);
    let peak = 0;
    for (let ty = 0; ty < th; ty++) {
        for (let tx = 0; tx < tw; tx++) {
            const v = box[ty * tw + tx];
            out[(oy + ty) * SIZE + ox + tx] = v;
            if (v > peak) peak = v;
        }
    }
    if (peak > 0) for (let i = 0; i < N_IN; i++) out[i] = Math.min(1, out[i] / peak);
    return out;
}

// Column j of W1: the 784 weights of neuron j of layer 1, as an image.
export const weightsOf = (j, net = MODEL) => Float32Array.from(net.W1, (row) => row[j]);

// The training history (public/mnist/history.json, written by training/train_mnist.py): the network at the start
// and after each epoch, with the cost J on the training images and the accuracy on the test images. It is only
// downloaded when it is needed. Each moment becomes a network like MODEL: { epoch, J, accuracy, scale, W1, b1, … }.
let historyPromise = null;
const bytesOf = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
const matrix = ({ s, q }, rows, cols) => {
    const ints = new Int16Array(bytesOf(q).buffer);
    return Array.from({ length: rows }, (_, r) => Float32Array.from(ints.subarray(r * cols, (r + 1) * cols), (v) => v * s));
};
export function loadHistory() {
    historyPromise ??= fetch(`${import.meta.env.BASE_URL}mnist/history.json`)
        .then((r) => {
            if (!r.ok) throw new Error(`history.json: ${r.status}`);
            return r.json();
        })
        .then((h) => h.snapshots.map((sn) => ({
            epoch: sn.epoch, J: sn.J, accuracy: sn.accuracy, scale: sn.scale,
            W1: matrix(sn.W1, N_IN, H1), b1: sn.b1, W2: matrix(sn.W2, H1, H2), b2: sn.b2, W3: matrix(sn.W3, H2, N_OUT), b3: sn.b3,
        })))
        .catch((e) => {
            historyPromise = null;
            throw e;
        });
    return historyPromise;
}

// Paint a 28 × 28 image into a canvas: 'gray' (the digit, white on black) or 'signed' (orange +, blue −,
// scaled by the largest |value|).
export function paint(canvas, values, mode = 'gray', dark = true) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(SIZE, SIZE);
    let big = 0;
    if (mode === 'signed') for (const v of values) big = Math.max(big, Math.abs(v));
    const bg = dark ? [15, 23, 42] : [248, 250, 252];
    for (let i = 0; i < N_IN; i++) {
        let rgb;
        if (mode === 'gray') {
            const g = Math.round(255 * Math.min(1, Math.max(0, values[i])));
            rgb = [g, g, g];
        } else {
            const t = big > 0 ? Math.min(1, Math.abs(values[i]) / big) : 0;
            const c = values[i] > 0 ? [249, 115, 22] : [59, 130, 246];
            rgb = bg.map((b, n) => Math.round(b + (c[n] - b) * t));
        }
        img.data.set([...rgb, 255], 4 * i);
    }
    ctx.putImageData(img, 0, 0);
}
