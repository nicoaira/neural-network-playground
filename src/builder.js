// Step 5: build a small network and train it here, in the browser, on a pattern with two input features.
// The network: 2 inputs (x₁, x₂) → 0 to 4 hidden layers → 1 output neuron, f = σ(z), the chance of class 1 (orange).
// Hidden neurons: a = max(0, z). Cost: the logistic loss of Part 06, averaged over the training points.
// Training: gradient descent with Adam, on all the training points at once (one step = one epoch).

export const RANGE = 1.5;            // the plane shown: x₁ and x₂ from −1.5 to 1.5

export function rng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
const gauss = (rand) => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

// The patterns: n points, half of each class; noise is the spread added to every point.
export const DATASETS = [
    {
        key: 'moons', name: 'Two moons',
        point(rand, c) {
            const t = Math.PI * rand();
            const [x, y] = c === 0 ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), 0.5 - Math.sin(t)];
            return [(x - 0.5) * 0.8, (y - 0.25) * 0.8 + 0.05];
        },
    },
    {
        key: 'rings', name: 'A circle inside a ring',
        point(rand, c) {
            const r = c === 1 ? 0.5 * Math.sqrt(rand()) : 0.8 + 0.35 * rand();
            const a = 2 * Math.PI * rand();
            return [r * Math.cos(a), r * Math.sin(a)];
        },
    },
    {
        key: 'spirals', name: 'Two spirals',
        point(rand, c) {
            const t = 0.12 + 0.88 * rand();
            const a = 1.6 * 2 * Math.PI * t + c * Math.PI;
            return [1.15 * t * Math.cos(a), 1.15 * t * Math.sin(a)];
        },
    },
];

// Points of a pattern, split into training (70 %) and test (30 %) points: { x1, x2, y }.
export function makeData(key, noise, seed = 7, n = 260) {
    const ds = DATASETS.find((d) => d.key === key);
    const rand = rng(seed);
    const pts = Array.from({ length: n }, (_, i) => {
        const c = i % 2;
        const [x1, x2] = ds.point(rand, c);
        return { x1: x1 + noise * gauss(rand), x2: x2 + noise * gauss(rand), y: c };
    });
    for (let i = pts.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [pts[i], pts[j]] = [pts[j], pts[i]];
    }
    const cut = Math.round(0.7 * n);
    return { train: pts.slice(0, cut), test: pts.slice(cut) };
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// A fully connected network. sizes: [2, hidden…, 1]. W[l][k * nOut + j] is the weight from neuron k of the layer
// before into neuron j of layer l + 1 (row k, column j of the course's W^[l+1]); b[l][j] its bias.
export class Net {
    constructor(sizes, seed = 1) {
        this.sizes = sizes;
        const rand = rng(seed);
        this.W = [];
        this.b = [];
        for (let l = 0; l < sizes.length - 1; l++) {
            const nIn = sizes[l];
            const nOut = sizes[l + 1];
            const last = l === sizes.length - 2;
            const lim = last ? Math.sqrt(6 / (nIn + nOut)) : Math.sqrt(6 / nIn);
            this.W.push(Float64Array.from({ length: nIn * nOut }, () => (2 * rand() - 1) * lim));
            this.b.push(new Float64Array(nOut).fill(last ? 0 : 0.1));
        }
        this.mW = this.W.map((w) => new Float64Array(w.length));
        this.vW = this.W.map((w) => new Float64Array(w.length));
        this.mb = this.b.map((v) => new Float64Array(v.length));
        this.vb = this.b.map((v) => new Float64Array(v.length));
        this.t = 0;
        this.acts = sizes.map((n) => new Float64Array(n));       // the activations of the last forward pass
        this.zs = sizes.map((n) => new Float64Array(n));
    }

    // Forward pass for one point; the activations of every layer stay in this.acts. Returns f.
    forward(x1, x2) {
        const { sizes, W, b, acts, zs } = this;
        acts[0][0] = x1;
        acts[0][1] = x2;
        for (let l = 0; l < sizes.length - 1; l++) {
            const nIn = sizes[l];
            const nOut = sizes[l + 1];
            const last = l === sizes.length - 2;
            for (let j = 0; j < nOut; j++) {
                let z = b[l][j];
                for (let k = 0; k < nIn; k++) z += acts[l][k] * W[l][k * nOut + j];
                zs[l + 1][j] = z;
                acts[l + 1][j] = last ? sigmoid(z) : z > 0 ? z : 0;
            }
        }
        return acts[sizes.length - 1][0];
    }

    // Cost J (average logistic loss) and accuracy on some points.
    evaluate(points) {
        let J = 0;
        let right = 0;
        for (const p of points) {
            const f = Math.min(1 - 1e-12, Math.max(1e-12, this.forward(p.x1, p.x2)));
            J -= p.y ? Math.log(f) : Math.log(1 - f);
            if ((f >= 0.5 ? 1 : 0) === p.y) right++;
        }
        return { J: J / points.length, accuracy: right / points.length };
    }

    // One step of gradient descent (Adam) on all the points. Returns the cost J before the step.
    step(points, lr) {
        const { sizes, W, b, acts, zs } = this;
        const L = sizes.length - 1;
        const gW = W.map((w) => new Float64Array(w.length));
        const gb = b.map((v) => new Float64Array(v.length));
        const delta = sizes.map((n) => new Float64Array(n));
        let J = 0;
        for (const p of points) {
            const f = this.forward(p.x1, p.x2);
            const fc = Math.min(1 - 1e-12, Math.max(1e-12, f));
            J -= p.y ? Math.log(fc) : Math.log(1 - fc);
            delta[L][0] = f - p.y;                                     // dJ/dz of the output (logistic loss + σ)
            for (let l = L - 1; l >= 0; l--) {
                const nIn = sizes[l];
                const nOut = sizes[l + 1];
                for (let j = 0; j < nOut; j++) {
                    const d = delta[l + 1][j];
                    gb[l][j] += d;
                    for (let k = 0; k < nIn; k++) gW[l][k * nOut + j] += acts[l][k] * d;
                }
                if (l > 0) {
                    for (let k = 0; k < nIn; k++) {
                        let s = 0;
                        for (let j = 0; j < nOut; j++) s += W[l][k * nOut + j] * delta[l + 1][j];
                        delta[l][k] = zs[l][k] > 0 ? s : 0;
                    }
                }
            }
        }
        const n = points.length;
        this.t++;
        const c1 = 1 - 0.9 ** this.t;
        const c2 = 1 - 0.999 ** this.t;
        const adam = (w, g, m, v) => {
            for (let i = 0; i < w.length; i++) {
                const gi = g[i] / n;
                m[i] = 0.9 * m[i] + 0.1 * gi;
                v[i] = 0.999 * v[i] + 0.001 * gi * gi;
                w[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
            }
        };
        for (let l = 0; l < L; l++) {
            adam(W[l], gW[l], this.mW[l], this.vW[l]);
            adam(b[l], gb[l], this.mb[l], this.vb[l]);
        }
        return J / n;
    }

    // Every neuron over an N × N grid of the plane (row 0 at the top: x₂ = +RANGE).
    // maps[l][j] is a Float32Array(N * N): the activation of neuron j of layer l (layer 0 = the inputs).
    grid(N) {
        const maps = this.sizes.map((n) => Array.from({ length: n }, () => new Float32Array(N * N)));
        for (let r = 0; r < N; r++) {
            const x2 = RANGE - ((r + 0.5) * 2 * RANGE) / N;
            for (let c = 0; c < N; c++) {
                const x1 = -RANGE + ((c + 0.5) * 2 * RANGE) / N;
                this.forward(x1, x2);
                for (let l = 0; l < this.sizes.length; l++) for (let j = 0; j < this.sizes[l]; j++) maps[l][j][r * N + c] = this.acts[l][j];
            }
        }
        return maps;
    }
}

// The line where a map crosses `level`, as segments [x0, y0, x1, y1] in grid units (marching squares).
export function contour(map, N, level = 0.5) {
    const segs = [];
    const v = (r, c) => map[r * N + c] - level;
    const lerp = (a, b) => a / (a - b);
    for (let r = 0; r < N - 1; r++) {
        for (let c = 0; c < N - 1; c++) {
            const a = v(r, c);
            const b2 = v(r, c + 1);
            const c2 = v(r + 1, c + 1);
            const d = v(r + 1, c);
            const pts = [];
            if ((a > 0) !== (b2 > 0)) pts.push([c + lerp(a, b2), r]);
            if ((b2 > 0) !== (c2 > 0)) pts.push([c + 1, r + lerp(b2, c2)]);
            if ((d > 0) !== (c2 > 0)) pts.push([c + lerp(d, c2), r + 1]);
            if ((a > 0) !== (d > 0)) pts.push([c, r + lerp(a, d)]);
            if (pts.length === 2) segs.push([...pts[0], ...pts[1]]);
            else if (pts.length === 4) segs.push([...pts[0], ...pts[1]], [...pts[2], ...pts[3]]);
        }
    }
    return segs;
}
