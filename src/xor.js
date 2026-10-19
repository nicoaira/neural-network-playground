// XOR: y = 1 when exactly one of x₁, x₂ is 1 ("one or the other, but not both"), like a corridor lamp with a
// switch at each end. No straight line separates its four points; a hidden layer of two neurons does.
import { sigmoid } from './network';

export const XOR = [
    { x1: 0, x2: 0, y: 0 },
    { x1: 0, x2: 1, y: 1 },
    { x1: 1, x2: 0, y: 1 },
    { x1: 1, x2: 1, y: 0 },
];

// The nine numbers of the 2-2-1 network, named as in the course: w{l}{j}{k} = w^[l]_{j,k} (from input k into
// neuron j of layer l) and b{l}{j} = b^[l]_j.
export const PARAMS = [
    { key: 'w111', l: 1, j: 1, k: 1 }, { key: 'w112', l: 1, j: 1, k: 2 }, { key: 'b11', l: 1, j: 1 },
    { key: 'w121', l: 1, j: 2, k: 1 }, { key: 'w122', l: 1, j: 2, k: 2 }, { key: 'b12', l: 1, j: 2 },
    { key: 'w211', l: 2, j: 1, k: 1 }, { key: 'w212', l: 2, j: 1, k: 2 }, { key: 'b21', l: 2, j: 1 },
];

// Neuron 1 is an OR (at least one is 1?), neuron 2 a NAND (not both?), and the output an AND of the two:
// yes only where both hidden neurons say yes, the band between their lines.
export const PRESETS = [
    {
        key: 'and', label: 'OR + NAND → AND',
        p: { w111: 6, w112: 6, b11: -3, w121: -6, w122: -6, b12: 9, w211: 8, w212: 8, b21: -12 },
    },
    {
        key: 'sharp', label: 'The same, sharper',
        p: { w111: 20, w112: 20, b11: -10, w121: -20, w122: -20, b12: 30, w211: 20, w212: 20, b21: -30 },
    },
    {
        key: 'or', label: 'Another solution: two corners, OR',
        p: { w111: 6, w112: -6, b11: -3, w121: -6, w122: 6, b12: -3, w211: 8, w212: 8, b21: -4 },
    },
    {
        key: 'twins', label: 'Two identical neurons',
        p: { w111: 6, w112: 6, b11: -3, w121: 6, w122: 6, b12: -3, w211: 8, w212: 8, b21: -12 },
    },
];
export const ONE_START = { w1: 6, w2: 6, b: -3 };

// Forward pass. act: the activation of the hidden layer, 'sigmoid' or 'linear' (none: a = z).
export function xorForward(p, x1, x2, act = 'sigmoid') {
    const g = act === 'sigmoid' ? sigmoid : (v) => v;
    const z1 = p.w111 * x1 + p.w112 * x2 + p.b11;
    const z2 = p.w121 * x1 + p.w122 * x2 + p.b12;
    const a1 = g(z1);
    const a2 = g(z2);
    const z = p.w211 * a1 + p.w212 * a2 + p.b21;
    const f = sigmoid(z);
    return { z1, z2, a1, a2, z, f, yhat: f >= 0.5 ? 1 : 0 };
}

// What a neuron does with two yes/no inputs: its answer for (0,0), (0,1), (1,0), (1,1), and that rule as a short
// question. For a hidden neuron the inputs are x₁ and x₂ ("on" = 1); for the output neuron they are the answers
// of the two hidden neurons.
const ASK_X = {
    FFFT: ['AND', 'both on?'], FTTT: ['OR', 'at least one on?'], TTTF: ['NAND', 'not both on?'],
    TFFF: ['NOR', 'neither on?'], FTTF: ['XOR', 'exactly one on?'], TFFT: ['XNOR', 'both the same?'],
    FFTT: ['', 'x₁ on?'], FTFT: ['', 'x₂ on?'], TTFF: ['', 'x₁ off?'], TFTF: ['', 'x₂ off?'],
    FFTF: ['', 'only x₁ on?'], FTFF: ['', 'only x₂ on?'], TFTT: ['', 'x₁ on or x₂ off?'], TTFT: ['', 'x₂ on or x₁ off?'],
    FFFF: ['', 'never yes'], TTTT: ['', 'always yes'],
};
const ASK_NEURONS = {
    FFFT: ['AND', 'both say yes?'], FTTT: ['OR', 'at least one says yes?'], TTTF: ['NAND', 'not both say yes?'],
    TFFF: ['NOR', 'neither says yes?'], FTTF: ['XOR', 'exactly one says yes?'], TFFT: ['XNOR', 'both say the same?'],
    FFTT: ['', 'neuron 1 says yes?'], FTFT: ['', 'neuron 2 says yes?'], TTFF: ['', 'neuron 1 says no?'],
    TFTF: ['', 'neuron 2 says no?'], FFTF: ['', 'only neuron 1 says yes?'], FTFF: ['', 'only neuron 2 says yes?'],
    TFTT: ['', 'a mix of both answers'], TTFT: ['', 'a mix of both answers'], FFFF: ['', 'never yes'], TTTT: ['', 'always yes'],
};
const CORNERS = [[0, 0], [0, 1], [1, 0], [1, 1]];
function logicOf(yesAt, table) {
    const yes = CORNERS.map(([u, v]) => yesAt(u, v));
    const [name, question] = table[yes.map((v) => (v ? 'T' : 'F')).join('')];
    return { yes, name, question: name ? `${question} (${name})` : question };
}
// A hidden neuron z = w1·x₁ + w2·x₂ + b says yes where z ≥ 0 (σ(z) ≥ 0.5).
export const neuronLogic = (w1, w2, b) => logicOf((u, v) => w1 * u + w2 * v + b >= 0, ASK_X);
// The output neuron, if the hidden neurons answered exactly 0 or 1.
export const outputLogic = (p) => logicOf((u, v) => p.w211 * u + p.w212 * v + p.b21 >= 0, ASK_NEURONS);
// The whole network, on the four XOR points.
export const networkLogic = (p, act) => logicOf((u, v) => xorForward(p, u, v, act).yhat === 1, ASK_X);

export const range = (lo, hi, n) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));

// Points where the line a·x + c·y = t crosses the box [x0,x1]×[y0,y1] (at most 2 kept).
export const lineInBox = (a, c, t, [x0, x1], [y0, y1]) => {
    const pts = [];
    if (Math.abs(c) > 1e-12) {
        for (const x of [x0, x1]) {
            const y = (t - a * x) / c;
            if (y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]);
        }
    }
    if (Math.abs(a) > 1e-12) {
        for (const y of [y0, y1]) {
            const x = (t - c * y) / a;
            if (x > x0 + 1e-9 && x < x1 - 1e-9) pts.push([x, y]);
        }
    }
    return pts.slice(0, 2);
};

// Part of the box where sign·(a·x + c·y − t) ≥ 0, as a closed polygon (Sutherland–Hodgman, one edge).
export const halfPlaneInBox = (a, c, t, sign, [x0, x1], [y0, y1]) => {
    const corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    const inside = ([x, y]) => sign * (a * x + c * y - t) >= 0;
    const out = [];
    for (let i = 0; i < corners.length; i++) {
        const P = corners[i];
        const Q = corners[(i + 1) % corners.length];
        const pin = inside(P);
        const qin = inside(Q);
        if (pin) out.push(P);
        if (pin !== qin) {
            const fp = a * P[0] + c * P[1] - t;
            const fq = a * Q[0] + c * Q[1] - t;
            const s = fp / (fp - fq);
            out.push([P[0] + s * (Q[0] - P[0]), P[1] + s * (Q[1] - P[1])]);
        }
    }
    if (out.length) out.push(out[0]);
    return out;
};
