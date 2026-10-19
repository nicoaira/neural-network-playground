// The beach network of the Deep Learning class: "Should I go to the beach this afternoon?"
// Eight inputs, a hidden layer of two neurons (will it be nice? will it be quiet?) and one output neuron.
// The weights are the illustrative ones of the slides, chosen by hand (not trained): the same numbers as
// slides/DeepLearning-class/figures/make_slide_figures.py in the course material.

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// A number as it is shown: 2 decimals and a real minus sign (never "−0.00").
export const num = (v, d = 2) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d).replace('-', '−');
// A weight or a bias: whole numbers without decimals (3, −2), the others with 2 (−0.95).
export const wnum = (v) => (Number.isInteger(v) ? String(v).replace('-', '−') : num(v));
// The same, in parentheses when negative, for products and sums: 0.20 × (−2).
export const par = (s, v) => (v < 0 && s !== '0' && s !== '0.00' ? `(${s})` : s);

// Hours as a number (17.5) → "17:30".
export const hhmm = (h) => `${String(Math.floor(h)).padStart(2, '0')}:${h % 1 ? '30' : '00'}`;

const sunWords = (v) => (v <= 0.15 ? 'cloudy' : v < 0.45 ? 'mostly cloudy' : v < 0.8 ? 'some sun' : v < 1 ? 'sunny' : 'full sun');

// The towel enters the network as a code: a color is not a number, and its weights are 0.
export const TOWELS = [
    { name: 'yellow', code: 0.5, color: '#f6c453' },
    { name: 'blue', code: 0.2, color: '#7aa7e0' },
    { name: 'red', code: 0.8, color: '#e06a5a' },
];
export const towel = (name) => TOWELS.find((t) => t.name === name);

// The eight inputs, in the order of the network (x₁ … x₈). `toX` turns what you know into the number the
// network gets (from 0 to 1); `show` says what you know in words, `how` how it becomes x.
export const INPUTS = [
    {
        name: 'sunshine', short: 'sun', kind: 'range', min: 0, max: 1, step: 0.05, unit: '',
        toX: (v) => v, show: sunWords, how: () => 'from 0 (cloudy) to 1 (full sun)',
    },
    {
        name: 'temperature', short: 'temp.', kind: 'range', min: 15, max: 35, step: 1, unit: '°C',
        toX: (v) => (v - 15) / 20, show: (v) => `${v} °C`, how: (v) => `(${v} − 15) / 20 = ${num((v - 15) / 20)}`,
    },
    {
        name: 'sea temperature', short: 'sea', kind: 'range', min: 15, max: 27, step: 0.5, unit: '°C',
        toX: (v) => (v - 15) / 12, show: (v) => `${v} °C`, how: (v) => `(${v} − 15) / 12 = ${num((v - 15) / 12)}`,
    },
    {
        name: 'wind', short: 'wind', kind: 'range', min: 0, max: 40, step: 1, unit: 'km/h',
        toX: (v) => v / 40, show: (v) => `${v} km/h`, how: (v) => `${v} / 40 = ${num(v / 40)}`,
    },
    {
        name: 'rain forecast', short: 'rain', kind: 'range', min: 0, max: 100, step: 5, unit: '%',
        toX: (v) => v / 100, show: (v) => `${v} %`, how: (v) => `${v} / 100 = ${num(v / 100)}`,
    },
    {
        name: 'weekend or holiday', short: 'weekend', kind: 'bool',
        toX: (v) => (v ? 1 : 0), show: (v) => (v ? 'yes' : 'no'), how: (v) => (v ? 'yes → 1' : 'no → 0'),
    },
    {
        name: 'late afternoon', short: 'late', kind: 'time', min: 10, max: 20, step: 0.5, unit: 'h',
        toX: (v) => (v > 17 ? 1 : 0), show: (v) => `arrive at ${hhmm(v)}`,
        how: (v) => `arrive at ${hhmm(v)}: ${v > 17 ? 'after 17:00 → 1' : 'not after 17:00 → 0'}`,
    },
    {
        name: 'towel color', short: 'towel', kind: 'towel',
        toX: (v) => towel(v).code, show: (v) => v, how: (v) => `${v} → ${towel(v).code} (a code: a color is not a number)`,
    },
];

export const HIDDEN = [
    { name: 'will it be nice?', short: 'nice?' },
    { name: 'will it be quiet?', short: 'quiet?' },
];
export const OUTPUT = 'go to the beach?';

// The signs of the weights table on the slides: +++ = 3, ++ = 2, + = 1, ~0 = 0. Neuron 2 uses them
// times 0.95. With these biases the hidden neurons of afternoons A, B and C light up as on the slides:
// 0.90 / 0.90, 0.95 / 0.05 and 0.00 / 1.00.
const NICE = [3, 2, 2, -2, -3, 0, 0, 0];
const QUIET = [-1, -2, 0, 1, 1, -3, 2, 0];
// W1[k][j]: row k = input k + 1, column j = neuron j + 1, as W^[1] on the slides (8 × 2).
// The code counts from 0; the notation counts from 1.
export const W1 = NICE.map((w, k) => [w, Math.round(95 * QUIET[k]) / 100]);
export const B1 = [-3.05, 2.3];
// The output neuron: f = σ(6 a₁ + 6 a₂ − 9), high only when it is nice AND quiet. W^[2] is 2 × 1.
export const W2 = [[6], [6]];
export const B2 = [-9];

// What you know about an afternoon: [sunshine, °C, sea °C, km/h, rain %, weekend?, arrival hour, towel].
export const PRESETS = [
    { key: 'today', label: 'This afternoon', detail: 'a sunny Saturday, arriving at 14:00', raw: [1, 32, 24, 8, 10, true, 14, 'yellow'] },
    { key: 'A', label: 'Afternoon A', detail: 'a sunny Tuesday, arriving at 17:30', raw: [1, 28, 23, 8, 0, false, 17.5, 'yellow'] },
    { key: 'B', label: 'Afternoon B', detail: 'a sunny Sunday, arriving at 13:00', raw: [1, 33, 25, 8, 0, true, 13, 'blue'] },
    { key: 'C', label: 'Afternoon C', detail: 'a rainy Tuesday, arriving at 17:30', raw: [0.1, 19, 20, 35, 80, false, 17.5, 'red'] },
];
export const sameRaw = (a, b) => a.every((v, k) => v === b[k]);

export const toX = (raw) => raw.map((v, k) => INPUTS[k].toX(v));

// Forward propagation, one layer after the other: z = x⃗ W + b⃗, a = σ(z).
export function forward(x) {
    const z1 = [0, 1].map((j) => x.reduce((s, xk, k) => s + xk * W1[k][j], B1[j]));
    const a1 = z1.map(sigmoid);
    const z2 = a1.reduce((s, aj, j) => s + aj * W2[j][0], B2[0]);
    const a2 = sigmoid(z2);
    return { x, z1, a1, z2, a2, yhat: a2 >= 0.5 ? 1 : 0 };
}

// The numbers of one layer for one example: its inputs (x⃗ or a⃗^[1]), weights, biases, z and a.
export const layerOf = (fw, l) => (l === 1
    ? { ins: fw.x, W: W1, b: B1, z: fw.z1, a: fw.a1 }
    : { ins: fw.a1, W: W2, b: B2, z: [fw.z2], a: [fw.a2] });
