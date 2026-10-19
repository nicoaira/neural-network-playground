import React from 'react';

// A symbol in the notation of the course, with its indices in the colors of the slides:
// the layer [l] orange, the neuron j blue, the input k gray, the example (i) green.
//   <Sym b="w" l={1} j={2} k={8} />   w with subscript 2,8 and superscript [1]
//   <Sym b="x" vec />                 x with an arrow (a vector: one example)
//   <Sym b="W" mat l={1} />           bold W^[1] (a matrix)
// x_k is written <Sym b="x" j={k} />: its index is a neuron of layer 0, blue as on the slides.
export function Sym({ b, vec = false, mat = false, l, j, k, i }) {
    const hasSup = l != null || i != null;
    const hasSub = j != null || k != null;
    const sup = hasSup && (
        <>
            {l != null && <span className="n-layer">[{l}]</span>}
            {i != null && <span className="n-ex">({i})</span>}
        </>
    );
    const sub = hasSub && (
        <>
            {j != null && <span className="n-unit">{j}</span>}
            {j != null && k != null && <span className="n-input">,</span>}
            {k != null && <span className="n-input">{k}</span>}
        </>
    );
    return (
        <span className={`sym${mat ? ' mat' : ''}`}>
            <span className={`sym-b${vec ? ' vec' : ''}`}>{b}</span>
            {hasSup && hasSub ? (
                <span className="ss"><span>{sup}</span><span>{sub}</span></span>
            ) : hasSup ? <sup>{sup}</sup> : hasSub ? <sub>{sub}</sub> : null}
        </span>
    );
}

// The prediction ŷ (y-hat).
export const Yhat = () => <span className="sym"><span className="sym-b">ŷ</span></span>;
