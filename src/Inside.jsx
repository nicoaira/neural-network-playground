import React, { useState } from 'react';
import { ViewTabs } from './ui';
import { PresetBar } from './Playground';
import NeuronView from './NeuronView';
import MatrixView from './MatrixView';

// Step 2: how the network computes its answer, one neuron at a time, then a whole layer as a matrix product.
export default function Inside({ raw, setRaw, fw, pTheme }) {
    const [view, setView] = useState('neuron');
    return (
        <>
            <div className="glass-panel sub-bar">
                <ViewTabs
                    options={[{ key: 'neuron', label: '1 · One neuron: dot product and sigmoid' }, { key: 'matrix', label: '2 · A whole layer: the matrix form' }]}
                    value={view} onChange={setView}
                />
                <div className="sub-bar-right">
                    <span className="muted small">Afternoon:</span>
                    <PresetBar raw={raw} setRaw={setRaw} />
                </div>
            </div>
            {view === 'neuron' ? <NeuronView fw={fw} pTheme={pTheme} /> : <MatrixView fw={fw} raw={raw} />}
        </>
    );
}
