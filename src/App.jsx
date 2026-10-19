import React, { useEffect, useMemo, useState } from 'react';
import { Sun, Moon } from 'lucide-react';
import { PRESETS, toX, forward } from './network';
import Playground from './Playground';
import Inside from './Inside';
import XorView from './XorView';
import { Sym } from './Sym';

const HASHES = ['#play', '#inside', '#xor'];
const stepOf = (hash) => Math.max(1, HASHES.indexOf(hash) + 1);

export default function App() {
    // The URL hash selects the tab, so a link can open a step directly (…/#inside, …/#xor).
    const [step, setStepState] = useState(() => stepOf(window.location.hash));
    const setStep = (s) => {
        setStepState(s);
        window.history.replaceState(null, '', HASHES[s - 1]);
    };
    const [theme, setTheme] = useState('dark');
    // What you know about the afternoon, shared by both steps (see INPUTS in network.js).
    const [raw, setRaw] = useState(() => [...PRESETS[0].raw]);
    const fw = useMemo(() => forward(toX(raw)), [raw]);

    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);
    useEffect(() => {
        const onHash = () => setStepState(stepOf(window.location.hash));
        window.addEventListener('hashchange', onHash);
        return () => window.removeEventListener('hashchange', onHash);
    }, []);

    const dark = theme === 'dark';
    const pTheme = {
        fontColor: dark ? '#e2e8f0' : '#1e293b',
        gridColor: dark ? '#334155' : '#cbd5e1',
    };

    return (
        <div className="app-container">
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                <div>
                    <div className="title">Neural Network Playground</div>
                    <div className="subtitle">Should I go to the beach this afternoon? A network with one hidden layer</div>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={() => setTheme(dark ? 'light' : 'dark')}
                    style={{ padding: '0.6rem', borderRadius: '50%', width: '42px', height: '42px', justifyContent: 'center', flexShrink: 0 }}
                    title={`Switch to ${dark ? 'light' : 'dark'} mode`}
                >
                    {dark ? <Sun size={20} /> : <Moon size={20} />}
                </button>
            </header>

            <div className="nav-tabs">
                <div className={`nav-tab ${step === 1 ? 'active' : ''}`} onClick={() => setStep(1)}>
                    Step 1: Play with the beach network
                </div>
                <div className={`nav-tab ${step === 2 ? 'active' : ''}`} onClick={() => setStep(2)}>
                    Step 2: Inside the computation
                </div>
                <div className={`nav-tab ${step === 3 ? 'active' : ''}`} onClick={() => setStep(3)}>
                    Step 3: The XOR problem
                </div>
            </div>

            {step === 1 && <Playground raw={raw} setRaw={setRaw} pTheme={pTheme} />}
            {step === 2 && <Inside raw={raw} setRaw={setRaw} fw={fw} pTheme={pTheme} />}
            {step === 3 && <XorView pTheme={pTheme} />}

            <div className="footer">
                Illustrative weights chosen by hand, not trained (the Deep Learning class slides). Every neuron computes
                a = σ(<Sym b="w" vec /> · <Sym b="x" vec /> + b), with the sigmoid σ(z) = 1 / (1 + e<sup>−z</sup>). The inputs go in as numbers from 0 to 1:
                sunshine 0 (cloudy) to 1 (full sun); temperature (T − 15) / 20; sea temperature (S − 15) / 12; wind / 40;
                rain forecast / 100; weekend or holiday and late afternoon (arriving after 17:00) 1 or 0; the towel color as a code.
            </div>
        </div>
    );
}
