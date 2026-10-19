import React, { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { preprocess, downsample, paint, SIZE } from './mnist';

const W = 280;     // the drawing, in canvas pixels (10 × the 28 × 28 image)
const BRUSH = 20;

// A square to draw a digit with the mouse, a finger or a pen: white on black, like the MNIST images.
// onInput(x) receives the 784 inputs while you draw: each 10 × 10 square of the drawing is one pixel, where you drew
// it; with center, the digit is first resized and centered as in MNIST. onPixel(i) reports the pixel under the
// pointer (null when it leaves). Through its ref: clear(), show(pixels) to put a 28 × 28 image on it, and refresh().
const DrawPad = forwardRef(function DrawPad({ onInput, onStart, onPixel, center = false }, ref) {
    const canvas = useRef(null);
    const last = useRef(null);
    const frame = useRef(0);
    const ctx = () => canvas.current.getContext('2d', { willReadFrequently: true });
    const blank = () => {
        const c = ctx();
        c.fillStyle = '#000';
        c.fillRect(0, 0, W, W);
    };
    useEffect(() => {
        blank();
        return () => cancelAnimationFrame(frame.current);
    }, []);

    // At most once per frame: read the drawing and prepare the 784 inputs.
    const send = () => {
        cancelAnimationFrame(frame.current);
        frame.current = requestAnimationFrame(() => {
            const d = ctx().getImageData(0, 0, W, W).data;
            const gray = new Float32Array(W * W);
            for (let i = 0; i < W * W; i++) gray[i] = d[4 * i] / 255;
            onInput(center ? preprocess(gray, W, W) : downsample(gray, W, W));
        });
    };
    useImperativeHandle(ref, () => ({
        clear: blank,
        refresh: send,
        show: (pixels) => {
            const small = document.createElement('canvas');
            small.width = SIZE;
            small.height = SIZE;
            paint(small, pixels, 'gray');
            blank();
            const c = ctx();
            c.imageSmoothingEnabled = true;
            c.drawImage(small, 0, 0, W, W);
        },
    }));

    const at = (e) => {
        const r = canvas.current.getBoundingClientRect();
        return [((e.clientX - r.left) * W) / r.width, ((e.clientY - r.top) * W) / r.height];
    };
    const line = (a, b) => {
        const c = ctx();
        c.strokeStyle = '#fff';
        c.lineWidth = BRUSH;
        c.lineCap = 'round';
        c.lineJoin = 'round';
        c.beginPath();
        c.moveTo(...a);
        c.lineTo(...b);
        c.stroke();
        send();
    };
    const down = (e) => {
        canvas.current.setPointerCapture(e.pointerId);
        onStart?.();
        last.current = at(e);
        line(last.current, last.current);
    };
    const cell = (v) => Math.min(SIZE - 1, Math.max(0, Math.floor((v * SIZE) / W)));
    const move = (e) => {
        const p = at(e);
        onPixel?.(cell(p[1]) * SIZE + cell(p[0]));
        if (!last.current) return;
        line(last.current, p);
        last.current = p;
    };
    const up = () => {
        last.current = null;
    };

    return (
        <div className={`draw-wrap ${center ? '' : 'grid'}`}>
            <canvas ref={canvas} width={W} height={W} className="draw-pad" aria-label="Draw a digit here"
                onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onPointerLeave={() => onPixel?.(null)} />
        </div>
    );
});

export default DrawPad;
