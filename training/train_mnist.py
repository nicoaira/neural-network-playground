"""Train the small MNIST network of Step 4 and export it for the playground.

    784 inputs (28 x 28 pixels) -> hidden layer 1 -> hidden layer 2 -> 10 outputs (the digits 0-9)

The hidden layers use ReLU, the output softmax (cross-entropy). Training uses the GPU when there is one.
The 60 000 training images are split into 55 000 for training and 5 000 for choosing the best epoch; the
10 000 test images are used once, at the end, to report the accuracy.

Light augmentation (small rotations, shifts, size changes and thicker strokes) makes the network cope better
with digits drawn with a mouse, which are less tidy than the scanned MNIST ones.

Usage (from this folder; MNIST is downloaded into ./data the first time):
    python train_mnist.py                       # 16 + 16 hidden neurons, writes ../src/mnist/model.json
    python train_mnist.py --hidden 32 32        # a larger network
    python train_mnist.py --no-augment --out /tmp/plain.json

The JSON holds the weights in the orientation of the course (z = x W + b): W1 is 784 x h1 (row k = input k,
column j = neuron j), W2 is h1 x h2, W3 is h2 x 10. "scale" gives, for each hidden neuron, the activation that
the playground draws as fully lit (its 99th percentile on the test images); "samples" are a few test images
(3 per digit, all classified correctly), as base64 of 784 bytes.
"""
import argparse
import base64
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

HERE = pathlib.Path(__file__).resolve().parent


def load(device):
    """MNIST as float tensors on the device: images (N, 1, 28, 28) in [0, 1] and labels (N,)."""
    out = []
    for train in (True, False):
        ds = datasets.MNIST(HERE / "data", train=train, download=True)
        x = torch.tensor(ds.data.numpy(), dtype=torch.float32).div_(255).unsqueeze(1)
        y = torch.tensor(ds.targets.numpy(), dtype=torch.long)
        out += [x.to(device), y.to(device)]
    return out


def augment(x, gen):
    """Random small rotation (±12°), size change (×0.9–1.1), shift (±2 px), and thicker strokes for a third."""
    n = x.shape[0]
    dev = x.device
    ang = (torch.rand(n, device=dev, generator=gen) * 2 - 1) * math.radians(12)
    sc = 1 / (0.9 + 0.2 * torch.rand(n, device=dev, generator=gen))
    tx, ty = ((torch.rand(2, n, device=dev, generator=gen) * 2 - 1) * (2 / 14))
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    thick = torch.rand(n, device=dev, generator=gen) < 0.33
    if thick.any():
        x[thick] = F.max_pool2d(x[thick], 3, stride=1, padding=1) * 0.7 + x[thick] * 0.3
    return x.clamp_(0, 1)


def make_model(h1, h2):
    return nn.Sequential(nn.Flatten(), nn.Linear(784, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 10))


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    return (model(x).argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, nargs=2, default=[16, 16], metavar=("H1", "H2"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=HERE.parent / "src" / "mnist" / "model.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    x_all, y_all, x_test, y_test = load(device)
    perm = torch.randperm(x_all.shape[0], generator=torch.Generator().manual_seed(args.seed)).to(device)
    x_tr, y_tr = x_all[perm[5000:]], y_all[perm[5000:]]
    x_val, y_val = x_all[perm[:5000]], y_all[perm[:5000]]

    h1, h2 = args.hidden
    model = make_model(h1, h2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = args.epochs * math.ceil(x_tr.shape[0] / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    best = (-1.0, None, 0)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(x_tr.shape[0], device=device, generator=gen)
        for i in range(0, x_tr.shape[0], args.batch):
            idx = order[i:i + args.batch]
            xb = x_tr[idx] if args.no_augment else augment(x_tr[idx], gen)
            loss = F.cross_entropy(model(xb), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        val = accuracy(model, x_val, y_val)
        if val > best[0]:
            best = (val, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {loss.item():.3f}  validation accuracy {val:.4f}  ({time.time() - t0:.0f} s)")

    model.load_state_dict(best[1])
    test_acc = accuracy(model, x_test, y_test)
    train_acc = accuracy(model, x_tr, y_tr)
    print(f"best epoch {best[2]}: validation {best[0]:.4f}, training {train_acc:.4f}, TEST {test_acc:.4f}")

    # Export: weights in the course orientation, display scales, a few test images.
    lin = [m for m in model if isinstance(m, nn.Linear)]
    with torch.no_grad():
        a1 = F.relu(lin[0](x_test.flatten(1)))
        a2 = F.relu(lin[1](a1))
        pred = lin[2](a2).argmax(1)
    # How bright to draw each hidden neuron: fully lit at its own 99th percentile on the test images.
    scale = [[round(max(float(q), 1e-3), 3) for q in torch.quantile(a, 0.99, dim=0)] for a in (a1, a2)]
    dead = [int((a.max(0).values == 0).sum()) for a in (a1, a2)]
    r4 = lambda t: np.round(t.detach().cpu().numpy().astype(float), 4).tolist()
    samples = []
    for digit in range(10):
        ok = ((y_test == digit) & (pred == digit)).nonzero().flatten()[:3].tolist()
        for i in ok:
            img = (x_test[i, 0].cpu().numpy() * 255).round().astype(np.uint8)
            samples.append({"label": digit, "index": i, "pixels": base64.b64encode(img.tobytes()).decode()})
    out = {
        "about": "Step 4 of the neural network playground: a small MLP trained on MNIST (training/train_mnist.py).",
        "layers": [784, h1, h2, 10],
        "W1": r4(lin[0].weight.T), "b1": r4(lin[0].bias),
        "W2": r4(lin[1].weight.T), "b2": r4(lin[1].bias),
        "W3": r4(lin[2].weight.T), "b3": r4(lin[2].bias),
        "scale": scale,
        "accuracy": {"test": round(test_acc, 4), "training": round(train_acc, 4), "validation": round(best[0], 4)},
        "training": {"epochs": args.epochs, "best_epoch": best[2], "batch": args.batch, "lr": args.lr,
                     "augment": not args.no_augment, "seed": args.seed, "torch": torch.__version__},
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB); neurons never active on the test images: {dead}")


if __name__ == "__main__":
    main()
