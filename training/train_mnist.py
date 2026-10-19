"""Train the small MNIST network of Step 4 and export it for the playground.

    784 inputs (28 x 28 pixels) -> hidden layer 1 -> hidden layer 2 -> 10 outputs (the digits 0-9)

The hidden layers use ReLU, the output softmax (cross-entropy). Training uses the GPU when there is one, on the
60 000 training images; the 10 000 test images are only used to report the accuracy (nothing is chosen with them).
The learning rate starts at --lr and decreases to 0 along a cosine over the epochs.

Light augmentation (small rotations, shifts, size changes and thicker strokes) makes the network cope better
with digits drawn with a mouse, which are less tidy than the scanned MNIST ones.

Usage (from this folder; MNIST is downloaded into ./data the first time):
    python train_mnist.py                       # 16 + 16 hidden neurons
    python train_mnist.py --hidden 32 32        # a larger network
    python train_mnist.py --no-augment --out /tmp/plain.json --history /tmp/plain-history.json

Two files are written:
  --out (../src/mnist/model.json): the network after the last epoch, which the playground uses. The weights are
      in the orientation of the course (z = x W + b): W1 is 784 x h1 (row k = input k, column j = neuron j), W2 is
      h1 x h2, W3 is h2 x 10. "scale" gives, for each hidden neuron, the activation drawn as fully lit (its 99th
      percentile on the test images); "samples" are 3 test images per digit (classified correctly), as base64 of
      784 bytes.
  --history (../public/mnist/history.json): the same network at the start (random weights), after 1/8, 1/4 and
      1/2 of the first epoch and after every epoch, with the cost J on the training images and the accuracy on the
      test images. Weights are stored as 16-bit integers: {"s": step, "q": base64 of int16}, w = s * q.
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


def linears(model):
    return [m for m in model if isinstance(m, nn.Linear)]


@torch.no_grad()
def measure(model, x_tr, y_tr, x_test, y_test):
    """Cost J (average cross-entropy) on the training images, accuracy on the test images, and the display scale
    of every hidden neuron (its 99th percentile on the test images)."""
    model.eval()
    J = sum(F.cross_entropy(model(x_tr[i:i + 10000]), y_tr[i:i + 10000], reduction="sum").item()
            for i in range(0, x_tr.shape[0], 10000)) / x_tr.shape[0]
    acc = (model(x_test).argmax(1) == y_test).float().mean().item()
    lin = linears(model)
    a1 = F.relu(lin[0](x_test.flatten(1)))
    a2 = F.relu(lin[1](a1))
    scale = [[round(max(float(q), 1e-3), 3) for q in torch.quantile(a, 0.99, dim=0)] for a in (a1, a2)]
    model.train()
    return J, acc, scale


def weights(model):
    """The weights in the course orientation (z = x W + b), as numpy arrays."""
    lin = linears(model)
    out = {}
    for n, m in enumerate(lin, start=1):
        out[f"W{n}"] = m.weight.detach().T.cpu().numpy().astype(np.float64)
        out[f"b{n}"] = m.bias.detach().cpu().numpy().astype(np.float64)
    return out


def quantize(w):
    """A weight matrix as 16-bit integers: w ≈ s * q."""
    s = max(float(np.abs(w).max()), 1e-12) / 32767
    q = np.round(w / s).astype("<i2")
    return {"s": s, "q": base64.b64encode(q.tobytes()).decode()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, nargs=2, default=[16, 16], metavar=("H1", "H2"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=HERE.parent / "src" / "mnist" / "model.json")
    ap.add_argument("--history", type=pathlib.Path, default=HERE.parent / "public" / "mnist" / "history.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    x_tr, y_tr, x_test, y_test = load(device)
    h1, h2 = args.hidden
    model = make_model(h1, h2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    per_epoch = math.ceil(x_tr.shape[0] / args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * per_epoch)
    # The moments recorded in the history: the start, 1/8, 1/4 and 1/2 of the first epoch, then every epoch.
    early = {round(per_epoch * f): f for f in (1 / 8, 1 / 4, 1 / 2)}

    history = []

    def record(epoch):
        J, acc, scale = measure(model, x_tr, y_tr, x_test, y_test)
        w = weights(model)
        history.append({
            "epoch": round(epoch, 4), "J": round(J, 4), "accuracy": round(acc, 4), "scale": scale,
            **{k: quantize(v) if k.startswith("W") else np.round(v, 4).tolist() for k, v in w.items()},
        })
        return J, acc

    record(0)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(x_tr.shape[0], device=device, generator=gen)
        for s, i in enumerate(range(0, x_tr.shape[0], args.batch), start=1):
            idx = order[i:i + args.batch]
            xb = x_tr[idx] if args.no_augment else augment(x_tr[idx], gen)
            loss = F.cross_entropy(model(xb), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if epoch == 1 and s in early:
                record(early[s])
        J, acc = record(epoch)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  cost J {J:.4f}  test accuracy {acc:.4f}  ({time.time() - t0:.0f} s)")

    # The network after the last epoch: the one the playground uses.
    J, test_acc, scale = measure(model, x_tr, y_tr, x_test, y_test)
    with torch.no_grad():
        model.eval()
        pred = model(x_test).argmax(1)
        train_acc = (model(x_tr).argmax(1) == y_tr).float().mean().item()
    print(f"after {args.epochs} epochs: cost J {J:.4f}, training accuracy {train_acc:.4f}, TEST accuracy {test_acc:.4f}")
    samples = []
    for digit in range(10):
        ok = ((y_test == digit) & (pred == digit)).nonzero().flatten()[:3].tolist()
        for i in ok:
            img = (x_test[i, 0].cpu().numpy() * 255).round().astype(np.uint8)
            samples.append({"label": digit, "index": i, "pixels": base64.b64encode(img.tobytes()).decode()})
    w = weights(model)
    out = {
        "about": "Step 4 of the neural network playground: a small MLP trained on MNIST (training/train_mnist.py).",
        "layers": [784, h1, h2, 10],
        **{k: np.round(v, 4).tolist() for k, v in w.items()},
        "scale": scale,
        "accuracy": {"test": round(test_acc, 4), "training": round(train_acc, 4)},
        "training": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "schedule": "cosine",
                     "augment": not args.no_augment, "seed": args.seed, "torch": torch.__version__},
        "samples": samples,
    }
    for path, data in ((args.out, out), (args.history, {"layers": [784, h1, h2, 10], "snapshots": history})):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, separators=(",", ":")))
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
