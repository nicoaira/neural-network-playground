"""Build mnist_keras.ipynb: the Step 4 network of the playground (784 -> 16 -> 16 -> 10), coded with Keras, with
interactive Plotly figures and ipywidgets panels. It runs on Google Colab (CPU or GPU) and in Jupyter.

    python build_notebook.py          # writes mnist_keras.ipynb next to this file (no outputs)

All the code for the plots and the panels is in one hidden cell (a Colab form, collapsed in Jupyter), so that the
visible cells only show the neural network. Edit the cells here and rebuild; do not edit the notebook by hand.
"""
import pathlib

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip()))
code = lambda s: cells.append(new_code_cell(s.strip()))


def hidden(title, s):
    """A cell whose code is hidden: a form with a title on Colab, a collapsed input in Jupyter."""
    cells.append(new_code_cell(f"#@title {title}\n" + s.strip(),
                               metadata={"cellView": "form", "jupyter": {"source_hidden": True}}))


md(r"""
# Reading handwritten digits with Keras

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicoaira/neural-network-playground/blob/main/notebooks/mnist_keras.ipynb)

In Step 4 of the [neural network playground](https://nicoaira.github.io/neural-network-playground/#digits) a small
network reads handwritten digits: **784 inputs** (the 28 × 28 pixels of an image), **two hidden layers of 16
neurons**, and **10 outputs**, one chance per digit. In this notebook you build and train the same network with
Keras, and you look inside it with interactive panels.

Run the cells in order (**Shift + Enter**). The panels have menus and sliders: move them and the plots update. On the
plots you can point at anything to read its values, and zoom or pan.
""")

md(r"""
## 0. Setup

**Run the next cell first.** It installs what is missing (on Google Colab only ipycanvas, the drawing canvas of
section 7) and, on Colab, turns on the interactive panels. On your own computer it installs nothing if you already
have TensorFlow, Plotly, ipywidgets and ipycanvas (see `requirements.txt`).

**CPU or GPU.** The notebook runs on both, with nothing to change: TensorFlow trains on a GPU when there is one. On
Colab you choose in *Runtime → Change runtime type* (CPU, or a GPU such as the T4). This network is so small that the
CPU is about as fast: the training takes about a minute either way.
""")

code(r"""
# Installs what is missing, and on Google Colab turns on the interactive panels.
import importlib.util, subprocess, sys

PACKAGES = {"tensorflow": "tensorflow", "plotly": "plotly", "ipywidgets": "ipywidgets", "ipycanvas": "ipycanvas>=0.13"}
missing = [pip_name for module, pip_name in PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    print("installing", ", ".join(missing), "...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
if "google.colab" in sys.modules:
    from google.colab import output
    output.enable_custom_widget_manager()     # Colab needs this to show widgets such as the drawing canvas
print("ready")
""")

code(r"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # hide TensorFlow's technical start-up messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input

keras.utils.set_random_seed(0)      # the same random start every time you run the notebook
print("TensorFlow", tf.__version__, "| Keras", keras.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Keras will train on", f"the GPU: {gpus[0].name}" if gpus else "the CPU (no GPU found)")
""")

md(r"""
The next cell holds the code that draws the plots and builds the interactive panels. It is hidden because it is not
about neural networks: **run it**, there is no need to read it (on Colab, *Show code* opens it).
""")

hidden("Helper functions for the plots and the panels (run this cell; no need to read it)", r'''
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output

pio.templates.default = "plotly_white"
ORANGE, BLUE, GREEN, GRAY, VIOLET = "#f97316", "#3b82f6", "#10b981", "#94a3b8", "#8b5cf6"
SIGNED = [[0, BLUE], [0.5, "#f8fafc"], [1, ORANGE]]     # blue: negative, white: 0, orange: positive
STATE = {}                                              # what the panels share (the test image chosen in Panel 4)


def _image_axes(fig, row=None, col=None):
    """Square axes without ticks, row 1 at the top (images are always the first plot of a figure)."""
    fig.update_yaxes(autorange="reversed", scaleanchor="x", showticklabels=False, row=row, col=col)
    fig.update_xaxes(showticklabels=False, constrain="domain", row=row, col=col)


def _gray(img):
    return go.Heatmap(z=img, colorscale="gray", zmin=0, zmax=255 if img.max() > 1 else 1, showscale=False, hoverinfo="skip")


def _chances(p, right=None):
    """The 10 chances as horizontal bars: the answer green, the right digit orange when the answer is wrong."""
    colors = [GREEN if d == p.argmax() else ORANGE if d == right else GRAY for d in range(10)]
    return go.Bar(x=100 * p, y=[str(d) for d in range(10)], orientation="h", marker_color=colors,
                  text=[f"{100 * v:.1f} %" for v in p], textposition="outside", hovertemplate="digit %{y}: %{x:.1f} %<extra></extra>")


def _panel(controls, show):
    """Widgets on top, and below them the plot, redrawn whenever a widget changes."""
    display(widgets.VBox([widgets.HBox(list(controls.values())), widgets.interactive_output(show, controls)]))


# ── Panel 1: browse the images ──
def image_figure(img, title="", size=340):
    rows, cols = np.mgrid[1:29, 1:29]
    fig = go.Figure(go.Heatmap(
        z=img, colorscale="gray", zmin=0, zmax=255 if img.max() > 1 else 1, showscale=False,
        customdata=np.dstack([rows, cols, 28 * (rows - 1) + cols]),
        hovertemplate="row %{customdata[0]}, column %{customdata[1]}<br>becomes input x%{customdata[2]}<br>value %{z}<extra></extra>"))
    _image_axes(fig)
    fig.update_layout(title=title, width=size, height=size + 40, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def browse_images(x, y):
    def show(digit, image):
        k = np.flatnonzero(y == digit)[image]
        image_figure(x[k], f"training image {k}: a {digit}").show()
    _panel({"digit": widgets.Dropdown(options=list(range(10)), value=3, description="digit"),
            "image": widgets.IntSlider(value=0, min=0, max=99, description="image")}, show)


# ── Panel 2: the 784 inputs of one image ──
def inputs_figure(X, y, k, row):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.26, 0.74], horizontal_spacing=0.06,
                        subplot_titles=(f"training image {k} (a {y[k]})", "its 784 inputs, row after row"))
    fig.add_trace(_gray(X[k].reshape(28, 28)), 1, 1)
    fig.add_shape(type="rect", x0=-0.5, x1=27.5, y0=row - 1.5, y1=row - 0.5, line=dict(color=ORANGE, width=3), row=1, col=1)
    n = np.arange(1, 785)
    img_row, img_col = (n - 1) // 28 + 1, (n - 1) % 28 + 1
    colors = np.where(img_row == row, ORANGE, np.where(img_row % 2 == 1, BLUE, GRAY))
    fig.add_trace(go.Bar(x=n, y=X[k], marker_color=colors, customdata=np.stack([img_row, img_col], axis=-1),
                         hovertemplate="x%{x}: row %{customdata[0]}, column %{customdata[1]}<br>value %{y:.2f}<extra></extra>"), 1, 2)
    _image_axes(fig, 1, 1)
    fig.update_xaxes(title="input number i", range=[0, 785], row=1, col=2)
    fig.update_yaxes(title="value of the input", range=[0, 1.05], row=1, col=2)
    fig.update_layout(height=340, showlegend=False, bargap=0, margin=dict(l=10, r=10, t=50, b=40))
    return fig


def show_inputs(X, y):
    _panel({"k": widgets.IntSlider(value=0, min=0, max=99, description="image"),
            "row": widgets.IntSlider(value=14, min=1, max=28, description="row")},
           lambda k, row: inputs_figure(X, y, k, row).show())


# ── Panel 3: training, live ──
class LivePlot(keras.callbacks.Callback):
    """A Keras callback. After every epoch it keeps the cost J, the accuracy on the validation (here: test) images and
    a copy of all the weights, and it redraws the plot."""

    def __init__(self, epochs):
        super().__init__()
        self.total = epochs
        self.epochs, self.J, self.test_acc, self.weights = [], [], [], []
        self.out = widgets.Output()
        display(self.out)
        with self.out:
            print("Training: the plot appears after the first epoch...")

    def on_train_begin(self, logs=None):
        self.weights.append(self.model.get_weights())          # epoch 0: the random start

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.J.append(logs["loss"])
        self.test_acc.append(logs["val_accuracy"])
        self.weights.append(self.model.get_weights())
        with self.out:
            clear_output(wait=True)
            self.figure().show()

    def figure(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=self.epochs, y=self.J, name="cost J (training images)", mode="lines+markers",
                                 line=dict(color=VIOLET, width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=self.epochs, y=[100 * a for a in self.test_acc], name="right on the test images (%)",
                                 mode="lines+markers", line=dict(color=GREEN, width=3)), secondary_y=True)
        fig.update_xaxes(title="epoch", range=[0, self.total + 0.5])
        fig.update_yaxes(title="cost J", range=[0, max(self.J) * 1.1], secondary_y=False)
        low = min(80, 5 * np.floor(100 * min(self.test_acc) / 5))
        fig.update_yaxes(title="accuracy (%)", range=[low, 100], tickvals=list(range(int(low), 101, 5)), showgrid=False,
                         secondary_y=True)
        fig.update_layout(height=380, legend=dict(orientation="h", y=1.15), margin=dict(l=10, r=10, t=40, b=40))
        return fig


# ── Which digits get confused ──
def show_confusion(y_true, y_pred):
    counts = np.zeros((10, 10), dtype=int)
    np.add.at(counts, (y_true, y_pred), 1)
    fig = go.Figure(go.Heatmap(z=counts, x=list(range(10)), y=list(range(10)), colorscale="Blues", text=counts,
                               texttemplate="%{text}", hovertemplate="a %{y} read as %{x}: %{z} images<extra></extra>"))
    fig.update_xaxes(title="the network says", dtick=1, side="top")
    fig.update_yaxes(title="the image is a", dtick=1, autorange="reversed", scaleanchor="x")
    fig.update_layout(width=520, height=520, margin=dict(l=10, r=10, t=60, b=10))
    fig.show()


# ── Panel 4: one test image, every neuron ──
def _neurons(a, name):
    return go.Heatmap(z=a.reshape(-1, 1), y=[f"{name} neuron {j + 1}" for j in range(len(a))], colorscale="Greens",
                      zmin=0, showscale=False, texttemplate="%{z:.1f}", hovertemplate="%{y}: %{z:.2f}<extra></extra>")


def inside_figure(X, y, A1, A2, P, k):
    p = P[k]
    fig = make_subplots(rows=1, cols=4, column_widths=[0.3, 0.12, 0.12, 0.46], horizontal_spacing=0.05,
                        subplot_titles=(f"test image {k}: a {y[k]}", "layer 1", "layer 2", f"the network says {p.argmax()}"))
    fig.add_trace(_gray(X[k].reshape(28, 28)), 1, 1)
    fig.add_trace(_neurons(A1[k], "layer 1"), 1, 2)
    fig.add_trace(_neurons(A2[k], "layer 2"), 1, 3)
    fig.add_trace(_chances(p, right=y[k]), 1, 4)
    _image_axes(fig, 1, 1)
    for col in (2, 3):
        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=1, col=col)
    fig.update_xaxes(title="chance (%)", range=[0, 115], row=1, col=4)
    fig.update_yaxes(title="digit", autorange="reversed", row=1, col=4)
    fig.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=50, b=40))
    return fig


def show_inside(X, y, A1, A2, P):
    wrong = np.flatnonzero(P.argmax(axis=1) != y)
    subset = widgets.ToggleButtons(options=["all test images", "only the mistakes"], value="all test images")
    pick = widgets.IntSlider(value=0, min=0, max=len(y) - 1, description="image")

    def on_subset(change):
        pick.max = len(wrong) - 1 if subset.value == "only the mistakes" else len(y) - 1
        pick.value = 0
    subset.observe(on_subset, "value")
    STATE.update(subset=subset, pick=pick, wrong=wrong)
    _panel({"which": subset, "k": pick},
           lambda which, k: inside_figure(X, y, A1, A2, P, wrong[k] if which == "only the mistakes" else k).show())


def chosen_image():
    """The test image chosen in Panel 4 (0 before Panel 4 runs)."""
    if "pick" not in STATE:
        return 0
    k = STATE["pick"].value
    return STATE["wrong"][k] if STATE["subset"].value == "only the mistakes" else k


# ── Panel 5: the weights of layer 1, epoch after epoch ──
def weights_figure(log, X, epoch, show, k):
    W1 = log.weights[epoch][0]                              # 784 × 16, the network after that epoch
    fig = make_subplots(rows=2, cols=8, horizontal_spacing=0.01, vertical_spacing=0.08,
                        subplot_titles=[f"neuron {j + 1}" for j in range(16)])
    for j in range(16):
        w = W1[:, j] * (X[k] if show == "weights × image" else 1)
        big = np.abs(w).max() or 1
        fig.add_trace(go.Heatmap(z=w.reshape(28, 28), x=np.arange(1, 29), y=np.arange(1, 29), colorscale=SIGNED,
                                 zmin=-big, zmax=big, showscale=False,
                                 hovertemplate=f"neuron {j + 1}" + "<br>row %{y}, column %{x}<br>%{z:.3f}<extra></extra>"),
                      j // 8 + 1, j % 8 + 1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    acc = "random weights" if epoch == 0 else f"{100 * log.test_acc[epoch - 1]:.1f} % of the test images right"
    fig.update_layout(height=430, title=f"layer 1 after epoch {epoch}: {acc}", margin=dict(l=10, r=10, t=70, b=10))
    return fig


def show_weights(log, X):
    controls = {"epoch": widgets.IntSlider(value=log.total, min=0, max=log.total, description="epoch"),
                "show": widgets.ToggleButtons(options=["weights", "weights × image"], value="weights")}
    if "pick" in STATE:                                     # redraw also when the image of Panel 4 changes
        controls.update(which=STATE["subset"], k=STATE["pick"])
    _panel(controls, lambda epoch, show, **_: weights_figure(log, X, epoch, show, chosen_image()).show())


# ── Panel 6: draw a digit ──
class DrawPad:
    """A 280 × 280 canvas to draw on with the mouse. Each 10 × 10 square becomes one of the 28 × 28 pixels.
    predict(x) gets the drawing as 1 × 784 inputs and returns the 10 chances."""
    SIZE, BRUSH = 280, 10

    def __init__(self, predict):
        from ipycanvas import Canvas
        self.predict = predict
        self.canvas = Canvas(width=self.SIZE, height=self.SIZE)
        self.ink = np.zeros((self.SIZE, self.SIZE), dtype="float32")      # the same drawing, kept in Python
        self.yy, self.xx = np.mgrid[0:self.SIZE, 0:self.SIZE]
        self.last = None
        self.result = widgets.Output()
        button = widgets.Button(description="Clear", icon="eraser")
        button.on_click(lambda _: self.clear())
        self.canvas.on_mouse_down(self.on_down)
        self.canvas.on_mouse_move(self.on_move)
        self.canvas.on_mouse_up(self.on_up)
        self.canvas.on_mouse_out(self.on_up)
        self.clear()
        display(widgets.HBox([widgets.VBox([self.canvas, button]), self.result]))

    def clear(self):
        self.ink[:] = 0
        c = self.canvas
        c.fill_style = "black"
        c.fill_rect(0, 0, self.SIZE, self.SIZE)
        c.fill_style = c.stroke_style = "white"
        c.line_width, c.line_cap = 2 * self.BRUSH, "round"
        self.read()

    def stroke(self, x0, y0, x1, y1):
        self.canvas.stroke_line(x0, y0, x1, y1)
        n = int(np.hypot(x1 - x0, y1 - y0) // (self.BRUSH / 2)) + 1
        for t in np.linspace(0, 1, n + 1):
            cx, cy = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            self.ink[(self.xx - cx) ** 2 + (self.yy - cy) ** 2 <= self.BRUSH ** 2] = 1

    def on_down(self, x, y):
        self.canvas.fill_circle(x, y, self.BRUSH)
        self.stroke(x, y, x, y)
        self.last = (x, y)

    def on_move(self, x, y):
        if self.last is not None:
            self.stroke(*self.last, x, y)
            self.last = (x, y)

    def on_up(self, x, y):
        if self.last is not None:
            self.last = None
            self.read()

    def pixels(self):
        return self.ink.reshape(28, 10, 28, 10).mean(axis=(1, 3)).reshape(1, 784)

    def read(self):
        x = self.pixels()
        with self.result:
            clear_output(wait=True)
            if x.max() == 0:
                print("Draw a digit in the black square.")
                return
            p = np.asarray(self.predict(x))
            fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("what the network gets", f"it says {p.argmax()}"))
            fig.add_trace(_gray(x.reshape(28, 28)), 1, 1)
            fig.add_trace(_chances(p), 1, 2)
            _image_axes(fig, 1, 1)
            fig.update_xaxes(title="chance (%)", range=[0, 115], row=1, col=2)
            fig.update_yaxes(autorange="reversed", row=1, col=2)
            fig.update_layout(width=560, height=330, showlegend=False, margin=dict(l=10, r=10, t=50, b=40))
            fig.show()


def draw_digit(predict):
    STATE["pad"] = DrawPad(predict)


print("helper functions ready")
''')

md(r"""
## 1. The images

MNIST has **60 000 training images** and **10 000 test images** of handwritten digits. Each image is 28 × 28 pixels
with a gray level from 0 (black) to 255 (white), and comes with its label: the digit it shows. Keras downloads the
dataset the first time (about 11 MB).
""")

code(r"""
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("training images:", x_train.shape, "  labels:", y_train.shape)
print("test images:    ", x_test.shape, "  labels:", y_test.shape)
print("the label of the first training image:", y_train[0])
""")

md(r"""
**Panel 1: browse the images.** Choose a digit, then move the slider to see different people's handwriting. Point
at a pixel: you see its row, its column, its gray level, and the number of the input it becomes (section 2).
""")

code(r"""
browse_images(x_train, y_train)
""")

md(r"""
## 2. From 28 × 28 pixels to 784 inputs

A neuron takes a list of numbers, not a table, so the image is laid out **row by row**: row 1 gives the inputs
$x_1 \dots x_{28}$, row 2 gives $x_{29} \dots x_{56}$, …, and row 28 ends with $x_{784}$. The pixel in row $r$,
column $c$ becomes the input number $i = 28\,(r-1) + c$. We also divide by 255, so that every input is a number
from 0 to 1.

With NumPy this is one `reshape`. Careful: Python counts from 0, so $x_1$ is column `0` and $x_{784}$ is column
`783`.
""")

code(r"""
X_train = x_train.reshape(-1, 784).astype("float32") / 255     # one row per image: 60000 × 784
X_test = x_test.reshape(-1, 784).astype("float32") / 255       # 10000 × 784
print("X_train:", X_train.shape, "  X_test:", X_test.shape)

r, c = 14, 16                       # a pixel in the middle of the first image
i = 28 * (r - 1) + c                # its input number
print(f"row {r}, column {c} -> input x{i} = {X_train[0, i - 1]:.3f}  (the pixel itself: {x_train[0, r - 1, c - 1]} / 255)")
""")

md(r"""
**Panel 2: the 784 inputs of one image.** Left, the image; right, its 784 inputs in order (the colors alternate from
one row of the image to the next). Choose a row: it is outlined in the image and its 28 inputs turn orange.
""")

code(r"""
show_inputs(X_train, y_train)
""")

md(r"""
## 3. The network

784 inputs → 16 neurons → 16 neurons → 10 outputs. In Keras each layer is a `Dense` layer: every neuron receives
every number of the layer before. The last layer uses **softmax**: its 10 outputs are chances, one per digit, that add
up to 1.
""")

code(r"""
model = keras.Sequential([
    Input(shape=(784,)),                  # layer 0: the 784 inputs
    Dense(16, activation="relu"),         # layer 1: 16 neurons
    Dense(16, activation="relu"),         # layer 2: 16 neurons
    Dense(10, activation="softmax"),      # layer 3: 10 outputs, the chance of each digit
])
model.summary()
""")

md(r"""
The network has **13 002 numbers to learn**: $784 \times 16 + 16$ in layer 1, $16 \times 16 + 16$ in layer 2 and
$16 \times 10 + 10$ in layer 3.

**Where are $\mathbf{W}$ and $\vec{b}$?** Each `Dense` layer keeps its weights as a matrix with **one row per input
and one column per neuron**, and its biases as a vector: exactly the $\mathbf{W}^{[l]}$ and $\vec{b}^{[l]}$ of the
slides, with $\vec{z}^{[l]} = \vec{a}^{[l-1]}\,\mathbf{W}^{[l]} + \vec{b}^{[l]}$. Right now they are random: the
network has not learned anything yet.
""")

code(r"""
for l, layer in enumerate(model.layers, start=1):
    W, b = layer.get_weights()
    print(f"layer {l}: W{l} {W.shape}, b{l} {b.shape}")
W1, b1 = model.layers[0].get_weights()
print("the weight from input x1 into neuron 1 of layer 1:", W1[0, 0])     # row 0 = input 1, column 0 = neuron 1
""")

md(r"""
## 4. Training

The **cost J** is the average loss over the training images: for 10 classes, the cross-entropy loss of logistic
regression extended to 10 outputs. Training repeats **gradient descent** steps on batches of 128 images; Adam is a
faster variant of gradient descent. The learning rate starts at 0.003 and shrinks to 0 along a cosine curve. One pass
over the 60 000 training images is an **epoch**. On a GPU the numbers can differ slightly from a CPU run: the
computer adds the same numbers in a different order.
""")

code(r"""
EPOCHS = 20
BATCH = 128
steps = EPOCHS * int(np.ceil(len(X_train) / BATCH))     # the number of gradient descent steps
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.CosineDecay(0.003, steps)),
    loss="sparse_categorical_crossentropy",            # the labels are digits 0-9, not lists of 10 numbers
    metrics=["accuracy"],
)
""")

md(r"""
**Panel 3: training, live.** `fit` trains the network. `LivePlot` is a Keras *callback* (defined in the helper cell):
Keras calls it after every epoch, and it redraws the plot and keeps a copy of all the weights, to look at them later
(Panel 5). The test images are passed as `validation_data` only to watch the accuracy: nothing is chosen with them.
""")

code(r"""
log = LivePlot(EPOCHS)
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_data=(X_test, y_test), callbacks=[log], verbose=0)
print(f"after {EPOCHS} epochs: cost J {log.J[-1]:.3f}, {100 * log.test_acc[-1]:.1f} % of the test images right")
""")

md(r"""
## 5. How good is it?

`model.predict` gives the 10 chances for every test image; the answer is the digit with the largest chance.
""")

code(r"""
P = model.predict(X_test, verbose=0)          # 10000 × 10: the chance of each digit, for every test image
pred = P.argmax(axis=1)                       # the answer: the digit with the largest chance
print(f"right: {np.mean(pred == y_test):.2%}   mistakes: {np.sum(pred != y_test)} of {len(y_test)}")
""")

md(r"""
**Which digits get confused?** Each square counts the test images of one digit (row) that the network read as
another (column). The diagonal holds the right answers. Point at a square to read it.
""")

code(r"""
show_confusion(y_test, pred)
""")

md(r"""
## 6. Inside the network

Keras did the forward propagation for us. Here it is with NumPy, layer after layer: $\vec{z} = \vec{a}\,\mathbf{W} +
\vec{b}$, then the activation of the layer. It gives the same chances as Keras.
""")

code(r"""
def forward(X, weights):
    # Forward propagation with NumPy: the activations of layers 1 and 2, and the 10 chances.
    W1, b1, W2, b2, W3, b3 = weights
    A1 = np.maximum(0, X @ W1 + b1)
    A2 = np.maximum(0, A1 @ W2 + b2)
    Z3 = A2 @ W3 + b3
    E = np.exp(Z3 - Z3.max(axis=-1, keepdims=True))      # softmax
    return A1, A2, E / E.sum(axis=-1, keepdims=True)

A1, A2, P_numpy = forward(X_test, model.get_weights())
print("the same chances as Keras:", np.allclose(P_numpy, P, atol=1e-5))
""")

md(r"""
**Panel 4: one test image, every neuron.** Choose a test image, or only the mistakes. You see the image, the 16 + 16
hidden neurons (brighter green = more active) and the 10 chances: the answer is green, and when the network is wrong
the right digit is orange.
""")

code(r"""
show_inside(X_test, y_test, A1, A2, P_numpy)
""")

md(r"""
**Panel 5: what each neuron of layer 1 looks at, epoch after epoch.** A neuron of layer 1 has one weight per pixel,
so its 784 weights (a column of $\mathbf{W}^{[1]}$) can be drawn as a 28 × 28 image: **orange** where a white pixel
pushes the neuron up, **blue** where it holds it back. Move the epoch slider: at epoch 0 the weights are random, and
training shapes them. With *weights × image* you see the image of Panel 4 through each neuron: only its white pixels
count.

The speckles around the edges are not learned: 67 pixels, all near the border, are black in every one of the 60 000
training images. Their inputs are always 0, so their gradient is always 0 and their weights keep their random start.
""")

code(r"""
show_weights(log, X_test)
""")

md(r"""
## 7. Draw your own digit

**Panel 6.** Draw a digit in the black square with the mouse: large and in the middle, like the MNIST digits. Each
10 × 10 square of the drawing becomes one of the 28 × 28 pixels. When you release the mouse, `draw_digit` calls the
function you give it with your drawing as 1 × 784 inputs: here, `model.predict`, which returns the 10 chances.
**Clear** starts again.
""")

code(r"""
draw_digit(lambda x: model.predict(x, verbose=0)[0])
""")

md(r"""
## Summary

- An MNIST image is 28 × 28 pixels; laid out row by row it becomes **784 inputs** from 0 to 1.
- The network is three `Dense` layers: 784 → 16 → 16 → 10. Keras keeps each layer's weights as a matrix
  $\mathbf{W}^{[l]}$ with one row per input and one column per neuron, as in the slides.
- `compile` chooses the cost and the optimizer; `fit` runs gradient descent, epoch after epoch; `predict` does the
  forward propagation, which you can write yourself with NumPy in a few lines.
- After training, each neuron of layer 1 has its own pattern of weights over the pixels. Nobody drew them: they come
  out of training.
""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
path = HERE / "mnist_keras.ipynb"
nbformat.write(nb, path)
print(f"wrote {path} ({len(cells)} cells)")
