# Neural Network Playground

Interactive companion to the Deep Learning class (neural networks: from one neuron to a network, and its notation) of
an introductory machine-learning course.

**Live page:** https://nicoaira.github.io/neural-network-playground/

The network answers one question: **should I go to the beach this afternoon?** Eight inputs (sunshine, temperature,
sea temperature, wind, rain forecast, weekend or holiday, late afternoon, towel color), a hidden layer of two neurons
(*will it be nice?* and *will it be quiet?*) and one output neuron that goes only when the afternoon is nice **and**
quiet.

- **Step 1: Play with the beach network.** Change any input with the sliders and buttons, or load afternoons A, B and C
  of the slides, and see every neuron light up. Point at any part of the drawing to see its name in the notation of
  the course and its value: a line is a weight w^[l]_{j,k}, the Σ half of a neuron its weighted sum z, the σ half its
  activation a, the small badge its bias b. The lines can show the weights or the signals w × x. A plot shows what
  happens to both hidden neurons and to the answer when only one input changes.
- **Step 2: Inside the computation.**
  - *One neuron*: for each neuron, the inputs paired with their weights, the products, the dot product w⃗ · x⃗, the
    bias and the sigmoid, with a button that adds one term at a time. Columns and lines light up together.
  - *A whole layer*: z⃗ = x⃗ W + b⃗ and a⃗ = σ(z⃗) as matrices, for layer 1 (8 × 2) or layer 2 (2 × 1), for one
    example or for m = 4 examples at once (X W + B). Point at a weight in W and it lights up in the matrix, on its line
    in the network, and in the sum it belongs to; point at a z to see its row-times-column computation.

- **Step 3: The XOR problem.** y = 1 when exactly one of x₁, x₂ is 1.
  - *One neuron is not enough*: move the straight line of a single neuron (logistic regression); it always gets at
    least one of the four points wrong.
  - *Two hidden neurons*: on the input plane each hidden neuron draws one straight line and says yes on its hatched
    side (neuron 1 amber, neuron 2 violet). The output neuron combines their answers: with an AND the green region
    is the overlap of the two sides (where the hatches cross), a band no single line can draw. Next to the plot, the
    network for one of the four points and a table of a₁, a₂, f, ŷ and y; point at a neuron (on the network or at its
    card) to see its decision boundary on the plot, with its answer at each point, and click it (or its name) to keep
    the boundary there, like the checkboxes above the plot. Below, all nine numbers have
    sliders, with presets (OR + NAND → AND, the same sharper, two corners combined with an OR, two identical neurons
    that fail), and the labels say what each neuron computes ("at least one on? (OR)"). Switch the hidden activation
    off (a = z) to see the network collapse back to one straight line.

- **Step 4: Handwritten digits.** A small network trained on MNIST: 784 inputs (the 28 × 28 pixels, row by row), two
  hidden layers of 16 neurons and 10 outputs, one chance per digit. Draw a digit with the mouse, a finger or a pen (or
  pick one of 30 test images) and every neuron lights up as you draw, with the chance of each digit next to the
  outputs. The image can be expanded into its 784 inputs (x₁ … x₆, ⋮, x₇₇₉ … x₇₈₄); point at any pixel to bring its
  input and its lines into the column. Below, the 784 weights of each neuron of layer 1 are drawn as a 28 × 28 image,
  or multiplied by your drawing. It gets 95.7 % of the 10 000 test images right.

The weights of the beach network are the illustrative ones of the slides, chosen by hand (not trained). The colors of the indices follow
the slides: layer [l] orange, neuron j blue, input k gray, example (i) green.

## The digits network

`training/train_mnist.py` trains it (PyTorch and torchvision, on the GPU when there is one) and writes
`src/mnist/model.json`, which the page loads. MNIST is downloaded into `training/data/` (not in git).

```bash
cd training
python train_mnist.py                  # 16 + 16 hidden neurons: 95.65 % of the test images (about 25 s on a laptop GPU)
python train_mnist.py --hidden 32 32   # 32 + 32: 97.7 %
```

5 000 of the 60 000 training images are kept aside to choose the best epoch; the 10 000 test images are used once, at
the end. Small random rotations, shifts, size changes and thicker strokes during training help with digits drawn on
screen.

## Run locally

```bash
npm install
npm run dev
```

Pushing to `main` builds the site and deploys it to GitHub Pages (`.github/workflows/deploy.yml`).
