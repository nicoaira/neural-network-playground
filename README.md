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

The weights are the illustrative ones of the slides, chosen by hand (not trained). The colors of the indices follow
the slides: layer [l] orange, neuron j blue, input k gray, example (i) green.

## Run locally

```bash
npm install
npm run dev
```

Pushing to `main` builds the site and deploys it to GitHub Pages (`.github/workflows/deploy.yml`).
