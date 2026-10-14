# Architecture notebooks

These notebooks are also executable chapters in the
[Sphinx tutorial book](https://rgs2151.github.io/NeuralJAXwork/docs/tutorials.html).
Documentation CI executes them in smoke mode and publishes their rendered code,
outputs, and figures to GitHub Pages after a successful main-branch build.

Install `python -m pip install -e '.[notebooks]'`, launch Jupyter from the repository
root, and select that environment's kernel. Open notebooks in numbered order or
choose a model family. Every notebook defaults to a tiny **offline synthetic smoke
run**, with fresh output under `runs/`. To train on the named dataset, set
`SMOKE=False`; use `DOWNLOAD=True` to permit missing dataset downloads.

The notebooks introduce the architecture, tensor shapes, objective, data/split
rules, controls, and visual interpretation. They call the same reusable experiment
runner as the CLI, avoiding divergent notebook-only implementations. Outputs are
not committed as if a demonstration run were a trained-model result.

1. Titanic dense classifier: confusion matrix, ROC, calibration, subgroup report.
2. MNIST CNN: digit classification and mistakes.
3. RNN forecasting: next value of independent sine trajectories.
4. LSTM forecasting: the same task with hidden/cell state.
5. Causal character transformer: perplexity and sampled text.
6. MNIST autoencoder: paired reconstructions and latent arrays.
7. Fashion-MNIST GAN: fixed-noise clothing generation over epochs.

See `docs/examples.md` for longer runs, dataset sources, licensing, and limitations.
A cats/dogs data notebook and full architecture-rendering UI are future additions.
