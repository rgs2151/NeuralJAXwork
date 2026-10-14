# Examples and datasets

The published notebooks use the complete local GPU recipes in
`examples/train_gallery.py`. Each [architecture tutorial](tutorials.rst) includes its
training configuration, evaluation, and saved figures.
The commands below describe how to start or customize a run.

All commands run from the repository root with `pip install -e '.[examples]'`.
Use `--smoke --epochs 1` for a small offline integration check. Normal runs use
the datasets described below and default to small subsets for development.

## Dense network: Titanic

```bash
python -m examples.run titanic --download --epochs 30 --seed 0
```

The source is [OpenML Titanic 40945](https://www.openml.org/d/40945). The example
creates stratified train/validation/test splits (70/15/15). It uses age, siblings/
spouses, parents/children, fare, passenger class, sex, and embarkation port.
It deliberately excludes rescue boat and body fields, which leak outcomes.
Median imputation, scaling, categorical filling, and one-hot vocabularies are
fitted on training rows only. Split IDs, feature order, and transformation
statistics are written to `data.json`; retain this alongside the weights.

The MLP returns two logits and trains with cross-entropy. The best validation-loss
checkpoint is evaluated once on the held-out test split. `classification.png`
shows a confusion matrix, ROC curve, and calibration plot; `report.json` includes
accuracy, a majority-class baseline, and counts/accuracy by sex. These small
subgroup statistics are descriptive, not causal or general fairness conclusions.
`predictions.npz` supports additional plots without retraining.

This is a labeled holdout experiment, not a Kaggle submission or scoring of its
unlabeled competition test file. Passenger relationships can create dependence;
a stricter generalization study would need family/group-aware splitting.

## CNN: handwritten MNIST digits

```bash
python -m examples.run cnn --download --limit 10000 --epochs 10
```

Two convolutions, pooling, dropout, and a linear head map `[B,1,28,28]` images to
ten digit logits. A seeded validation subset comes from the official training
set; the official test set is evaluated only after selection. `--limit` also caps
evaluation samples for fast exploratory runs. Split IDs are saved.

Artifacts include learning curves, confusion matrix, and misclassified images
with true/predicted labels. The loader uses the
[upstream Keras MNIST mirror and published SHA-256](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py),
without depending on TensorFlow or Keras. Dataset attribution/license is described
by that upstream source (MNIST derives from NIST and is credited to its authors).
The code license does not replace dataset terms.

## RNN and LSTM: next-value forecasting

```bash
python -m examples.run rnn --epochs 20 --limit 1024
python -m examples.run lstm --epochs 20 --limit 1024
```

Both models receive twelve sine-wave observations and predict the next value.
Each trajectory has an independently sampled phase/frequency. Different seeds
produce train, validation, and test trajectories, avoiding overlap from slicing
one time series. The models share a linear prediction head and MSE objective.

`forecast.png` plots predicted versus true values; the diagonal is perfect
prediction. The report includes persistence baseline MSE (predict the last seen
value). Compare architectures at equal seeds, data, and training budgets. This
controlled task illustrates recurrent mechanics; it does not establish that LSTM
is superior on long-range dependencies. Layer tests cover padding, carry state, and chunk equivalence.

## Transformer: character next-token prediction

```bash
python -m examples.run transformer --epochs 10
python -m examples.run transformer --text-file /path/to/corpus.txt --epochs 20
```

Token and position embeddings feed pre-normalized causal-attention blocks and a
vocabulary projection. Targets are shifted one character. The fixed ASCII
vocabulary prevents fitting a vocabulary on held-out text. Contiguous text ranges
form train/validation/test; windows never cross a split boundary.

The default corpus is a short offline example with repeated
phrases across splits. Its validation score is not evidence of generalization.
For a real corpus, inspect source/license and split contamination. Perplexity and
sampled text are saved; generation uses an independent seed and a sliding context.
The implementation uses character-level tokenization and does not cache attention
keys and values. Special ASCII controls are part of the toy vocabulary, so early
samples can contain unprintable characters.

## Autoencoders: MNIST reconstruction

```bash
python -m examples.run autoencoder --download --limit 10000 --epochs 10
```

The dense autoencoder compresses flattened digits to sixteen latent values and
reconstructs `[0,1]` images using a sigmoid decoder and MSE. Compare adjacent
original/reconstructed image pairs, inspect held-out reconstruction loss, and
reuse saved latent codes for projections. The published run uses width 256 and 16 latent values. Reconstruction loss
measures pixel error, not classification accuracy. `models.ConvAutoencoder` provides a spatial alternative
using strided convolutions and transposed convolutions; the default runner uses
the dense model for easier latent inspection.

## GAN: Fashion-MNIST generation

```bash
python -m examples.run gan --download --limit 10000 --epochs 30
```

The [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) by
Zalando contains clothing images; the upstream repository supplies its MIT license.
The loader uses the [Keras-hosted IDX files](https://github.com/keras-team/keras/blob/master/keras/src/datasets/fashion_mnist.py)
and validates IDX headers and sizes. It does not currently pin Fashion-MNIST file
hashes; downloaded file integrity is structurally checked, not cryptographically
version-locked.

Training images are scaled to `[-1,1]`. A convolutional generator maps 64-dimensional noise
to an image; a convolutional discriminator emits a raw logit. Each batch takes one
non-saturating discriminator update and one generator update. The inactive model
is in evaluation mode, and fake images are detached only in the discriminator
phase. Both model/optimizer states and runtime RNGs are saved together.

Fixed noise produces a sample grid every epoch. Inspect clothing structure,
diversity, and repeated outputs alongside both loss curves. Monitor sample diversity for mode collapse. The example does not compute FID.
The official test set is not used to tune GAN training. Better architecture and
quantitative generative evaluation are later experiments.

## Cats and dogs

`models.PetCNN` is implemented and smoke-tested for RGB inputs. A downloadable
cats/dogs dataset notebook is not included in 0.1.0: acquisition, licensing,
duplicate-aware splits, and a longer image-loading pipeline need a separate
example. You can train it with the standard classifier step on NCHW batches.

## Run artifacts and cost

Each new run directory contains JSON config/data/architecture files, JSONL metrics,
`last.npz` for resuming an explicit loop, `best.npz` for supervised model selection,
PNG diagnostics, and a JSON report. Prediction arrays are saved for tabular,
image classification, forecasting, and reconstruction examples. Notebooks display these saved artifacts.

The full examples are excluded from default pytest. Dataset downloads happen only
when `--download` is supplied and cached files are missing. Local dataset and run
folders are gitignored. Budget with `--limit`, `--epochs`, and `--batch-size`;
compilation startup can dominate very small runs. Use a fresh `--output` path.
