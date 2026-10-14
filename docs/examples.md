# Examples and datasets

The published notebooks use the complete local GPU recipes in
`examples/train_gallery.py`. Each [architecture tutorial](learning.rst) includes its
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

Artifacts include learning curves, a confusion matrix, a correct-prediction gallery,
and a separate error gallery with true/predicted labels. `gallery.npz` and
`gallery.json` preserve the selected images, probabilities, indices, and selection
rules for visualization. The loader uses the
[upstream Keras MNIST mirror and published SHA-256](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py),
without depending on TensorFlow or Keras. Dataset attribution/license is described
by that upstream source (MNIST derives from NIST and is credited to its authors).
The code license does not replace dataset terms.

## RNN: seasonal temperature forecasting

```bash
python -m examples.run rnn --download --epochs 80 --limit 100000 --batch-size 64
```

The RNN predicts 24 hourly New York temperatures from 168 known hours. Data come
from [Open-Meteo's historical ERA5 API](https://open-meteo.com/en/docs/historical-weather-api),
covering 2017–2025. Temperature normalization is fitted before 2023; calendar
features encode hour and day of year. The model learns a correction to yesterday's
hourly profile, with a zero-initialized output projection. Baselines repeat that
profile or hold the last temperature constant.

Training targets precede 2023, validation targets fall in 2023, and test targets
fall in 2024–2025. Targets never cross split boundaries; known history may cross
into the preceding split. The four seasonal forecast panels use fixed calendar
dates, and the lead-time error plot evaluates all held-out daily origins.
ERA5 is reanalysis, so this is a historical forecasting experiment rather than an
operational weather-forecast replay. Data attribution: ERA5 / Copernicus Climate
Change Service, distributed by Open-Meteo under CC BY 4.0.

## LSTM: market volatility forecasting

```bash
python -m examples.run lstm --download --epochs 80 --limit 100000 --batch-size 64 --model-width 16
```

The LSTM uses 60 known trading sessions to predict the next five sessions'
annualized root-mean-square log return: `sqrt(252 * mean(next_5_returns**2))`.
[S&P 500 closing levels](https://fred.stlouisfed.org/series/SP500) come from
S&P Dow Jones Indices LLC via FRED. The cached period is October 2016–December
2025; FRED's available history rolls forward. Source URLs, file hashes, and actual
split dates are recorded in `data.json`. Source data terms remain separate from
the package license.

Inputs contain past returns, absolute returns, and trailing 5- and 20-session
volatility. The model learns a correction to the trailing-20-session estimate
in standardized log-volatility space. Normalization uses training targets only.
Chronological target splits match the weather experiment. Targets crossing split
boundaries are excluded; adjacent five-session outcomes overlap within a split.
Reported errors are descriptive, not independent statistical trials.

The market plot pairs known index history with predicted and subsequent realized
volatility. Baselines use trailing 5- and 20-session estimates. This is a volatility
forecast, not a stock-price or trading-return model. Large unanticipated market
shocks remain visible in the evaluation.

Both recurrent examples use validation-based checkpoint selection, gradient
clipping, and early stopping. Full float32 matrix products stabilize GPU replay.
`--smoke` retains a small synthetic sequence check without network access.

## Forecast records for visualization

`predictions.npz` stores all held-out histories, forecasts, targets, baselines,
and timestamps in physical units. `visualization.json` describes the fields.
An animation can move through consecutive origins without inference or retraining:
show history through `issued_at`, draw the forecast over `target_times`, and reveal
the complete outcome at `outcome_available_at`. Weather targets are in °C;
market volatility is stored as a fraction and multiplied by 100 for percent labels.
Market `future_prices` are outcomes, not known inputs. These artifacts and all
checkpoints stay in the gitignored run directory.

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
