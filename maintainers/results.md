# Training results

These results were produced locally on **GPU 0, NVIDIA GeForce RTX 5090**, using
Python 3.10, JAX 0.5.3, Flax NNX 0.10.4, and the CUDA 12 plugin.
All runs use seed 0. Checkpoints and downloaded data are gitignored; saved notebook
outputs and the [run records](results.json) preserve the
configurations, metrics, histories, hardware, and checkpoint hashes.

| Example | Epochs | Training examples | Test examples | Held-out result |
| --- | ---: | ---: | ---: | --- |
| titanic | 100 | 916 | 197 | Accuracy **81.22%** |
| cnn | 10 | 55,000 | 10,000 | Accuracy **99.06%** |
| rnn | 80 (selected 75) | 2,184 | 731 daily forecasts | Temperature MAE **1.81°C** |
| lstm | 23 (selected 8) | 1,479 | 498 forecast origins | Volatility MAE **4.56 percentage points** |
| transformer | 20 | 27,881 | 2,048 | Cross-entropy **1.7873**; perplexity **5.973** |
| autoencoder | 25 | 55,000 | 10,000 | MSE **0.010804** |
| gan | 30 | 55,000 | 1,000 | Fixed-noise galleries; no quantitative quality claim |

For the GAN, the listed test subset is loaded by the shared image loader but is
not used for optimization or quantitative evaluation. Evaluation uses fixed-noise sample galleries; FID, precision/recall, and coverage
metrics were not computed.

The four-stage CNN was retrained in `runs/gallery-four-layer/cnn`. Its lowest-validation-loss checkpoint (epoch 5) replaces the two-stage CNN result.

## Evaluation protocol

The CNN and autoencoder use 55,000 training images, a seeded 5,000-image validation
split, and all 10,000 official MNIST test images. Titanic uses a stratified
916/196/197 split and preprocessing fitted only on training rows. RNN and LSTM
use historical weather and market data, with training targets before 2023,
validation targets in 2023, and test targets in 2024–2025. Targets crossing a
split boundary are excluded. Prior observations can be used as input history.
The RNN predicts 24 hourly temperatures from seven days of history; repeating
yesterday gives 2.57°C MAE. The LSTM predicts five-session annualized RMS log
return from 60 prior sessions; trailing-20-session volatility gives 5.70 percentage
points MAE. Overlapping market targets are not independent statistical trials.
Both forecast models use full float32 matrix products, and the selected checkpoints
reproduce saved predictions in a fresh process on the same GPU. The initial
reduced-precision runs were replaced to resolve GPU replay differences; final
records use `runs/forecast-gallery-v2`.

The transformer uses contiguous 80/10/10 character ranges of Tiny Shakespeare,
a 128-character context, width 128, three blocks, and a fixed ASCII vocabulary.
The evaluated validation/test sets each contain the first 2,048 windows from
their respective text ranges. Its final test loss is not its validation loss.
The selected checkpoint predates the final epoch because validation loss rose.

Supervised runs restore the minimum-validation-loss checkpoint before test
inference. These are exploratory teaching examples, not a preregistered benchmark
or a multi-seed comparison. The autoencoder bottleneck was increased from eight
to sixteen after an initial visual review; that exploratory choice should not be
interpreted as independent test-set model selection. Generated text retains its
errors, and fixed-noise GAN galleries show all selected noise vectors.

## Reproduction and scope

The recipes are in `examples/train_gallery.py`. Use
`python -m examples.train_gallery --gpu 0`; see the publishing guide for GPU setup
and corpus acquisition. Notebook execution happens locally after training. CI
renders the saved notebook outputs without running code.

Timing values in the run records include the training loop, validation, and
checkpoint work, but exclude initial data preparation and some post-training
plotting. They are measurements on this workstation, not speed claims against
other frameworks. Dataset preprocessing, batching, and compile behavior differ
between tasks.
