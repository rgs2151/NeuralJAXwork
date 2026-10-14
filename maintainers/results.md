# Training results

These results were produced locally on **GPU 0, NVIDIA GeForce RTX 5090**, using
Python 3.10, JAX 0.5.3, Flax NNX 0.10.4, and the CUDA 12 plugin. GPU 1 was not used.
All runs use seed 0. Checkpoints and downloaded data are gitignored; saved notebook
outputs and the [run records](results.json) preserve the
configurations, metrics, histories, hardware, and checkpoint hashes.

| Example | Epochs | Training examples | Test examples | Held-out result |
| --- | ---: | ---: | ---: | --- |
| titanic | 100 | 916 | 197 | Accuracy **81.22%** |
| cnn | 10 | 55,000 | 10,000 | Accuracy **98.94%** |
| rnn | 40 | 4,096 | 128 | MSE **0.000037** |
| lstm | 40 | 4,096 | 128 | MSE **0.000043** |
| transformer | 20 | 27,881 | 2,048 | Cross-entropy **1.7873**; perplexity **5.973** |
| autoencoder | 25 | 55,000 | 10,000 | MSE **0.010804** |
| gan | 30 | 55,000 | 1,000 | Fixed-noise galleries; no quantitative quality claim |

For the GAN, the listed test subset is loaded by the shared image loader but is
not used for optimization or quantitative evaluation. Review the fixed-noise sample
galleries, including defects. No FID, precision/recall, or coverage score is claimed.

## Evaluation protocol

The CNN and autoencoder use 55,000 training images, a seeded 5,000-image validation
split, and all 10,000 official MNIST test images. Titanic uses a stratified
916/196/197 split and preprocessing fitted only on training rows. RNN and LSTM
use 4,096 independent training trajectories and 128 independent trajectories for
each evaluation split. Their test persistence baseline MSE is 0.022309.

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
