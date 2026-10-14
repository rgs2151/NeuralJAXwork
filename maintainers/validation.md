# Validation

Validation has two separate responsibilities: fast package correctness checks and
local training/evaluation of examples. Documentation rendering does neither; it
validates and displays the saved output record.

## Package checks

The CPU suite exercises API/state behavior on tiny arrays, including eager/JIT
agreement, parameter sharing, dropout, BatchNorm, stable losses, recurrent carries,
causal attention, convolution arithmetic, dense/convolutional GAN update boundaries,
checkpoint continuation, loader restoration, trainer update counts, chronological forecast boundaries, past-only rolling
features, and training-only normalization. It needs
no dataset downloads. The package and wheel are built in CI on Python 3.10/3.12.

## Local GPU runs

The seven model examples were trained on an NVIDIA RTX 5090. JAX 0.5.3, Flax NNX 0.10.4, Optax 0.2.4, and the JAX CUDA 12 plugin
were used. The [results page](results.md) reports held-out metrics, training sizes,
selection rules, and caveats. The notebooks include fixed-noise Fashion-MNIST outputs and MNIST reconstruction
pairs.

Twelve notebooks—five API tutorials and seven model examples—were executed locally
and saved with outputs. The notebook metadata records a digest of code cells.
Editing code without refreshing outputs causes documentation validation to fail,
preventing stale figures/results from silently accompanying changed code.

## Documentation checks

Sphinx uses `nb_execution_mode = 'off'`. A warning-free build and
`scripts/check_docs.py` verify saved outputs, absence of notebook errors, matching
source digests, rendered figures where applicable, navigation, source downloads,
and the project logo. No notebook execution occurs in CI. The Pages deployment
runs only after this static build passes.

GPU results are single-workstation, single-seed demonstrations, not distributed
training coverage, comprehensive hardware compatibility, or competitive benchmarks.
