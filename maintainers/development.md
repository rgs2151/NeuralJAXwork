# Development

## Local checks

```bash
python -m pip install -e '.[dev,examples,docs]'
JAX_PLATFORMS=cpu python -m pytest
python -m build
python -m sphinx -W --keep-going -b html . _build/html
python scripts/check_docs.py _build/html
```

Default tests do not download data or train full models. They cover eager/JIT
agreement and learning, parameter sharing, dropout/BatchNorm state, stable losses,
convolution arithmetic/transpose adjoint behavior, RNN/LSTM carries and padding,
causal/masked attention gradients, each model family's update, isolated GAN
updates, checkpoint next-update equality, loader resume, inspection noninterference,
and trainer update counts. They are intentionally small, with no performance
threshold tied to one machine. CI jobs have a ten-minute timeout.

Optional integration checks:

```bash
python -m examples.run cnn --smoke --epochs 1
python -m examples.run gan --smoke --epochs 1
```

Notebook execution and dataset training happen locally, outside CI.
The documentation workflow renders saved notebook outputs and deploys successful
main-branch builds to Pages; see [publishing](publishing.md).
Keep full-dataset training outside the default test suite. Add focused numerical
and state tests when introducing a new primitive.

## Versioning and packaging

The package is an alpha, with a tested NNX dependency range. Breaking API changes
must update guides and examples together. Use `src/` packaging so the wheel is
independent of repository-root imports. The wheel includes the core package;
examples, notebooks, and design discussions are repository resources. The old
uppercase package is archived under `legacy/` and excluded from installation.

Building artifacts locally does not publish to PyPI. Release publication requires
a chosen registry/account, release credentials, and an explicit release action.
GPU/TPU users must configure a compatible JAX installation for their hardware;
this repository's CI exercises CPU. The published example runs were also tested
locally on one RTX 5090. Dependency bounds preserve compatibility with the
supported NNX API.

## Current limits

Unsupported features include distributed training, mixed precision policy, parameter groups,
parameter-freezing helpers, schedulers, model reconstruction from JSON, operation
graph rendering, KV caches, bidirectional recurrence, pretrained models, and ONNX export.
Adaptive average pooling currently supports `(1,1)` only. Embedding/class-index
ranges and recurrent length values are caller data preconditions. Convolution
arguments cover the documented subset, not all PyTorch keyword combinations.

Direct NNX transformation semantics apply. The original design's universal
transactional rollback and strict raw-array/container-registration guarantees are
not implemented. Optimizers validate compatible tree structure through NNX/Optax,
not provenance of an unrelated same-shaped gradient tree. See the explicit
[design status](../design/IMPLEMENTATION.md) for the differences between the proposal and the release.

## Contribution checklist

Keep new modules composable, explicit about shapes/state/randomness, and usable
without plotting dependencies. Add a tiny smoke/numerical test and update API
reference. Report dataset metrics with their run configuration and saved outputs.
Keep changes to legacy code separate from the installed package.
