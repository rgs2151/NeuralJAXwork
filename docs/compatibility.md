# Compatibility and limitations

NeuralJAXwork 0.1.0 is an alpha release. Install it from a repository checkout;
it is not yet published to PyPI. The installed package uses the lowercase
`neuraljaxwork` import. The archived 2023 API and its checkpoints are incompatible.

## Runtime

The package supports Python 3.10–3.13, JAX 0.5.3, Flax NNX 0.10.4, and
Optax 0.2.4 within the dependency bounds in `pyproject.toml`. CPU correctness
checks run in CI on Python 3.10 and 3.12. GPU use requires a compatible JAX
runtime; the model tutorials were run on an NVIDIA RTX 5090 with CUDA 12.

## Supported operations

Modules compose through ordinary Python and JAX operations. Layer arguments
cover the documented interfaces rather than every PyTorch keyword combination.
Adaptive average pooling supports output size `(1, 1)`. Recurrent modules are
unidirectional; transformer attention does not provide a KV cache.

Embedding indices, class indices, and recurrent sequence lengths must fall
within the ranges expected by their corresponding layers or losses.

## Training and state

Training is single-device. There is no distributed sampler, mixed-precision
policy, optimizer parameter groups, or built-in parameter-freezing helper.
NNX transformation semantics apply; failed updates do not provide automatic
transactional rollback. Optimizers check tree compatibility, but cannot establish
that a same-shaped gradient tree came from the intended model.

## Serialization and inspection

Checkpoint loading requires a matching architecture and optimizer configuration.
Architecture JSON describes module ownership, shapes, and aliases; it is not an
executable model format or a full operation graph. There is no ONNX export or
pretrained model registry.
