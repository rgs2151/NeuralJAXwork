"""Atomic, non-pickle NPZ checkpoints for preconstructed matching object graphs."""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from flax import nnx

from . import inspection
from .nn.module import Module


def _signature(objects):
    result = {}
    for name, value in objects.items():
        if isinstance(value, Module):
            result[name] = inspection.describe(value)
        elif hasattr(value, "config"):
            result[name] = value.config
    return result


def save(path, *, metadata=None, progress=None, data_state=None, **objects):
    """Save named models/optimizers/RNGs together, preserving joint graph paths.

    Restore requires the same named participants, architecture and optimizer config.
    Training modes are recorded separately. There is no executable deserialization.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = nnx.state(objects)
    flat = list(nnx.to_flat_state(state))
    modes = {
        name: {p: m.training for p, m in obj.named_modules()}
        for name, obj in objects.items()
        if isinstance(obj, Module)
    }
    manifest = {
        "schema": 1,
        "paths": [list(p) for p, _ in flat],
        "types": [f"{v.type.__module__}.{v.type.__qualname__}" for _, v in flat],
        "signature": _signature(objects),
        "modes": modes,
        "metadata": metadata or {},
        "progress": progress or {},
        "data_state": data_state,
    }
    arrays = {}
    for i, (_, variable) in enumerate(flat):
        value = variable.value
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        arrays[f"a{i}"] = np.asarray(value)
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    try:
        with os.fdopen(fd, "wb") as file:
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _canonical_types(value):
    """Read pre-rename checkpoint type names without changing user metadata."""
    if isinstance(value, dict):
        return {key: _canonical_types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_types(item) for item in value]
    if isinstance(value, str) and value.startswith("neuraljaxwork."):
        return "neuraljaxlib." + value.removeprefix("neuraljaxwork.")
    return value


def restore(path, **objects):
    """Validate the entire snapshot before updating any live participant."""
    state = nnx.state(objects)
    flat = list(nnx.to_flat_state(state))
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        manifest["signature"] = _canonical_types(manifest["signature"])
        manifest["types"] = _canonical_types(manifest["types"])
        if manifest["schema"] != 1 or manifest["paths"] != [list(p) for p, _ in flat]:
            raise ValueError("Checkpoint schema or state paths do not match")
        if manifest["signature"] != _signature(objects):
            raise ValueError("Checkpoint architecture/optimizer configuration mismatch")
        pending = []
        for i, (p, variable) in enumerate(flat):
            array = archive[f"a{i}"]
            if jax.dtypes.issubdtype(variable.value.dtype, jax.dtypes.prng_key):
                array = jax.random.wrap_key_data(
                    jax.device_put(array), impl=jax.random.key_impl(variable.value)
                )
            if (
                array.shape != variable.value.shape
                or array.dtype != variable.value.dtype
                or manifest["types"][i]
                != f"{variable.type.__module__}.{variable.type.__qualname__}"
            ):
                raise ValueError(f"Checkpoint type/shape/dtype mismatch at {p}")
            pending.append((p, variable.replace(jax.device_put(array))))
    nnx.update(objects, nnx.from_flat_state(pending))
    for name, modes in manifest["modes"].items():
        for p, module in objects[name].named_modules():
            module.training = modes[p]
    return SimpleNamespace(
        metadata=manifest["metadata"],
        progress=manifest["progress"],
        data_state=manifest["data_state"],
    )
