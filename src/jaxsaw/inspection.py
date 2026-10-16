"""JSON-safe architecture hierarchy and non-mutating output shape inspection."""

import json
from pathlib import Path

import jax
from flax import nnx

from .nn.module import Module


def describe(model):
    """Describe ownership/configuration and aliases, not a complete execution graph."""
    seen = {}

    def visit(value, path):
        if isinstance(value, (nnx.Variable, Module, list, dict, tuple)):
            if id(value) in seen:
                return {"alias": seen[id(value)]}
            seen[id(value)] = path
        if isinstance(value, nnx.Variable):
            return {
                "kind": type(value).__name__,
                "shape": list(value.value.shape),
                "dtype": str(value.value.dtype),
            }
        if isinstance(value, Module):
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {
                    name: visit(child, f"{path}.{name}")
                    for name, child in vars(value).items()
                    if not name.startswith("_") and name != "training"
                },
            }
        if isinstance(value, (tuple, list)):
            return [visit(child, f"{path}.{i}") for i, child in enumerate(value)]
        if isinstance(value, dict):
            return {str(k): visit(v, f"{path}.{k}") for k, v in value.items()}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return {"opaque_type": type(value).__qualname__}

    return {
        "schema_version": 1,
        "scope": "module hierarchy; not an execution graph",
        "root": visit(model, "model"),
    }


def summary(model):
    lines = ["Module | Type | Own parameters"]
    for path, module in model.named_modules():
        count = sum(v.value.size for v in vars(module).values() if isinstance(v, nnx.Param))
        lines.append(f"{path or '<root>'} | {type(module).__name__} | {count:,}")
    total = sum(p.value.size for _, p in model.named_parameters())
    return "\n".join(lines + [f"Total unique parameters: {total:,}"])


def write_json(model, path):
    Path(path).write_text(json.dumps(describe(model), indent=2) + "\n")


def output_spec(model, *inputs, **kwargs):
    """Run an evaluation snapshot; return only output shapes/dtypes."""
    graph, state = nnx.split(model)
    clone = nnx.merge(graph, state)
    clone.eval()
    outputs = clone(*inputs, **kwargs)
    return jax.tree.map(lambda x: {"shape": list(x.shape), "dtype": str(x.dtype)}, outputs)
