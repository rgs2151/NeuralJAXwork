"""Local structured run records; plotting libraries are optional consumers."""

import json
import time
from pathlib import Path

import jax


class Run:
    def __init__(self, path, config):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / "manifest.json"
        if manifest.exists():
            raise FileExistsError(f"Run exists: {self.path}; choose a fresh run directory")
        manifest.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "config": config,
                    "jax": jax.__version__,
                    "devices": [str(d) for d in jax.devices()],
                },
                indent=2,
            )
        )

    def log_metrics(self, *, step=None, epoch=None, split, **metrics):
        record = {
            "step": step,
            "epoch": epoch,
            "split": split,
            "time": time.time(),
            **{k: float(v) for k, v in metrics.items()},
        }
        with (self.path / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")

    def log_artifact(self, name, path):
        with (self.path / "artifacts.jsonl").open("a") as file:
            file.write(json.dumps({"name": name, "path": str(path)}) + "\n")
