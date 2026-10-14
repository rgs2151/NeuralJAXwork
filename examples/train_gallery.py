"""Train the documentation gallery locally on one GPU; never invoked by CI."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RECIPES = {
    "gan": ["--epochs", "30", "--limit", "55000", "--batch-size", "128"],
    "cnn": ["--epochs", "10", "--limit", "55000", "--eval-limit", "10000", "--batch-size", "128"],
    "autoencoder": [
        "--epochs",
        "25",
        "--limit",
        "55000",
        "--eval-limit",
        "10000",
        "--batch-size",
        "256",
    ],
    "titanic": ["--epochs", "100", "--batch-size", "64"],
    "rnn": ["--epochs", "80", "--limit", "100000", "--batch-size", "64"],
    "lstm": ["--epochs", "80", "--limit", "100000", "--batch-size", "64", "--model-width", "16"],
    "transformer": [
        "--epochs",
        "20",
        "--limit",
        "30000",
        "--eval-limit",
        "2048",
        "--batch-size",
        "64",
        "--context-length",
        "128",
        "--model-width",
        "128",
        "--model-depth",
        "3",
        "--text-file",
        "data/tinyshakespeare.txt",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--root", default="runs/gallery")
    parser.add_argument("--tasks", nargs="+", choices=RECIPES, default=list(RECIPES))
    args = parser.parse_args()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpu,
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
    )
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if (root / task / "report.json").exists():
            report = json.loads((root / task / "report.json").read_text())
            if task in ("rnn", "lstm") and "evaluation" not in report:
                raise ValueError(
                    f"{root / task} contains an older sequence task; choose a fresh --root"
                )
            print(f"{task}: completed run already exists; preserving it", flush=True)
            continue
        with (root / f"{task}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "examples.run",
                    task,
                    "--download",
                    "--output",
                    str(root / task),
                    *RECIPES[task],
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"{task}: completed; artifacts in {root / task}", flush=True)


if __name__ == "__main__":
    main()
