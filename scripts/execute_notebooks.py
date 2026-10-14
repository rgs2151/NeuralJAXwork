"""Execute and save notebooks locally on one GPU. Never called by CI."""

import argparse
import hashlib
import os
import time
from pathlib import Path


def source_digest(notebook):
    return hashlib.sha256(
        "\n".join(c.source for c in notebook.cells if c.cell_type == "code").encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        MPLBACKEND="Agg",
    )
    import nbformat
    from nbclient import NotebookClient

    root = Path(__file__).resolve().parents[1]
    paths = [Path(p) for p in args.notebooks] or sorted((root / "notebooks").glob("*.ipynb"))
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        ).execute()
        notebook.metadata.setdefault("neuraljaxlib", {}).update(
            source_sha256=source_digest(notebook),
            executed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            gpu_selection=args.gpu,
        )
        nbformat.write(notebook, path)
        print(f"Saved {path.name}", flush=True)


if __name__ == "__main__":
    main()
