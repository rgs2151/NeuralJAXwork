# Documentation workflow

Sphinx, MyST-NB, and the PyData Sphinx Theme render the documentation. Notebook
execution is disabled with `nb_execution_mode = 'off'`. This setting applies to
local documentation builds and GitHub Actions alike. Saved cell outputs are the
published record; rendering never starts training.

## Local training and notebook execution

Install the notebook dependencies and a compatible JAX GPU runtime. Select one
GPU explicitly. The documented local environment uses JAX 0.5.3 with its CUDA 12
plugin and CUDA libraries, on an NVIDIA RTX 5090:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[notebooks,docs,dev]'
python -m pip install 'jax[cuda12]==0.5.3'
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python -m examples.train_gallery
```

The gallery runner's recipes are in `examples/train_gallery.py`. It preserves
completed runs. Download Tiny Shakespeare to `data/tinyshakespeare.txt` as
described in the transformer tutorial before running that recipe. Training
checkpoints and full artifacts live under gitignored `runs/`; datasets live under
`data/`.

Run `python scripts/execute_notebooks.py --gpu 0` to execute and save notebook
outputs locally before publishing. The command records a source digest so CI can
reject stale outputs after code-cell edits without executing those cells. They display
measured results from the run directory and include the commands needed to
reproduce them. Retain the run configuration, seed, evaluation split, hardware,
and limitations alongside the figures. Figures and metrics must come from the recorded run.

## Render without execution

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html . _build/html
python scripts/check_docs.py _build/html
python -m http.server --directory _build/html 8000
```

The renderer does not need a GPU, notebook kernel, dataset, or checkpoint. It reads
saved notebook outputs and public API docstrings. Validation checks that the
notebooks contain successful saved outputs, and the HTML includes navigation,
figures where expected, source downloads, and the project logo.

## CI/CD

`check_demos.yml` runs small API/package tests on CPU. `doc_maker.yml` builds and
validates the saved documentation on pull requests and pushes. It does not register
a kernel or invoke notebook execution or training. Pull requests upload a preview
artifact; successful `main` builds publish to GitHub Pages using GitHub Actions.

The public site is [NeuralJAXwork](https://rgs2151.github.io/NeuralJAXwork/).
Model training and documentation rendering are intentionally separate workflows.
