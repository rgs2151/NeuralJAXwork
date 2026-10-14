# Building and publishing the tutorial book

The documentation is a Sphinx book using MyST-NB and the Sphinx Book Theme.
Notebook sources in `notebooks/` are included directly in the table of contents.
They are executed during the build and rendered with their reports and figures;
there is no separate hand-copied tutorial version.

## Build locally

```bash
python -m pip install -e '.[dev,docs]'
python -m ipykernel install --sys-prefix --name python3
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m sphinx -W --keep-going -b html . _build/html
python scripts/check_docs.py _build/html
python -m http.server --directory _build/html 8000
```

Open `http://localhost:8000` to inspect the whole book. The kernel must belong to
the environment containing the installed package and example dependencies.

Every tutorial defaults to synthetic smoke mode. Builds require no dataset
downloads or full training runs. MyST-NB executes notebooks with a 90-second
per-cell timeout, and notebook errors fail the build. Warnings are treated as
errors. The default pytest suite remains separate and small.

## CI/CD

`.github/workflows/doc_maker.yml` runs on pull requests, pushes to `main`, and
manual dispatch. Its build job installs the package, runs the smoke tests,
executes all seven notebooks, builds Sphinx, and checks that chapters, outputs,
figures, and notebook downloads are present. The rendered book is uploaded as
the `tutorial-book` Actions artifact for review.

Only a successful build on `main` deploys. GitHub's `configure-pages`,
`upload-pages-artifact`, and `deploy-pages` actions publish the book to
[GitHub Pages](https://rgs2151.github.io/NeuralJAXwork/). Pull requests build the
same book but never deploy it. The deployment job uses the `github-pages`
environment and the required `pages: write` / `id-token: write` permissions.
Repository Pages settings must use **GitHub Actions** as the publishing source.

A green local build is not proof of publication: check the workflow's deploy job
and the live site. Full-dataset tutorial results are intentionally not part of
this deployment; published smoke plots are labeled as mechanics checks.
