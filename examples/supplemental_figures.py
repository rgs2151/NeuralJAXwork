"""Generate additional diagnostics from saved predictions; no model training."""

import json
from pathlib import Path

import numpy as np


def create(
    root="runs/gallery",
    autoencoder="runs/gallery-refined/autoencoder",
    forecasts="runs/forecast-gallery-v2",
):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from . import datasets
    from .forecasting import draw_forecasts

    root = Path(root)

    for task in ("rnn", "lstm"):
        path = Path(forecasts) / task
        with np.load(path / "predictions.npz", allow_pickle=False) as archive:
            records = {name: archive[name] for name in archive.files}
        draw_forecasts(task, records, path)
    path = Path(autoencoder)
    saved = np.load(path / "predictions.npz")
    _, (_, labels) = datasets.images("mnist")
    points = PCA(n_components=2, random_state=0).fit_transform(saved["latent"][:2000])
    fig, ax = plt.subplots(figsize=(7, 5))
    artist = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=labels[:2000],
        cmap="tab10",
        s=7,
        alpha=0.65,
        vmin=-0.5,
        vmax=9.5,
    )
    fig.colorbar(artist, ax=ax, ticks=range(10), label="Test digit label (not used for training)")
    ax.set(
        xlabel="First principal component",
        ylabel="Second principal component",
        title="Held-out latent codes: PCA",
    )
    fig.tight_layout()
    fig.savefig(path / "latent.png", dpi=150)
    plt.close(fig)
    path = root / "titanic"
    report = json.loads((path / "report.json").read_text())
    saved = np.load(path / "predictions.npz")
    report["test_auc"] = float(roc_auc_score(saved["targets"], saved["probabilities"][:, 1]))
    report["test_brier"] = float(brier_score_loss(saved["targets"], saved["probabilities"][:, 1]))
    (path / "report.json").write_text(json.dumps(report, indent=2))
    groups = report["subgroups"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(list(groups), [v["accuracy"] for v in groups.values()], color="#265d87")
    for i, v in enumerate(groups.values()):
        ax.text(i, v["accuracy"] + 0.02, f"n={v['count']}", ha="center")
    ax.set(ylim=(0, 1), ylabel="Held-out accuracy", title="Titanic: subgroup results")
    fig.tight_layout()
    fig.savefig(path / "subgroups.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    create()
