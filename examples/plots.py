"""Optional plots; consumers of saved predictions and history, never training hooks."""

import numpy as np


def learning_curve(history, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for key in ("train_loss", "val_loss", "g_loss", "d_loss"):
        if key in history[0]:
            ax.plot(range(1, len(history) + 1), [r[key] for r in history], label=key)
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def image_grid(images, path, title, labels=None, columns=8):
    import matplotlib.pyplot as plt

    images = np.asarray(images)[:32]
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows * 1.3), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i].squeeze(), cmap="gray", vmin=0, vmax=1)
            if labels is not None:
                ax.set_title(str(labels[i]), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def classification(targets, probabilities, path, names=None):
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibrationDisplay
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    fig, axes = plt.subplots(1, 3 if probabilities.shape[1] == 2 else 1, figsize=(12, 3))
    axes = np.atleast_1d(axes)
    ConfusionMatrixDisplay.from_predictions(
        targets, probabilities.argmax(-1), display_labels=names, ax=axes[0], colorbar=False
    )
    if len(axes) == 3:
        RocCurveDisplay.from_predictions(targets, probabilities[:, 1], ax=axes[1])
        CalibrationDisplay.from_predictions(targets, probabilities[:, 1], n_bins=5, ax=axes[2])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
