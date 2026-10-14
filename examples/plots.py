"""Plot saved predictions and training history."""

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

    binary = probabilities.shape[1] == 2
    fig, axes = plt.subplots(1, 3 if binary else 1, figsize=(12, 3) if binary else (6, 5))
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


def digit_gallery(images, targets, probabilities, output):
    """Export deterministic success and error examples from evaluated MNIST images."""
    import json
    from pathlib import Path

    output = Path(output)
    predicted = probabilities.argmax(-1)
    correct = np.concatenate(
        [np.flatnonzero((targets == label) & (predicted == label))[:2] for label in range(10)]
    )
    wrong = np.flatnonzero(predicted != targets)[:16]
    if len(correct):
        labels = [f"{targets[i]} → {predicted[i]}" for i in correct]
        image_grid(
            images[correct],
            output / "successes.png",
            "Correct predictions · two per digit",
            labels,
            columns=5,
        )
    if len(wrong):
        labels = [f"{targets[i]} → {predicted[i]}" for i in wrong]
        image_grid(
            images[wrong],
            output / "mistakes.png",
            "Failure analysis · true → predicted",
            labels,
            columns=4,
        )
    ids = np.concatenate([correct, wrong])
    np.savez_compressed(
        output / "gallery.npz",
        images=images[ids],
        targets=targets[ids],
        probabilities=probabilities[ids],
        test_indices=ids,
        correct=predicted[ids] == targets[ids],
    )
    (output / "gallery.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "file": "gallery.npz",
                "split": "official MNIST test",
                "success_selection": "first two correct examples per digit in dataset order",
                "failure_selection": "first sixteen errors in dataset order",
                "success_count": len(correct),
                "failure_count": len(wrong),
                "population_accuracy": float(np.mean(predicted == targets)),
                "note": "Illustrative gallery; selected class/error counts do not represent the population distribution.",
            },
            indent=2,
        )
    )
