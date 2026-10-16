"""Export separate MRI gallery assets without modifying approved originals."""

import base64
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
GALLERY = ROOT / "gallery"
RUN = ROOT / "runs/neuroscience"


def save_data(folder, name, data):
    folder.mkdir(exist_ok=True)
    (folder / "samples.js").write_text(
        f"window.{name} = " + json.dumps(data, separators=(",", ":")) + ";\n"
    )


def export():
    cnn = GALLERY / "cnn_brain_mri"
    gan = GALLERY / "gan_brain_mri"
    cache = np.load(RUN / "cnn/predictions.npz")
    labels = ["Meningioma", "Glioma", "Pituitary"]
    samples = []
    # One correct example per class, from distinct held-out participants.
    for k, label in enumerate(labels):
        indices = np.flatnonzero((cache["labels"] == k) & (cache["logits"].argmax(1) == k))
        i = int(indices[0])
        image = Image.fromarray(cache["images"][i, 0]).resize((64, 64), Image.Resampling.BILINEAR)
        samples.append(
            dict(
                prediction=label,
                testId=str(cache["ids"][i]),
                maps=[
                    dict(
                        size=64,
                        values=np.rint(np.asarray(image) * 255).astype(int).ravel().tolist(),
                    )
                ],
            )
        )
    save_data(
        cnn,
        "CNN_SAMPLES",
        dict(
            samples=samples,
            source="Jun Cheng, brain tumor dataset, https://figshare.com/articles/dataset/brain_tumor_dataset/1512427",
            license="CC BY 4.0",
            checkpointSha256=hashlib.sha256((RUN / "cnn/best.npz").read_bytes()).hexdigest(),
            note="Correct held-out examples selected for display. Input is resized to 64px; pooling is illustrative, not a feature-map visualization. Evaluation uses all held-out samples at 128px.",
        ),
    )
    s = (
        (GALLERY / "cnn/animation.js")
        .read_text()
        .replace(
            "A handwritten digit is max-pooled four times in place, then replaced by its cached CNN prediction.",
            "A brain MRI slice is pooled illustratively in place, then replaced by its cached tumor-type prediction.",
        )
        .replace("'font-size': 300", "'font-size': 32")
    )
    (cnn / "animation.js").write_text(s)
    cache = np.load(RUN / "gan/trajectory.npz")
    trajectory = cache["frames"]
    # Preserve the first 12 seeded latents, without picking visually favorable results.
    epochs = np.unique(np.r_[0, 1, 2, 3, np.linspace(4, len(trajectory) - 1, 27).astype(int)])
    frames = np.rint(np.clip(trajectory[epochs], 0, 1) * 255).astype(np.uint8)
    samples = [
        dict(index=i, pixels=base64.b64encode(frames[:, i, 0].tobytes()).decode())
        for i in range(12)
    ]
    save_data(
        gan,
        "GAN_SAMPLES",
        dict(
            schema=1,
            size=64,
            frames=len(epochs),
            samples=samples,
            epochs=epochs.tolist(),
            source="MPI-Leipzig T2 MRI, https://openneuro.org/datasets/ds000221",
            license="CC0",
            checkpointSha256=hashlib.sha256((RUN / "gan/latest.npz").read_bytes()).hexdigest(),
            note="Fixed latent vectors across actual training checkpoints; interpolation between checkpoints is illustrative, not iterative GAN inference.",
        ),
    )
    s = (
        (GALLERY / "gan/animation.js")
        .read_text()
        .replace("Fashion-MNIST", "Brain MRI")
        .replace("epoch three to thirty", "early to final training checkpoints")
    )
    # Preserve the one-pixel-at-28px gap as an identical fraction of the panel width.
    s = s.replace("batchSize=4,gap=1,", "batchSize=4,gap=data.size/28,")
    (gan / "animation.js").write_text(s)
    for name, original in [("cnn_brain_mri", "cnn"), ("gan_brain_mri", "gan")]:
        folder = GALLERY / name
        html = (GALLERY / original / "index.html").read_text()
        (folder / "index.html").write_text(html)
        scripts = "\n".join((folder / f).read_text() for f in ["samples.js", "animation.js"])
        mount = "CNN" if original == "cnn" else "GAN"
        data = "CNN_SAMPLES" if original == "cnn" else "GAN_SAMPLES"
        (GALLERY / "assets" / f"{name}.html").write_text(
            '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JAXsaw brain MRI '
            + mount
            + '</title></head><body style="margin:0;background:white;color:#171717"><div id="animation" style="width:240px"></div><script>'
            + scripts
            + f'\nwindow.animation=mount{mount}(document.getElementById("animation"),{data});</script></body></html>'
        )
    # A separate five-panel composition; originals and their GIF stay untouched.
    html = (
        (GALLERY / "composite/index.html")
        .read_text()
        .replace("../cnn/", "../cnn_brain_mri/")
        .replace("../gan/", "../gan_brain_mri/")
        .replace('src="animation.js"', 'src="../composite/animation.js"')
    )
    (GALLERY / "neuroscience/index.html").write_text(html)
    scripts = []
    for name in ["cnn_brain_mri", "rnn", "autoencoder", "gan_brain_mri", "transformer"]:
        scripts.extend((GALLERY / name / f).read_text() for f in ["samples.js", "animation.js"])
    scripts.append((GALLERY / "composite/animation.js").read_text())
    (GALLERY / "assets/neuroscience.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JAXsaw MRI gallery</title></head><body style="margin:0;background:white"><div id="gallery-composite" style="width:100%;max-width:1248px;margin:auto"></div><script>'
        + "\n".join(scripts)
        + '\nwindow.gallery=mountGallery(document.getElementById("gallery-composite"));</script></body></html>'
    )
    for name in ["cnn", "gan"]:
        shutil.copyfile(
            RUN / name / "report.json", GALLERY / "neuroscience" / f"{name}-results.json"
        )
    print("Exported MRI variants; approved animations unchanged.")


if __name__ == "__main__":
    export()
