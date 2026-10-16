"""Recover compact training frames from cached, fixed-noise GAN figure grids."""

import base64
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import jaxsaw as js
from jaxsaw import models

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/gallery/gan"
INDICES = [2, 4, 7, 12, 16, 17, 18, 20, 21, 22, 28, 31]
with np.load(RUN / "samples.npz", allow_pickle=False) as cache:
    final = np.rint(np.clip(cache["samples"][:, 0] / 2 + 0.5, 0, 1) * 255).astype(np.uint8)
    noise = cache["noise"].copy()
# Reconstruct the cached figure's data-to-pixel transform, sampling pixel centers.
fig, axes = plt.subplots(4, 8, figsize=(8, 5.2), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
    ax.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)
fig.suptitle("Fashion-MNIST GAN: fixed noise")
fig.tight_layout()
fig.canvas.draw()
height = fig.canvas.get_width_height()[1]
yy, xx = np.mgrid[:28, :28]
coordinates = np.stack([xx.ravel(), yy.ravel()], axis=1)
centers = [ax.transData.transform(coordinates) for ax in axes.flat]
plt.close(fig)


def recover(path):
    pixels = np.asarray(Image.open(path).convert("L"))
    assert pixels.shape == (520, 800)
    return np.stack(
        [
            pixels[
                np.clip((height - xy[:, 1]).astype(int), 0, height - 1), xy[:, 0].astype(int)
            ].reshape(28, 28)
            for xy in centers
        ]
    )


frames = [recover(RUN / f"samples-{epoch:03}.png") for epoch in range(30)]
error = np.abs(frames[-1].astype(float) - final)
assert error.mean() < 5, f"Grid registration failed: mean pixel error {error.mean()}"
# Epoch zero is the same initialized generator and same latent batch as the run.
model = models.FashionGenerator(rngs=js.Rngs(params=0)).eval()
initial = np.rint(np.clip(np.asarray(model(noise))[:, 0] / 2 + 0.5, 0, 1) * 255).astype(np.uint8)
frames.insert(0, initial)
frames[-1] = final
samples = []
for index in INDICES:
    payload = np.stack([frame[index] for frame in frames]).tobytes()
    samples.append(dict(index=index, pixels=base64.b64encode(payload).decode()))
result = dict(
    schema=1,
    size=28,
    frames=len(frames),
    samples=samples,
    source="runs/gallery/gan",
    sourceSha256=hashlib.sha256((RUN / "samples.npz").read_bytes()).hexdigest(),
    epochs=list(range(31)),
    gridRecoveryMeanAbsoluteError=float(error.mean()),
    note="Fixed latent vectors across training epochs. Epochs 1–29 recovered from rendered figure grids; final frame uses raw saved samples. Epoch 0 reconstructs the seeded initial generator. Playback interpolation is illustrative.",
)
Path(__file__).with_name("samples.js").write_text(
    "window.GAN_SAMPLES = " + json.dumps(result, separators=(",", ":")) + ";\n"
)
print(
    f"Exported {len(samples)} samples × {len(frames)} frames; recovery MAE {error.mean():.3f}/255."
)
