"""Export compact CNN activations from the cached tutorial model; no training."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from neuraljaxlib import Rngs, checkpoint, models

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/gallery-four-layer/cnn"
model = models.DigitCNN(width=16, rngs=Rngs(params=0))
# Match the checkpoint's separate, equal pooling configuration tuples.
model.pool.stride = tuple(list(model.pool.stride))
checkpoint.restore(RUN / "best.npz", model=model)
model.eval()
archive = np.load(RUN / "gallery.npz")
samples = []
for digit in (7, 3, 8):
    index = np.flatnonzero((archive["targets"] == digit) & archive["correct"])[0]
    x = jnp.asarray(archive["images"][index : index + 1])
    padded = jnp.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)))
    maps = [np.asarray(padded)[0, 0]]
    scales = [1]
    channels = []
    activation = padded
    for conv in (model.conv1, model.conv2, model.conv3, model.conv4):
        convolved = jax.nn.relu(conv(activation))
        activation = model.pool(convolved)
        channel = int(np.asarray(convolved)[0].std(axis=(1, 2)).argmax())
        channels.append(channel)
        a, b = np.asarray(convolved)[0, channel], np.asarray(activation)[0, channel]
        n = b.shape[0]
        np.testing.assert_array_equal(a.reshape(n, 2, n, 2).max(axis=(1, 3)), b)
        maps.extend([a, b])
        scales.extend([max(float(a.max()), 1e-8)] * 2)
    probabilities = np.asarray(jax.nn.softmax(model(x)))[0]
    assert probabilities.argmax() == digit
    samples.append(
        dict(
            digit=int(digit),
            testIndex=int(archive["test_indices"][index]),
            prediction=int(probabilities.argmax()),
            channels=channels,
            probabilities=probabilities.round(7).tolist(),
            maps=[
                dict(size=a.shape[0], values=np.rint(a / s * 255).astype(int).ravel().tolist())
                for a, s in zip(maps, scales, strict=True)
            ],
        )
    )
result = dict(
    schema=2,
    checkpointSha256=hashlib.sha256((RUN / "best.npz").read_bytes()).hexdigest(),
    selection="First correctly classified cached example of digits 7, 3, and 8.",
    display="Four-stage model; maps[0] is padded input. Animation pools input illustratively.",
    samples=samples,
)
Path(__file__).with_name("samples.js").write_text(
    "window.CNN_SAMPLES = " + json.dumps(result, separators=(",", ":")) + ";\n"
)
print("Exported three verified predictions and pooled feature maps.")
