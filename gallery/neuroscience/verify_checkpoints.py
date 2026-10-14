"""Verify saved weights reproduce the cached results, without training."""

import jax.numpy as jnp
import numpy as np
from models import BrainCNN, BrainDiscriminator, BrainGenerator

import neuraljaxlib as nj
from neuraljaxlib import checkpoint, optim


@nj.jit
def infer(model, images):
    return model(images)


def main():
    model = BrainCNN(rngs=nj.Rngs(params=42))
    checkpoint.restore("runs/neuroscience/cnn/best.npz", model=model)
    model.eval()
    data = np.load("runs/neuroscience/cnn/predictions.npz")
    batch = np.array_split(data["images"], int(np.ceil(len(data["images"]) / 32)))[0]
    np.testing.assert_allclose(
        np.asarray(infer(model, jnp.asarray(batch))),
        data["logits"][: len(batch)],
        atol=1e-4,
        rtol=1e-4,
    )
    g = BrainGenerator(rngs=nj.Rngs(params=42))
    d = BrainDiscriminator(rngs=nj.Rngs(params=43))
    go = optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    do = optim.Adam(d.parameters(), lr=0.0001, betas=(0.5, 0.999))
    rngs = nj.Rngs(sample=44)
    checkpoint.restore(
        "runs/neuroscience/gan/latest.npz",
        generator=g,
        discriminator=d,
        g_optimizer=go,
        d_optimizer=do,
        rngs=rngs,
    )
    g.eval()
    data = np.load("runs/neuroscience/gan/trajectory.npz")
    actual = np.asarray(infer(g, jnp.asarray(data["noise"]))) / 2 + 0.5
    np.testing.assert_allclose(actual, data["frames"][-1], atol=1e-5, rtol=1e-5)
    print("Both checkpoints reproduce cached predictions and final generated frames.")


if __name__ == "__main__":
    main()
