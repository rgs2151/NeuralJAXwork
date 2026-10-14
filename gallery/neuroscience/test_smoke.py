"""Small CPU checks; no data downloads or training runs."""

import jax.numpy as jnp
import numpy as np
from models import BrainCNN, BrainDiscriminator, BrainGenerator

import neuraljaxlib as nj


def test_shapes_and_finite_outputs():
    cnn = BrainCNN(rngs=nj.Rngs(params=1)).eval()
    g = BrainGenerator(rngs=nj.Rngs(params=2)).eval()
    d = BrainDiscriminator(rngs=nj.Rngs(params=3)).eval()
    logits = cnn(jnp.zeros((2, 1, 128, 128)))
    generated = g(jnp.ones((2, 128)))
    scores = d(generated)
    assert logits.shape == (2, 3)
    assert generated.shape == (2, 1, 64, 64)
    assert scores.shape == (2, 1)
    assert np.isfinite(np.asarray(logits)).all()
    assert np.isfinite(np.asarray(scores)).all()
    assert float(jnp.max(jnp.abs(generated))) <= 1


def test_cnn_checkpoint_after_jit(tmp_path):
    from neuraljaxlib import checkpoint

    model = BrainCNN(rngs=nj.Rngs(params=1)).eval()
    images = jnp.ones((1, 1, 128, 128))
    forward = nj.jit(lambda model, x: model(x))
    expected = np.asarray(forward(model, images))
    path = tmp_path / "model.npz"
    checkpoint.save(path, model=model)
    restored = BrainCNN(rngs=nj.Rngs(params=2)).eval()
    checkpoint.restore(path, model=restored)
    np.testing.assert_allclose(np.asarray(forward(restored, images)), expected, atol=1e-6)
