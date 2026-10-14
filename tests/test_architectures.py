import jax
import jax.numpy as jnp
import numpy as np
import pytest

import neuraljaxwork as nj
from neuraljaxwork import models, nn, optim, training


@pytest.mark.parametrize("kind", ["rnn", "lstm"])
def test_recurrent_mask_chunk_and_update(kind):
    cls = nn.RNN if kind == "rnn" else nn.LSTM
    layer = cls(1, 3, rngs=nj.Rngs(params=1))
    x = jnp.arange(8.0, dtype=jnp.float32).reshape(2, 4, 1) / 8
    full, carry = layer(x)
    first, state = layer(x[:, :2])
    second, next_state = layer(x[:, 2:], state)
    np.testing.assert_allclose(jnp.concatenate([first, second], 1), full, atol=1e-6)
    for a, b in zip(jax.tree.leaves(carry), jax.tree.leaves(next_state), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)
    padded, state = layer(x, lengths=jnp.array([2, 0]))
    np.testing.assert_array_equal(padded[1], 0)
    np.testing.assert_array_equal(padded[0, 2:], 0)
    model = models.SequenceRegressor(hidden=3, kind=kind, rngs=nj.Rngs(params=3))
    opt = optim.Adam(model.parameters())
    result = training.regression_step(model, opt, (x, jnp.ones((2, 1))))
    assert np.isfinite(result["loss"]) and int(opt.step_count) == 1


def test_attention_causal_and_fully_masked_gradients():
    model = models.TinyLanguageModel(8, 4, width=4, depth=1, heads=2, rngs=nj.Rngs(params=1))
    a = jnp.array([[1, 2, 3, 4]])
    b = a.at[:, 3].set(7)
    np.testing.assert_allclose(model(a)[:, :3], model(b)[:, :3], atol=1e-6)
    opt = optim.Adam(model.parameters())
    assert np.isfinite(training.classifier_step(model, opt, (a, b))["loss"])
    attn = nn.MultiheadAttention(4, 2, rngs=nj.Rngs(params=0))
    x = jnp.ones((1, 2, 4))
    mask = jnp.zeros((1, 1, 2, 2), dtype=bool)
    output, weights = attn(x, x, x, mask=mask, need_weights=True)
    np.testing.assert_array_equal(output, 0)
    np.testing.assert_array_equal(weights, 0)
    grads = nj.grad(lambda m: m(x, x, x, mask=mask).sum())(attn)
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(grads))


def test_cnn_and_autoencoder_updates():
    images = jnp.ones((2, 1, 28, 28)) * 0.5
    model = models.DigitCNN(width=2, rngs=nj.Rngs(params=0))
    before = [
        np.asarray(c.weight.value).copy()
        for c in (model.conv1, model.conv2, model.conv3, model.conv4)
    ]
    opt = optim.Adam(model.parameters())
    assert np.isfinite(
        training.cnn_step(model, opt, (images, jnp.array([0, 1])), nj.Rngs(dropout=2))["loss"]
    )
    assert model.eval()(images).shape == (2, 10)
    x = jnp.pad(images, ((0, 0), (0, 0), (2, 2), (2, 2)))
    for conv, size, old in zip(
        (model.conv1, model.conv2, model.conv3, model.conv4), (16, 8, 4, 2), before, strict=True
    ):
        x = model.pool(jax.nn.relu(conv(x)))
        assert x.shape[-2:] == (size, size)
        assert not np.array_equal(old, conv.weight.value)
    ae = models.Autoencoder(latent_size=2, width=4, rngs=nj.Rngs(params=2))
    ae_opt = optim.Adam(ae.parameters())
    assert np.isfinite(training.autoencoder_step(ae, ae_opt, images)["loss"])
    assert ae(images)["reconstruction"].shape == images.shape
    assert ae(images)["latent"].shape == (2, 2)


def test_gan_updates_only_selected_network_and_has_signal():
    g = models.Generator(latent_size=2, width=4, rngs=nj.Rngs(params=1))
    d = models.Discriminator(width=4, rngs=nj.Rngs(params=2))
    go, do = optim.Adam(g.parameters()), optim.Adam(d.parameters())
    z, real = jnp.ones((2, 2)), jnp.ones((2, 1, 28, 28))
    g_before = g.state_dict()
    training.discriminator_step(g, d, do, real, z)
    for key, value in g.state_dict().items():
        np.testing.assert_array_equal(value, g_before[key])
    d_before = d.state_dict()
    result = training.generator_step(g, d, go, z)
    assert np.isfinite(result["g_loss"])
    assert any(not np.array_equal(v, g_before[k]) for k, v in g.state_dict().items())
    for key, value in d.state_dict().items():
        np.testing.assert_array_equal(value, d_before[key])


def test_convolution_math_and_transpose():
    conv = nn.Conv2d(1, 1, 2, bias=False, rngs=nj.Rngs(params=0))
    conv.weight.value = jnp.ones((1, 1, 2, 2))
    x = jnp.arange(9.0, dtype=jnp.float32).reshape(1, 1, 3, 3)
    np.testing.assert_allclose(conv(x), [[[[8.0, 12.0], [20.0, 24.0]]]])
    transpose = nn.ConvTranspose2d(1, 1, 2, bias=False, rngs=nj.Rngs(params=1))
    transpose.weight.value = conv.weight.value
    y = jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    np.testing.assert_allclose(jnp.sum(conv(x) * y), jnp.sum(x * transpose(y)))
    ae = models.ConvAutoencoder(channels=2, rngs=nj.Rngs(params=0))
    images = jnp.ones((1, 1, 28, 28))
    assert ae(images)["reconstruction"].shape == images.shape
    assert np.isfinite(training.autoencoder_step(ae, optim.Adam(ae.parameters()), images)["loss"])


def test_pet_and_sequence_classifier():
    pet = models.PetCNN(width=2, rngs=nj.Rngs(params=0))
    x = jnp.ones((2, 3, 8, 8))
    assert pet(x).shape == (2, 2)
    assert np.isfinite(
        training.classifier_step(pet, optim.SGD(pet.parameters()), (x, jnp.array([0, 1])))["loss"]
    )
    model = models.LSTMClassifier(6, 2, embedding_dim=2, hidden=3, rngs=nj.Rngs(params=0))
    output = model(jnp.array([[1, 2], [3, 4]]), lengths=jnp.array([1, 2]))
    assert output["logits"].shape == (2, 2)


def test_fashion_gan_phase_state():
    g = models.FashionGenerator(latent_size=2, width=2, rngs=nj.Rngs(params=1))
    d = models.FashionDiscriminator(width=2, rngs=nj.Rngs(params=2))
    go, do = optim.Adam(g.parameters()), optim.Adam(d.parameters())
    g.eval()
    saved = g.state_dict()
    z = jnp.ones((2, 2))
    real = jnp.ones((2, 1, 28, 28))
    training.discriminator_step(g, d, do, real, z)
    for key, value in g.state_dict().items():
        np.testing.assert_array_equal(value, saved[key])
    g.train()
    training.generator_step(g, d.eval(), go, z)
    assert g.eval()(z).shape == real.shape
    assert any(not np.array_equal(v, saved[k]) for k, v in g.state_dict().items())
