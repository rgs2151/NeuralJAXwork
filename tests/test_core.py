import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxsaw as js
from jaxsaw import nn, optim
from jaxsaw.nn import functional as F


def test_explicit_training_eager_and_jit():
    model = nn.Linear(2, 1, rngs=js.Rngs(params=0))
    opt = optim.SGD(model.parameters(), lr=0.1)
    x, y = jnp.array([[1.0, 0.0], [0.0, 1.0]]), jnp.array([[1.0], [-1.0]])

    def step(model, opt):
        loss, grads = js.value_and_grad(lambda m: F.mse_loss(m(x), y))(model)
        opt.step(grads)
        return loss

    graph, state = js.split((model, opt))
    clone, clone_opt = js.merge(graph, state)
    np.testing.assert_allclose(step(model, opt), js.jit(step)(clone, clone_opt), rtol=1e-6)
    for name, value in model.state_dict().items():
        np.testing.assert_allclose(value, clone.state_dict()[name], atol=1e-6)
    compiled = js.jit(step)
    before = float(F.mse_loss(model(x), y))
    for _ in range(12):
        compiled(model, opt)
    assert float(F.mse_loss(model(x), y)) < before / 5
    assert int(opt.step_count) == 13


def test_initialization_sharing_and_roundtrip():
    rngs = js.Rngs(params=2)
    a, b = nn.Linear(2, 2, rngs=rngs), nn.Linear(2, 2, rngs=rngs)
    assert not np.array_equal(a.weight.value, b.weight.value)
    model = nn.Sequential(a, a)
    assert len(list(model.parameters())) == 2
    state = model.state_dict()
    a.weight.value += 1
    model.load_state_dict(state)
    np.testing.assert_array_equal(a.weight.value, state["items.0.weight"])
    bad = dict(state)
    bad["items.0.weight"] = np.zeros((9, 9), dtype=np.float32)
    with pytest.raises(ValueError):
        model.load_state_dict(bad)


def test_dropout_and_batchnorm_state():
    dropout = nn.Dropout(0.5)
    x = jnp.ones((64,))
    rngs = js.Rngs(dropout=1)
    first = js.jit(lambda m, r: m(x, rngs=r))(dropout, rngs)
    second = dropout(x, rngs=rngs)
    assert not np.array_equal(first, second)
    dropout.eval()
    np.testing.assert_array_equal(dropout(x), x)
    bn = nn.BatchNorm2d(1)
    data = jnp.arange(8.0, dtype=jnp.float32).reshape(2, 1, 2, 2)
    js.jit(lambda m: js.value_and_grad(lambda candidate: candidate(data).sum())(m))(bn)
    np.testing.assert_allclose(bn.running_mean.value, [0.35], atol=1e-6)
    saved = bn.state_dict()
    bn.eval()(data)
    np.testing.assert_array_equal(bn.running_mean.value, saved["running_mean"])


def test_loss_shapes_masks_and_gradients():
    logits = jnp.zeros((2, 3))
    labels = jnp.array([0, 1])
    np.testing.assert_allclose(F.cross_entropy(logits, labels), jnp.log(3), rtol=1e-6)
    mask = jnp.array([False, False])
    assert F.cross_entropy(logits, labels, mask=mask) == 0
    np.testing.assert_array_equal(
        jax.grad(lambda x: F.cross_entropy(x, labels, mask=mask))(logits), 0
    )
    assert jnp.isfinite(
        F.binary_cross_entropy_with_logits(jnp.array([100.0, -100.0]), jnp.array([1.0, 0.0]))
    )
    with pytest.raises(ValueError):
        F.mse_loss(jnp.zeros((2, 1)), jnp.zeros((2,)))
