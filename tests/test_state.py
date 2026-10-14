import json

import jax.numpy as jnp
import numpy as np
import pytest

import neuraljaxlib as nj
from neuraljaxlib import checkpoint, data, inspection, nn, optim


def test_checkpoint_next_update_and_loader_resume(tmp_path):
    model = nn.Linear(2, 1, rngs=nj.Rngs(params=0))
    opt = optim.Adam(model.parameters())
    rngs = nj.Rngs(sample=3)

    @nj.jit
    def step(model, opt, rngs):
        x = nj.random.normal(rngs.next("sample"), (2, 2))
        loss, grads = nj.value_and_grad(lambda m: (m(x) ** 2).mean())(model)
        opt.step(grads)
        return loss

    step(model, opt, rngs)
    loader = data.DataLoader(data.TensorDataset(np.arange(9)), batch_size=2, shuffle=True, seed=4)
    it = loader.iter_epoch(0)
    next(it)
    snapshot = loader.state_dict()
    expected_batch = next(it)
    path = tmp_path / "state.npz"
    checkpoint.save(path, model=model, optimizer=opt, rngs=rngs, data_state=snapshot)
    expected = step(model, opt, rngs)
    expected_state = model.state_dict()
    result = checkpoint.restore(path, model=model, optimizer=opt, rngs=rngs)
    loader.load_state_dict(result.data_state)
    np.testing.assert_array_equal(next(loader.iter_epoch(0))[0], expected_batch[0])
    np.testing.assert_allclose(step(model, opt, rngs), expected, atol=1e-7)
    for key, value in model.state_dict().items():
        np.testing.assert_allclose(value, expected_state[key], atol=1e-7)
    with pytest.raises(ValueError):
        checkpoint.restore(
            path, model=model, optimizer=optim.Adam(model.parameters(), lr=0.02), rngs=rngs
        )


def test_inspection_does_not_change_mode_or_buffers():
    model = nn.BatchNorm2d(2)
    state = model.state_dict()
    shape = inspection.output_spec(model, jnp.ones((2, 2, 4, 4)))
    assert shape["shape"] == [2, 2, 4, 4] and model.training
    for k, v in model.state_dict().items():
        np.testing.assert_array_equal(v, state[k])
    layer = nn.Linear(2, 2, rngs=nj.Rngs(params=0))
    description = inspection.describe(nn.Sequential(layer, layer))
    assert description["root"]["fields"]["items"][1] == {"alias": "model.items.0"}


def test_trainer_owns_no_extra_update(tmp_path):
    from neuraljaxlib import training
    from neuraljaxlib.nn import functional as F

    class ClassifierTask(training.Task):
        def training_step(self, batch):
            return training.classifier_step(self.model, self.optimizer, batch)

        def validation_step(self, batch):
            x, y = batch
            logits = self.model(x)
            return {
                "loss": F.cross_entropy(logits, y),
                "count": len(y),
                "correct": (logits.argmax(-1) == y).sum(),
            }

    model = nn.Linear(2, 2, rngs=nj.Rngs(params=0))
    optimizer = optim.SGD(model.parameters())
    task = ClassifierTask(model, optimizer)
    ds = data.TensorDataset(np.ones((4, 2), dtype=np.float32), np.array([0, 1, 0, 1]))
    callback = training.ModelCheckpoint(tmp_path / "best.npz")
    history = training.Trainer(1, callbacks=[callback]).fit(
        task,
        train_loader=data.DataLoader(ds, batch_size=2),
        val_loader=data.DataLoader(ds, batch_size=2),
    )
    assert int(optimizer.step_count) == 2 and "val/loss" in history[0]
    assert (tmp_path / "best.npz").exists()


def test_checkpoint_before_package_rename(tmp_path):
    model = nn.Linear(2, 1, rngs=nj.Rngs(params=5))
    path = tmp_path / "before-rename.npz"
    expected = np.asarray(model(jnp.ones((1, 2))))
    checkpoint.save(path, model=model)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    manifest = json.loads(str(arrays["manifest"]))
    manifest["signature"] = json.loads(
        json.dumps(manifest["signature"]).replace("neuraljaxlib.", "neuraljaxwork.")
    )
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(path, **arrays)
    fresh = nn.Linear(2, 1, rngs=nj.Rngs(params=9))
    checkpoint.restore(path, model=fresh)
    np.testing.assert_array_equal(fresh(jnp.ones((1, 2))), expected)
