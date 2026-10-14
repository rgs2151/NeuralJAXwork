"""Small single-process datasets and reproducible batch loading."""

import jax
import jax.numpy as jnp
import numpy as np


class TensorDataset:
    def __init__(self, *arrays):
        if not arrays or len({len(a) for a in arrays}) != 1:
            raise ValueError("Provide equally sized arrays")
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, index):
        return tuple(a[index] for a in self.arrays)


def collate(samples):
    return jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *samples)


class DataLoader:
    """Deterministic epoch shuffle; state records the next sample cursor."""

    def __init__(
        self, dataset, batch_size=32, shuffle=False, seed=0, drop_last=False, collate_fn=collate
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.seed, self.drop_last = shuffle, seed, drop_last
        self.collate_fn = collate_fn
        self.epoch, self.cursor = 0, 0

    def iter_epoch(self, epoch):
        if self.epoch != epoch:
            self.epoch, self.cursor = epoch, 0
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).shuffle(indices)
        while self.cursor < len(indices):
            end = min(self.cursor + self.batch_size, len(indices))
            selected = indices[self.cursor : end]
            if self.drop_last and len(selected) < self.batch_size:
                self.cursor = len(indices)
                break
            batch = self.collate_fn([self.dataset[int(i)] for i in selected])
            self.cursor = end
            yield batch

    def __iter__(self):
        # A fresh ordinary loop is one epoch; explicit iter_epoch supports resume.
        if self.cursor >= len(self.dataset):
            self.epoch += 1
            self.cursor = 0
        return self.iter_epoch(self.epoch)

    def state_dict(self):
        return {
            key: getattr(self, key)
            for key in ("epoch", "cursor", "batch_size", "shuffle", "seed", "drop_last")
        } | {"size": len(self.dataset)}

    def load_state_dict(self, state):
        current = self.state_dict()
        for key in ("size", "batch_size", "shuffle", "seed", "drop_last"):
            if state[key] != current[key]:
                raise ValueError(f"Loader configuration mismatch: {key}")
        if not 0 <= state["cursor"] <= len(self.dataset) or state["epoch"] < 0:
            raise ValueError("Invalid loader cursor/epoch")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
