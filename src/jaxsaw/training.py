"""Small reusable update steps; all optimization is explicit."""

import jax.numpy as jnp

from .core import jit, random, stop_gradient, value_and_grad
from .nn import functional as F


@jit
def classifier_step(model, optimizer, batch):
    x, y = batch

    def objective(candidate):
        logits = candidate(x)
        return F.cross_entropy(logits, y), (logits.argmax(-1) == y).sum()

    (loss, correct), grads = value_and_grad(objective, has_aux=True)(model)
    optimizer.step(grads)
    return {"loss": loss, "correct": correct, "count": y.size}


@jit
def cnn_step(model, optimizer, batch, rngs):
    x, y = batch

    def objective(candidate):
        logits = candidate(x, rngs=rngs)
        return F.cross_entropy(logits, y), (logits.argmax(-1) == y).sum()

    (loss, correct), grads = value_and_grad(objective, has_aux=True)(model)
    optimizer.step(grads)
    return {"loss": loss, "correct": correct, "count": y.size}


@jit
def regression_step(model, optimizer, batch):
    x, y = batch
    loss, grads = value_and_grad(lambda m: F.mse_loss(m(x)["prediction"], y))(model)
    optimizer.step(grads)
    return {"loss": loss, "count": y.shape[0]}


@jit
def autoencoder_step(model, optimizer, images):
    loss, grads = value_and_grad(lambda m: F.mse_loss(m(images)["reconstruction"], images))(model)
    optimizer.step(grads)
    return {"loss": loss, "count": images.shape[0]}


@jit
def discriminator_step(generator, discriminator, optimizer, real, z):
    fake = stop_gradient(generator(z))

    def objective(candidate):
        real_logits, fake_logits = candidate(real), candidate(fake)
        return F.binary_cross_entropy_with_logits(
            real_logits, jnp.ones_like(real_logits)
        ) + F.binary_cross_entropy_with_logits(fake_logits, jnp.zeros_like(fake_logits))

    loss, grads = value_and_grad(objective)(discriminator)
    optimizer.step(grads)
    return {"d_loss": loss}


@jit
def generator_step(generator, discriminator, optimizer, z):
    def objective(candidate):
        logits = discriminator(candidate(z))
        return F.binary_cross_entropy_with_logits(logits, jnp.ones_like(logits))

    loss, grads = value_and_grad(objective)(generator)
    optimizer.step(grads)
    return {"g_loss": loss}


def gan_step(generator, discriminator, g_optimizer, d_optimizer, real, rngs, latent_size=64):
    """One D update then one G update; inactive model buffers stay in eval mode."""
    generator.eval()
    discriminator.train()
    z = random.normal(rngs.next("sample"), (real.shape[0], latent_size))
    result = discriminator_step(generator, discriminator, d_optimizer, real, z)
    generator.train()
    discriminator.eval()
    z = random.normal(rngs.next("sample"), (real.shape[0], latent_size))
    return {**result, **generator_step(generator, discriminator, g_optimizer, z)}


class Task:
    """Optional experiment wrapper; training_step owns all optimizer updates.

    Override training_step/validation_step. Declare persistent graph objects in
    checkpoint_objects; the default contains model, optimizer, and optional RNGs.
    """

    def __init__(self, model, optimizer, rngs=None):
        self.model, self.optimizer, self.rngs = model, optimizer, rngs

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def set_phase(self, training):
        self.model.train(training)

    def checkpoint_objects(self):
        objects = {"model": self.model, "optimizer": self.optimizer}
        if self.rngs is not None:
            objects["rngs"] = self.rngs
        return objects


class ModelCheckpoint:
    """Save the best validation metric; checkpoint state remains explicit."""

    def __init__(self, path, monitor="val/loss", mode="min"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be min or max")
        self.path, self.monitor, self.mode = path, monitor, mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def __call__(self, task, epoch, metrics):
        from . import checkpoint

        value = metrics[self.monitor]
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            checkpoint.save(
                self.path,
                **task.checkpoint_objects(),
                progress={"next_epoch": epoch + 1},
                metadata={"metrics": metrics},
            )


class Trainer:
    """Host-side epoch loop delegating optimization to the task step."""

    def __init__(self, max_epochs, callbacks=(), run=None):
        self.max_epochs, self.callbacks, self.run = max_epochs, tuple(callbacks), run

    def fit(self, task, *, train_loader, val_loader):
        history = []
        for epoch in range(self.max_epochs):
            record = {}
            for phase, loader, step in (
                ("train", train_loader, task.training_step),
                ("val", val_loader, task.validation_step),
            ):
                task.set_phase(phase == "train")
                total, count, correct, has_correct = 0.0, 0, 0, False
                for batch in loader.iter_epoch(epoch):
                    result = step(batch)
                    n = int(result["count"])
                    total += float(result["loss"]) * n
                    count += n
                    has_correct = "correct" in result
                    correct += int(result.get("correct", 0))
                if not count:
                    raise ValueError(f"No valid samples in {phase}")
                values = {"loss": total / count}
                if has_correct:
                    values["accuracy"] = correct / count
                record.update({f"{phase}/{k}": v for k, v in values.items()})
                if self.run is not None:
                    self.run.log_metrics(epoch=epoch, split=phase, **values)
            for callback in self.callbacks:
                callback(task, epoch, record)
            history.append(record)
        return history
