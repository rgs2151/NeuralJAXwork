"""Reproducible local CNN and GAN experiments; no CI training or downloads."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from models import BrainCNN, BrainDiscriminator, BrainGenerator
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

import jaxsaw as js
from jaxsaw import checkpoint, optim, training


def grid(x, path):
    x = np.rint(np.clip(np.asarray(x)[:, 0], 0, 1) * 255).astype("uint8")
    n = min(len(x), 32)
    h, w = x.shape[1:]
    canvas = np.zeros((4 * h, 8 * w), np.uint8)
    for i in range(n):
        canvas[(i // 8) * h : (i // 8 + 1) * h, (i % 8) * w : (i % 8 + 1) * w] = x[i]
    Image.fromarray(canvas).save(path)


@js.jit
def infer(model, x):
    return model(x)


def cnn(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load("data/neuroscience/cnn.npz")
    x = data["images"]
    y = data["labels"].astype(np.int32)
    fold = data["fold"]
    groups = data["patients"]
    train = np.flatnonzero(fold >= 2)
    val = np.flatnonzero(fold == 1)
    test = np.flatnonzero(fold == 0)
    for a, b in ((train, val), (train, test), (val, test)):
        assert not set(groups[a]) & set(groups[b])
    model = BrainCNN(rngs=js.Rngs(params=42))
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    rng = np.random.default_rng(42)
    best = -1
    history = []

    def evaluate(indices):
        model.eval()
        logits = np.concatenate(
            [
                np.asarray(infer(model, jnp.asarray(x[ids])))
                for ids in np.array_split(indices, max(1, int(np.ceil(len(indices) / 32))))
            ]
        )
        p = logits.argmax(1)
        return dict(
            accuracy=float(accuracy_score(y[indices], p)),
            balanced_accuracy=float(balanced_accuracy_score(y[indices], p)),
            confusion_matrix=confusion_matrix(y[indices], p, labels=[0, 1, 2]).tolist(),
        ), logits

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        order = rng.permutation(train)
        losses = []
        for offset in range(0, len(order) - 31, 32):
            ids = order[offset : offset + 32]
            batch = x[ids].copy()
            flips = rng.random(32) < 0.5
            batch[flips] = batch[flips, :, :, ::-1]
            # Mild intensity and translation augmentation; labels remain unchanged.
            batch = np.clip(batch * rng.uniform(0.9, 1.1, (32, 1, 1, 1)), 0, 1).astype(np.float32)
            dy, dx = rng.integers(-5, 6, 2)
            padded = np.pad(batch, ((0, 0), (0, 0), (5, 5), (5, 5)))
            batch = padded[:, :, 5 + dy : 133 + dy, 5 + dx : 133 + dx]
            result = training.classifier_step(
                model, optimizer, (jnp.asarray(batch), jnp.asarray(y[ids]))
            )
            losses.append(float(result["loss"]))
        metrics, _ = evaluate(val)
        row = dict(
            epoch=epoch + 1,
            loss=float(np.mean(losses)),
            validation=metrics,
            seconds=time.time() - start,
        )
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["balanced_accuracy"] > best:
            best = metrics["balanced_accuracy"]
            checkpoint.save(out / "best.npz", model=model, metadata={"epoch": epoch + 1})
            best_epoch = epoch + 1
        (out / "history.json").write_text(json.dumps(history, indent=2))
    checkpoint.restore(out / "best.npz", model=model)
    metrics, logits = evaluate(test)
    report = dict(
        dataset="Cheng brain tumor MRI",
        classes=["Meningioma", "Glioma", "Pituitary"],
        seed=42,
        best_epoch=best_epoch,
        selection="validation balanced accuracy",
        test=metrics,
        splits={
            name: dict(images=len(ids), patients=len(set(groups[ids])))
            for name, ids in [("train", train), ("validation", val), ("test", test)]
        },
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out / "predictions.npz",
        indices=test,
        logits=logits,
        labels=y[test],
        images=x[test],
        patients=groups[test],
        ids=data["ids"][test],
    )
    print(json.dumps(report), flush=True)


def gan(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load("data/neuroscience/gan.npz")
    train = data["images"][~data["test"]]
    test = data["images"][data["test"]]
    assert not set(data["patients"][data["test"]]) & set(data["patients"][~data["test"]])
    g = BrainGenerator(rngs=js.Rngs(params=42))
    d = BrainDiscriminator(rngs=js.Rngs(params=43))
    go = optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    do = optim.Adam(d.parameters(), lr=0.0001, betas=(0.5, 0.999))
    rngs = js.Rngs(sample=44)
    fixed = jax.random.normal(jax.random.key(45), (32, 128))
    rng = np.random.default_rng(46)
    frames = []
    history = []
    grid(train[:32], out / "real-train.png")
    grid(test[:32], out / "real-test.png")

    def sample():
        g.eval()
        return np.asarray(infer(g, fixed)) / 2 + 0.5

    frames.append(sample())
    for epoch in range(args.epochs):
        start = time.time()
        order = rng.permutation(len(train))
        losses = []
        for offset in range(0, len(order) - 31, 32):
            real = train[order[offset : offset + 32]].copy() * 2 - 1
            flip = rng.random(32) < 0.5
            real[flip] = real[flip, :, :, ::-1]
            result = training.gan_step(g, d, go, do, jnp.asarray(real), rngs, latent_size=128)
            losses.append([float(result["d_loss"]), float(result["g_loss"])])
        images = sample()
        frames.append(images)
        row = dict(
            epoch=epoch + 1,
            d_loss=float(np.mean(losses, axis=0)[0]),
            g_loss=float(np.mean(losses, axis=0)[1]),
            seconds=time.time() - start,
        )
        history.append(row)
        print(json.dumps(row), flush=True)
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            grid(images, out / f"samples-{epoch + 1:03}.png")
            checkpoint.save(
                out / "latest.npz",
                generator=g,
                discriminator=d,
                g_optimizer=go,
                d_optimizer=do,
                rngs=rngs,
                metadata={"epoch": epoch + 1},
            )
            np.savez_compressed(
                out / "trajectory.npz", frames=np.array(frames), noise=np.asarray(fixed)
            )
            (out / "history.json").write_text(json.dumps(history, indent=2))
    # Pixel-space nearest-neighbor check is a diagnostic, not a perceptual quality score.
    flat = images.reshape(len(images), -1)
    reference = train.reshape(len(train), -1)
    mse = ((flat[:, None, :] - reference[None, :, :]) ** 2).mean(2)
    nearest = mse.argmin(1)
    grid(train[nearest], out / "nearest-training.png")
    report = dict(
        dataset="MPI-Leipzig Mind-Brain-Body (ds000221)",
        epochs=args.epochs,
        train_images=len(train),
        test_images=len(test),
        train_participants=len(set(data["patients"][~data["test"]])),
        test_participants=len(set(data["patients"][data["test"]])),
        nearest_training_rmse=np.sqrt(mse.min(1)).tolist(),
        generated_pixel_std=float(images.std()),
        mean_across_sample_std=float(images.std(0).mean()),
        note="Fixed latent trajectory. Pixel nearest neighbors do not establish novelty or clinical validity.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["cnn", "gan"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(jax.devices(), flush=True)
    cnn(a) if a.task == "cnn" else gan(a)
