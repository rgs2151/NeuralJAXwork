"""Run small reference experiments: python -m examples.run --help."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import neuraljaxwork as nj
from neuraljaxwork import checkpoint, data, inspection, models, optim, training
from neuraljaxwork.experiments import Run
from neuraljaxwork.nn import functional as F

from . import datasets, plots

# Original tiny corpus for an offline mechanics demo, not a language benchmark.
TEXT = (
    "the moon rises over the quiet river. the sun warms the green meadow. "
    "a small bird sings beside the water. the wind moves through the trees.\n"
) * 100


def run_experiment(
    task,
    *,
    epochs=2,
    batch_size=32,
    limit=512,
    seed=0,
    output=None,
    download=False,
    cache="data",
    smoke=False,
    text_file=None,
    eval_limit=1000,
    context_length=32,
    model_width=32,
    model_depth=2,
):
    if epochs < 1 or limit < 4 or batch_size < 1:
        raise ValueError("epochs>=1, limit>=4, batch_size>=1 are required")
    output = Path(output or f"runs/{task}-{time.time_ns()}")
    config = dict(
        task=task,
        epochs=epochs,
        batch_size=batch_size,
        limit=limit,
        seed=seed,
        smoke=smoke,
        text_file=text_file,
        eval_limit=eval_limit,
        context_length=context_length,
        model_width=model_width,
        model_depth=model_depth,
    )
    run = Run(output, config)
    rngs = nj.Rngs(params=seed)
    runtime = nj.Rngs(dropout=seed + 1, sample=seed + 2)
    metadata = {"synthetic_smoke": smoke}
    if task == "titanic":
        if smoke:
            r = np.random.default_rng(seed)
            x = r.normal(size=(48, 7)).astype(np.float32)
            y = (x[:, 0] > 0).astype(np.int32)
            train, val, test = (x[:32], y[:32]), (x[32:40], y[32:40]), (x[40:], y[40:])
        else:
            (train, val, test), metadata = datasets.titanic(cache, download, seed)
        model = models.MLP(train[0].shape[1], [16 if smoke else 64, 16], 2, rngs=rngs)
    elif task in ("cnn", "autoencoder", "gan"):
        if smoke:
            r = np.random.default_rng(seed)
            train = (
                r.random((16, 1, 28, 28), dtype=np.float32),
                np.arange(16, dtype=np.int32) % 10,
            )
            val = test = (train[0][-4:].copy(), train[1][-4:].copy())
            train = (train[0][:-4], train[1][:-4])
        else:
            source = "fashion_mnist" if task == "gan" else "mnist"
            official_train, test = datasets.images(source, cache, download)
            ids = np.random.default_rng(seed).permutation(len(official_train[0]))
            val_ids, train_ids = ids[:5000], ids[5000:]
            train = tuple(a[train_ids[:limit]] for a in official_train)
            val = tuple(a[val_ids[: min(limit, eval_limit)]] for a in official_train)
            test = tuple(a[: min(limit, eval_limit)] for a in test)
            metadata = {
                "dataset": source,
                "train_ids": train_ids[:limit].tolist(),
                "val_ids": val_ids[: min(limit, eval_limit)].tolist(),
                "test_ids": list(range(len(test[0]))),
                "normalization": "[0,1] NCHW",
            }
        if task == "cnn":
            model = models.DigitCNN(width=2 if smoke else 16, rngs=rngs)
        elif task == "autoencoder":
            model = models.Autoencoder(
                latent_size=8 if smoke else 16, width=8 if smoke else 256, rngs=rngs
            )
        else:
            latent_size = 16 if smoke else 64
            model = (
                models.Generator(latent_size=16, width=8, rngs=rngs)
                if smoke
                else models.FashionGenerator(latent_size=latent_size, rngs=rngs)
            )
            discriminator = (
                models.Discriminator(width=8, rngs=rngs)
                if smoke
                else models.FashionDiscriminator(rngs=rngs)
            )
    elif task in ("rnn", "lstm"):
        train = datasets.sine_sequences(16 if smoke else limit, length=12, seed=seed)
        val = datasets.sine_sequences(8 if smoke else 128, length=12, seed=seed + 1)
        test = datasets.sine_sequences(8 if smoke else 128, length=12, seed=seed + 2)
        model = models.SequenceRegressor(hidden=4 if smoke else 32, kind=task, rngs=rngs)
        metadata = {"dataset": "independent sine trajectories", "seeds": [seed, seed + 1, seed + 2]}
    elif task == "transformer":
        text = Path(text_file).read_text() if text_file else TEXT
        # Fixed byte alphabet prevents fitting a vocabulary on held-out text.
        vocabulary = list(map(chr, range(128)))
        if any(ord(c) >= 128 for c in text):
            raise ValueError("This small character example requires ASCII text")
        cut1, cut2 = int(0.8 * len(text)), int(0.9 * len(text))
        length = 8 if smoke else context_length
        train = datasets.character_sequences(text[:cut1], vocabulary, length)
        val = datasets.character_sequences(text[cut1:cut2], vocabulary, length)
        test = datasets.character_sequences(text[cut2:], vocabulary, length)
        train = tuple(a[: 8 if smoke else limit] for a in train)
        val, test = (
            tuple(a[: 8 if smoke else eval_limit] for a in val),
            tuple(a[: 8 if smoke else eval_limit] for a in test),
        )
        model = models.TinyLanguageModel(
            len(vocabulary),
            length,
            width=8 if smoke else model_width,
            depth=1 if smoke else model_depth,
            heads=2,
            rngs=rngs,
        )
        metadata = {
            "vocabulary": vocabulary,
            "split_offsets": [cut1, cut2],
            "source": text_file or "original repeated demo corpus; not a benchmark",
        }
    else:
        raise ValueError(f"Unknown task {task}")

    (output / "data.json").write_text(json.dumps(metadata, indent=2))
    inspection.write_json(model, output / "architecture.json")
    (output / "summary.txt").write_text(inspection.summary(model))
    optimizer = optim.Adam(
        model.parameters(),
        lr=3e-4 if task == "transformer" else (1e-3 if task != "gan" else 2e-4),
        betas=(0.5, 0.999) if task == "gan" else (0.9, 0.999),
    )
    if task == "gan":
        d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        fixed_z = nj.random.normal(jax.random.key(seed + 100), (32, latent_size))
    loader = data.DataLoader(
        data.TensorDataset(*train), batch_size=batch_size, shuffle=True, seed=seed
    )
    history, best = [], float("inf")

    @nj.jit
    def evaluate(candidate, x, y):
        prediction = candidate(x)
        if task == "autoencoder":
            return F.mse_loss(prediction["reconstruction"], x)
        if task in ("rnn", "lstm"):
            return F.mse_loss(prediction["prediction"], y)
        return F.cross_entropy(prediction, y)

    def validation_loss(pair):
        losses, counts = [], []
        for x, y in data.DataLoader(data.TensorDataset(*pair), batch_size=batch_size):
            losses.append(float(evaluate(model, x, y)))
            counts.append(len(x))
        return float(np.average(losses, weights=counts))

    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        totals, count = {}, 0
        for x, y in loader.iter_epoch(epoch):
            if task == "gan":
                result = training.gan_step(
                    model,
                    discriminator,
                    optimizer,
                    d_optimizer,
                    x * 2 - 1,
                    runtime,
                    latent_size=latent_size,
                )
            elif task == "cnn":
                result = training.cnn_step(model, optimizer, (x, y), runtime)
            elif task == "autoencoder":
                result = training.autoencoder_step(model, optimizer, x)
            elif task in ("rnn", "lstm"):
                result = training.regression_step(model, optimizer, (x, y))
            else:
                result = training.classifier_step(model, optimizer, (x, y))
            for key in ("loss", "g_loss", "d_loss"):
                if key in result:
                    totals[key] = totals.get(key, 0.0) + float(result[key]) * len(x)
            count += len(x)
        row = {key if task == "gan" else "train_loss": v / count for key, v in totals.items()}
        model.eval()
        if task != "gan":
            row["val_loss"] = validation_loss(val)
            if row["val_loss"] < best:
                best = row["val_loss"]
                checkpoint.save(output / "best.npz", model=model)
        else:
            samples = np.asarray(model(fixed_z)) / 2 + 0.5
            plots.image_grid(
                samples, output / f"samples-{epoch:03}.png", "Fashion-MNIST GAN: fixed noise"
            )
        history.append(row)
        run.log_metrics(epoch=epoch, split="epoch", **row)
        print(task, epoch + 1, row)

    participants = {"model": model, "optimizer": optimizer, "rngs": runtime}
    if task == "gan":
        participants.update(discriminator=discriminator, d_optimizer=d_optimizer)
    checkpoint.save(
        output / "last.npz",
        **participants,
        progress={"next_epoch": epochs},
        data_state=loader.state_dict(),
        metadata=config,
    )
    plots.learning_curve(history, output / "learning.png")
    report = {
        "history": history,
        "smoke": smoke,
        "training_seconds": time.perf_counter() - started,
        "devices": [str(d) + " " + d.device_kind for d in jax.devices()],
        "config": config,
        "train_samples": len(train[0]),
        "validation_samples": len(val[0]),
        "test_samples": len(test[0]),
    }
    if task == "gan":
        np.savez(
            output / "samples.npz", samples=np.asarray(model(fixed_z)), noise=np.asarray(fixed_z)
        )
    if task != "gan":
        checkpoint.restore(output / "best.npz", model=model)
        model.eval()
        report["test_loss"] = validation_loss(test)
        # Bounded prediction artifacts from the selected validation checkpoint.
        batches = []
        for i in range(0, len(test[0]), batch_size):
            value = model(jnp.asarray(test[0][i : i + batch_size]))
            if task in ("rnn", "lstm"):
                value = {"prediction": value["prediction"]}
            batches.append(value)
        prediction = jax.tree.map(lambda *xs: np.concatenate(xs), *batches)
        if task == "autoencoder":
            shown = np.stack([test[0][:8], prediction["reconstruction"][:8]], axis=1).reshape(
                -1, 1, 28, 28
            )
            plots.image_grid(
                shown, output / "reconstructions.png", "Original / reconstruction pairs"
            )
            np.savez(output / "predictions.npz", **prediction, targets=test[0])
        elif task in ("rnn", "lstm"):
            import matplotlib.pyplot as plt

            pred = prediction["prediction"]
            fig, ax = plt.subplots()
            ax.scatter(test[1].ravel(), pred.ravel(), s=12)
            ax.plot([-1, 1], [-1, 1], color="gray")
            ax.set(xlabel="True next value", ylabel="Predicted next value")
            fig.savefig(output / "forecast.png")
            plt.close(fig)
            report["persistence_baseline_mse"] = float(np.mean((test[0][:, -1] - test[1]) ** 2))
            np.savez(output / "predictions.npz", predictions=pred, targets=test[1])
        elif task == "transformer":
            report["perplexity"] = float(np.exp(min(report["test_loss"], 80)))
            tokens = [ord(c) for c in "First Citizen:\n"]
            sample_rng = nj.Rngs(sample=seed + 101)

            @nj.jit
            def sample_next(candidate, context, valid, index, key):
                logits = candidate(context, valid=valid)[0, index] / 0.8
                return nj.random.categorical(key, logits)

            for _ in range(400):
                context = tokens[-model.max_length :]
                size = len(context)
                padded = jnp.asarray([context + [0] * (model.max_length - size)], dtype=jnp.int32)
                valid = jnp.arange(model.max_length)[None] < size
                next_token = sample_next(model, padded, valid, size - 1, sample_rng.next("sample"))
                tokens.append(int(next_token))
            (output / "generation.txt").write_text("".join(vocabulary[i] for i in tokens))
        else:
            probabilities = np.asarray(jax.nn.softmax(prediction, axis=-1))
            report["test_accuracy"] = float(np.mean(probabilities.argmax(-1) == test[1]))
            report["majority_baseline_accuracy"] = float(
                np.mean(test[1] == np.bincount(train[1]).argmax())
            )
            plots.classification(test[1], probabilities, output / "classification.png")
            np.savez(output / "predictions.npz", probabilities=probabilities, targets=test[1])
            if task == "cnn":
                wrong = np.flatnonzero(probabilities.argmax(-1) != test[1])[:16]
                if len(wrong):
                    labels = [f"{test[1][i]}→{probabilities[i].argmax()}" for i in wrong]
                    plots.image_grid(
                        test[0][wrong], output / "mistakes.png", "True → predicted", labels
                    )
            elif "test_sex" in metadata:
                report["subgroups"] = {}
                for group in sorted(set(metadata["test_sex"])):
                    selected = np.array(metadata["test_sex"]) == group
                    report["subgroups"][group] = {
                        "count": int(selected.sum()),
                        "accuracy": float(
                            np.mean(probabilities[selected].argmax(-1) == test[1][selected])
                        ),
                    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", choices=["titanic", "cnn", "rnn", "lstm", "transformer", "autoencoder", "gan"]
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--text-file")
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--model-depth", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny synthetic mechanics check; no dataset download"
    )
    args = vars(parser.parse_args())
    print("Artifacts:", run_experiment(**args))


if __name__ == "__main__":
    main()
