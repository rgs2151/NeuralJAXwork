"""Train chronological weather and market forecasts and export evaluation records."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxsaw as js
from jaxsaw import checkpoint, data, inspection, models, nn, optim
from jaxsaw.experiments import Run
from jaxsaw.nn import functional as F

from . import forecast_data, plots


class ForecastModel(nn.Module):
    """Learn a recurrent correction to an available historical baseline."""

    def __init__(self, task, hidden=32, *, rngs):
        super().__init__()
        self.task = task
        self.sequence = models.SequenceRegressor(
            features=5 if task == "rnn" else 4,
            hidden=hidden,
            outputs=24 if task == "rnn" else 1,
            kind=task,
            rngs=rngs,
        )
        self.sequence.head.weight.value = jnp.zeros_like(self.sequence.head.weight.value)
        self.sequence.head.bias.value = jnp.zeros_like(self.sequence.head.bias.value)

    def forward(self, x):
        # Use full float32 products consistently across GPU compilation contexts.
        with jax.default_matmul_precision("highest"):
            correction = self.sequence(x)["prediction"]
        baseline = x[:, -24:, 0] if self.task == "rnn" else x[:, -1, -1:]
        return baseline + correction


@js.jit
def train_step(model, optimizer, x, y):
    loss, grads = js.value_and_grad(lambda candidate: F.mse_loss(candidate(x), y))(model)
    grads, norm = optim.clip_grad_norm(grads, 1.0)
    optimizer.step(grads)
    return loss, norm


@js.jit
def predict(model, x):
    return model(x)


def infer(model, inputs, batch_size):
    return np.concatenate(
        [
            np.asarray(predict(model, jnp.asarray(inputs[i : i + batch_size])))
            for i in range(0, len(inputs), batch_size)
        ]
    )


def decode(values, metadata):
    values = values * metadata["target_scale"] + metadata["target_mean"]
    return np.exp(values) if metadata["target_transform"] == "log" else values


def metrics(targets, predictions, log=False):
    result = {
        "mae": float(np.mean(np.abs(predictions - targets))),
        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
    }
    if log:
        result["log_rmse"] = float(
            np.sqrt(
                np.mean(
                    (np.log(np.maximum(predictions, 1e-6)) - np.log(np.maximum(targets, 1e-6))) ** 2
                )
            )
        )
    return result


def draw_forecasts(task, records, output):
    import matplotlib.pyplot as plt

    pred, true = records["predictions"], records["targets"]
    if task == "rnn":
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
        wanted = ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]
        for ax, date in zip(axes.flat, wanted, strict=True):
            matches = np.flatnonzero(np.char.startswith(records["origins"], date))
            i = int(matches[0]) if len(matches) else 0
            ax.plot(
                np.arange(-48, 0),
                records["history"][i, -48:],
                color="#555555",
                label="Known history",
            )
            ax.plot(np.arange(24), true[i], color="black", label="Actual")
            ax.plot(np.arange(24), pred[i], color="#8B1426", label="RNN forecast")
            ax.plot(
                np.arange(24),
                records["baseline_daily"][i],
                color="#6B8CA6",
                ls=":",
                label="Repeat yesterday",
            )
            ax.axvline(-0.5, color="#888888", lw=1, ls="--")
            ax.set(
                title=str(records["origins"][i])[:10],
                xlabel="Hours from forecast start",
                ylabel="Temperature (°C)",
            )
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Next-day temperature forecasts · New York · held-out seasons")
        fig.tight_layout()
        fig.savefig(output / "forecast.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, label in [
            ("predictions", "RNN"),
            ("baseline_daily", "Repeat yesterday"),
            ("baseline_persistence", "Last temperature"),
        ]:
            ax.plot(np.arange(1, 25), np.mean(np.abs(records[key] - true), axis=0), label=label)
        ax.set(xlabel="Forecast lead (hours)", ylabel="Mean absolute error (°C)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "horizon_errors.png", dpi=150)
        plt.close(fig)
    else:
        dates = records["origins"].astype("datetime64[D]")
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(dates, records["history"][:, -1], color="#333333")
        axes[0].set(
            ylabel="S&P 500 index",
            title="Market history and next-five-session volatility · held-out period",
        )
        axes[1].plot(
            dates, true[:, 0] * 100, color="#888888", alpha=0.8, label="Realized next five sessions"
        )
        axes[1].plot(
            dates,
            records["baseline_20d"][:, 0] * 100,
            color="#6B8CA6",
            ls=":",
            label="Trailing 20 sessions",
        )
        axes[1].plot(dates, pred[:, 0] * 100, color="#8B1426", label="LSTM forecast")
        axes[1].set(ylabel="Annualized RMS return (%)", xlabel="Forecast start date")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "forecast.png", dpi=150)
        plt.close(fig)


def run_forecast(
    task,
    *,
    epochs=80,
    batch_size=64,
    limit=100000,
    seed=0,
    output=None,
    download=False,
    cache="data",
    hidden=32,
    patience=15,
):
    if task not in ("rnn", "lstm") or epochs < 1 or batch_size < 1 or limit < 4:
        raise ValueError("Use rnn/lstm, positive epochs/batch size, and limit>=4")
    splits, metadata = forecast_data.load(task, cache, download)
    train, val, test = splits
    train = {key: value[:limit] for key, value in train.items()}
    output = Path(output or f"runs/{task}-{time.time_ns()}")
    config = dict(
        task=task,
        epochs=epochs,
        batch_size=batch_size,
        limit=limit,
        seed=seed,
        hidden=hidden,
        patience=patience,
        learning_rate=0.001,
        matmul_precision="highest",
    )
    run = Run(output, config)
    (output / "data.json").write_text(json.dumps(metadata, indent=2))
    model = ForecastModel(task, hidden=hidden, rngs=js.Rngs(params=seed))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    inspection.write_json(model, output / "architecture.json")
    (output / "summary.txt").write_text(inspection.summary(model))
    loader = data.DataLoader(
        data.TensorDataset(train["inputs"], train["labels"]),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    history, best, best_epoch = [], float("inf"), 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for x, y in loader.iter_epoch(epoch):
            loss, _ = train_step(model, optimizer, x, y)
            total += float(loss) * len(x)
            count += len(x)
        model.eval()
        val_pred = infer(model, val["inputs"], batch_size)
        val_loss = float(np.mean((val_pred - val["labels"]) ** 2))
        row = {"train_loss": total / count, "val_loss": val_loss}
        if not np.isfinite(list(row.values())).all():
            raise ValueError("Non-finite training/validation loss")
        history.append(row)
        run.log_metrics(epoch=epoch + 1, split="epoch", **row)
        if val_loss < best:
            best, best_epoch = val_loss, epoch + 1
            checkpoint.save(output / "best.npz", model=model)
        print(task, epoch + 1, row, flush=True)
        if epoch + 1 - best_epoch >= patience:
            break
    checkpoint.save(
        output / "last.npz",
        model=model,
        optimizer=optimizer,
        progress={"next_epoch": len(history)},
        data_state=loader.state_dict(),
        metadata=config,
    )
    checkpoint.restore(output / "best.npz", model=model)
    model.eval()
    predictions = decode(infer(model, test["inputs"], batch_size), metadata)
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite decoded test prediction")
    records = dict(
        test,
        predictions=predictions,
        issued_at=test["history_times"][:, -1],
        outcome_available_at=test["target_times"][:, -1],
    )
    np.savez_compressed(output / "predictions.npz", **records)
    comparisons = {"model": metrics(test["targets"], predictions, task == "lstm")}
    for key in test:
        if key.startswith("baseline_"):
            comparisons[key] = metrics(test["targets"], test[key], task == "lstm")
    report = dict(
        config=config,
        history=history,
        selected_epoch=best_epoch,
        training_seconds=time.perf_counter() - started,
        devices=[str(d) + " " + d.device_kind for d in jax.devices()],
        train_samples=len(train["inputs"]),
        validation_samples=len(val["inputs"]),
        test_samples=len(test["inputs"]),
        evaluation=comparisons,
        units=metadata["units"],
        source_sha256=metadata["source_sha256"],
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "visualization.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "task": task,
                "file": "predictions.npz",
                "split": "test",
                "ordering": "chronological; all held-out origins",
                "frames": len(predictions),
                "units": metadata["units"],
                "history_times": "known observations",
                "issued_at": "last known observation",
                "origins": "first forecast target",
                "outcome_available_at": "last target observation; reveal realized outcome only here",
                "target_times": "forecast target timestamps",
                "predictions": "model outputs in physical units",
                "targets": "held-out observations; not model inputs",
                "note": (
                    "Market targets overlap for adjacent origins; metrics are descriptive, not independent trials."
                    if task == "lstm"
                    else "Daily origins predict non-overlapping 24-hour target windows."
                ),
            },
            indent=2,
        )
    )
    plots.learning_curve(history, output / "learning.png")
    draw_forecasts(task, records, output)
    print(json.dumps(comparisons, indent=2), flush=True)
    return output
