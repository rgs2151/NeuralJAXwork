"""Small offline checks for forecast targets, temporal isolation, and updates."""

import jax.numpy as jnp
import numpy as np
import pytest

import neuraljaxlib as nj
from examples.forecast_data import market_windows, weather_windows
from examples.forecasting import ForecastModel, train_step
from neuraljaxlib import optim


def test_weather_horizon_and_training_only_scaling():
    times = np.arange(
        np.datetime64("2022-12-01T00"), np.datetime64("2024-01-05T00"), np.timedelta64(1, "h")
    )
    values = 10 + np.sin(np.arange(len(times)) / 24) + np.arange(len(times)) / 10000
    splits, meta = weather_windows(times, values)
    train, val, test = splits
    assert train["inputs"].shape[1:] == (168, 5)
    for split in splits:
        assert np.all(split["history_times"][:, -1] < split["target_times"][:, 0])
        assert split["targets"].shape[1] == 24
        np.testing.assert_array_equal(split["baseline_daily"], split["history"][:, -24:])
    assert np.all(train["target_times"] < "2023-01-01")
    assert np.all(val["target_times"] >= "2023-01-01")
    assert np.all(val["target_times"] < "2024-01-01")
    assert np.all(test["target_times"] >= "2024-01-01")
    changed = values.copy()
    changed[times >= np.datetime64("2023-01-01")] += 500
    other, other_meta = weather_windows(times, changed)
    assert meta == other_meta
    np.testing.assert_array_equal(train["inputs"], other[0]["inputs"])
    np.testing.assert_array_equal(train["labels"], other[0]["labels"])
    i = np.flatnonzero(times == np.datetime64(test["origins"][0]))[0]
    np.testing.assert_allclose(test["targets"][0], values[i : i + 24])
    with pytest.raises(ValueError, match="hourly grid"):
        weather_windows(np.delete(times, 20), np.delete(values, 20))


def test_market_targets_and_past_only_features():
    times = np.arange(np.datetime64("2022-01-01"), np.datetime64("2024-02-01"))
    times = times[np.is_busday(times)]
    rng = np.random.default_rng(42)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(times))))
    splits, meta = market_windows(times, closes)
    train, val, test = splits
    assert train["inputs"].shape[1:] == (60, 4)
    for split in splits:
        assert np.isfinite(split["inputs"]).all()
        assert np.all(split["history_times"][:, -1] < split["target_times"][:, 0])
    assert np.all(train["target_times"] < "2023-01-01")
    assert np.all(val["target_times"] < "2024-01-01")
    first = test["origins"][0]
    i = np.flatnonzero(times == np.datetime64(first))[0]
    returns = np.diff(np.log(closes))
    expected = np.sqrt(252 * np.mean(returns[i - 1 : i + 4] ** 2))
    np.testing.assert_allclose(test["targets"][0, 0], expected, rtol=1e-6)
    baseline = np.sqrt(252 * np.mean(returns[i - 21 : i - 1] ** 2))
    np.testing.assert_allclose(test["baseline_20d"][0, 0], baseline, rtol=1e-6)
    changed = closes.copy()
    changed[times >= np.datetime64(first)] *= np.linspace(
        2, 3, (times >= np.datetime64(first)).sum()
    )
    other, other_meta = market_windows(times, changed)
    assert meta == other_meta
    np.testing.assert_array_equal(train["inputs"], other[0]["inputs"])
    np.testing.assert_array_equal(train["labels"], other[0]["labels"])
    np.testing.assert_array_equal(test["inputs"][0], other[2]["inputs"][0])
    assert other[2]["targets"][0] != test["targets"][0]


@pytest.mark.parametrize("task,features", [("rnn", 5), ("lstm", 4)])
def test_forecast_baseline_initialization_and_update(task, features):
    model = ForecastModel(task, hidden=2, rngs=nj.Rngs(params=0))
    x = jnp.ones((2, 24, features))
    base = x[:, -24:, 0] if task == "rnn" else x[:, -1, -1:]
    np.testing.assert_array_equal(model(x), base)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    loss, norm = train_step(model, optimizer, x, base + 0.5)
    assert np.isfinite(loss) and np.isfinite(norm)
    assert float(jnp.mean((model(x) - base - 0.5) ** 2)) < float(loss)
