"""Chronological, past-only forecasting datasets and reproducible source records."""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive?latitude=40.7128&longitude=-74.0060"
    "&start_date=2017-01-01&end_date=2025-12-31&hourly=temperature_2m&models=era5&timezone=UTC"
)
MARKET_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2016-10-15&coed=2025-12-31"
)


def fetch(name, url, cache, download):
    path = Path(cache) / name
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path}: pass --download to fetch the dataset")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                temp.write_bytes(response.read())
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return path


def _validate(times, values):
    if len(times) != len(values) or not np.isfinite(values).all():
        raise ValueError("Series must have matching timestamps and finite values")
    if np.any(np.diff(times) <= np.timedelta64(0, "s")):
        raise ValueError("Timestamps must be strictly increasing")


def _partition(times, origins, horizon):
    start, end = times[origins], times[origins + horizon - 1]
    cut1, cut2 = np.datetime64("2023-01-01"), np.datetime64("2024-01-01")
    masks = (end < cut1, (start >= cut1) & (end < cut2), start >= cut2)
    if not all(mask.any() for mask in masks):
        raise ValueError("Data must cover training, 2023 validation, and 2024+ test periods")
    return masks


def weather_windows(times, temperature):
    """Seven days of hourly history predict 24 hours, at daily UTC origins."""
    times, temperature = np.asarray(times, dtype="datetime64[h]"), np.asarray(temperature)
    _validate(times, temperature)
    if not np.all(np.diff(times) == np.timedelta64(1, "h")):
        raise ValueError("Weather data must have an unbroken hourly grid")
    past = temperature[times < np.datetime64("2023-01-01")]
    mean, scale = float(past.mean()), float(past.std())
    if scale <= 0:
        raise ValueError("Training temperature has zero variance")
    hour = (times - times.astype("datetime64[D]")).astype(int)
    day = (times.astype("datetime64[D]") - times.astype("datetime64[Y]")).astype(int)
    features = np.stack(
        [
            (temperature - mean) / scale,
            np.sin(hour * 2 * np.pi / 24),
            np.cos(hour * 2 * np.pi / 24),
            np.sin(day * 2 * np.pi / 365.25),
            np.cos(day * 2 * np.pi / 365.25),
        ],
        axis=-1,
    ).astype(np.float32)
    origins = np.arange(168, len(times) - 24 + 1, 24)
    all_data = {
        "inputs": np.stack([features[i - 168 : i] for i in origins]),
        "targets": np.stack([temperature[i : i + 24] for i in origins]).astype(np.float32),
        "history": np.stack([temperature[i - 168 : i] for i in origins]).astype(np.float32),
        "history_times": np.stack([times[i - 168 : i] for i in origins]).astype(str),
        "target_times": np.stack([times[i : i + 24] for i in origins]).astype(str),
        "origins": times[origins].astype(str),
    }
    all_data["baseline_daily"] = all_data["history"][:, -24:]
    all_data["baseline_persistence"] = np.repeat(all_data["history"][:, -1:], 24, axis=1)
    all_data["labels"] = ((all_data["targets"] - mean) / scale).astype(np.float32)
    splits = [{k: v[m] for k, v in all_data.items()} for m in _partition(times, origins, 24)]
    return splits, {
        "target_mean": mean,
        "target_scale": scale,
        "target_transform": "identity",
        "features": ["temperature", "sin_hour", "cos_hour", "sin_day", "cos_day"],
        "lookback": 168,
        "horizon": 24,
        "stride": 24,
        "units": "degrees Celsius",
    }


def market_windows(times, closes):
    """60 known returns predict annualized RMS return over five future sessions."""
    times, closes = np.asarray(times, dtype="datetime64[D]"), np.asarray(closes)
    _validate(times, closes)
    if np.any(closes <= 0):
        raise ValueError("Market closing prices must be positive")
    returns = np.diff(np.log(closes))
    times, closes = times[1:], closes[1:]
    # Trailing windows end at the current observation, including no future returns.
    vol5 = np.full(len(returns), np.nan)
    vol20 = np.full(len(returns), np.nan)
    for width, dest in [(5, vol5), (20, vol20)]:
        for i in range(width - 1, len(returns)):
            dest[i] = np.sqrt(252 * np.mean(returns[i - width + 1 : i + 1] ** 2))
    origins = np.arange(79, len(returns) - 5 + 1)
    masks = _partition(times, origins, 5)
    targets = np.array([np.sqrt(252 * np.mean(returns[i : i + 5] ** 2)) for i in origins])[:, None]
    log_targets = np.log(np.maximum(targets, 1e-6))
    mean, scale = float(log_targets[masks[0]].mean()), float(log_targets[masks[0]].std())
    if scale <= 0:
        raise ValueError("Training volatility has zero variance")
    features = np.stack(
        [
            returns / 0.01,
            np.abs(returns) / 0.01,
            (np.log(np.maximum(vol5, 1e-6)) - mean) / scale,
            (np.log(np.maximum(vol20, 1e-6)) - mean) / scale,
        ],
        axis=-1,
    )
    all_data = {
        "inputs": np.stack([features[i - 60 : i] for i in origins]).astype(np.float32),
        "targets": targets.astype(np.float32),
        "labels": ((log_targets - mean) / scale).astype(np.float32),
        "history": np.stack([closes[i - 60 : i] for i in origins]).astype(np.float32),
        "history_returns": np.stack([returns[i - 60 : i] for i in origins]).astype(np.float32),
        "history_times": np.stack([times[i - 60 : i] for i in origins]).astype(str),
        "target_times": np.stack([times[i : i + 5] for i in origins]).astype(str),
        "origins": times[origins].astype(str),
        "baseline_5d": vol5[origins - 1, None].astype(np.float32),
        "baseline_20d": vol20[origins - 1, None].astype(np.float32),
        "future_prices": np.stack([closes[i : i + 5] for i in origins]).astype(np.float32),
    }
    splits = [{k: v[m] for k, v in all_data.items()} for m in masks]
    return splits, {
        "target_mean": mean,
        "target_scale": scale,
        "target_transform": "log",
        "features": [
            "return / 0.01",
            "absolute return / 0.01",
            "log_volatility_5d",
            "log_volatility_20d",
        ],
        "lookback": 60,
        "horizon": 5,
        "stride": 1,
        "units": "annualized volatility (fraction)",
        "annualization": 252,
        "definition": "sqrt(252 * mean(next_5_log_returns**2))",
    }


def load(task, cache="data", download=False):
    if task == "rnn":
        path = fetch("nyc-era5-2017-2025.json", WEATHER_URL, cache, download)
        raw = json.loads(path.read_text())
        splits, metadata = weather_windows(raw["hourly"]["time"], raw["hourly"]["temperature_2m"])
        metadata.update(
            source=WEATHER_URL,
            dataset="NYC ERA5 hourly temperature via Open-Meteo",
            location={k: raw[k] for k in ("latitude", "longitude", "elevation")},
            attribution="ERA5 / Copernicus Climate Change Service, distributed by Open-Meteo (CC BY 4.0)",
        )
    elif task == "lstm":
        import pandas as pd

        path = fetch("sp500-2016-2025.csv", MARKET_URL, cache, download)
        raw = pd.read_csv(path)
        raw["SP500"] = pd.to_numeric(raw["SP500"], errors="coerce")
        raw = raw.dropna(subset=["SP500"])
        raw = raw[
            (raw["observation_date"] >= "2016-10-15") & (raw["observation_date"] <= "2025-12-31")
        ]
        splits, metadata = market_windows(raw["observation_date"], raw["SP500"])
        metadata.update(
            source=MARKET_URL,
            dataset="S&P 500 daily closing index via FRED",
            attribution="S&P Dow Jones Indices LLC; retrieved from FRED, Federal Reserve Bank of St. Louis",
            source_terms="https://fred.stlouisfed.org/series/SP500",
        )
    else:
        raise ValueError("Forecast task must be rnn or lstm")
    metadata.update(
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_file=path.name,
        split_policy="targets before 2023 / during 2023 / during 2024–2025",
        boundary_policy="targets crossing a split boundary excluded; prior history may cross it",
    )
    for name, split in zip(("train", "validation", "test"), splits, strict=True):
        metadata[name] = {
            "count": len(split["origins"]),
            "first_origin": split["origins"][0],
            "last_origin": split["origins"][-1],
        }
    return splits, metadata
