"""Export a fixed chronological weather excerpt for the RNN clock."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "runs/forecast-gallery-v2/rnn/predictions.npz"
with np.load(path, allow_pickle=False) as data:
    dates = data["target_times"][:, 0]
    selected = np.flatnonzero((dates >= "2024-01-01") & (dates < "2024-02-19"))
    assert len(selected) == 49
    times = data["target_times"][selected].ravel()
    assert np.all(np.diff(times.astype("datetime64[h]")) == np.timedelta64(1, "h"))
    observed = data["targets"][selected].ravel()
    forecast = data["predictions"][selected].ravel()
    result = dict(
        schema=1,
        units="°C",
        historyHours=576,
        playbackHours=576,
        sourceSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        times=times.tolist(),
        observed=observed.round(3).tolist(),
        forecast=forecast.round(3).tolist(),
        issuedAt=np.repeat(data["issued_at"][selected], 24).tolist(),
        note="Archived daily 24-hour forecasts aligned to their realized hourly temperatures; not rolling one-hour forecasts. Neuron animation is schematic.",
    )
Path(__file__).with_name("samples.js").write_text(
    "window.RNN_SAMPLES = " + json.dumps(result, separators=(",", ":")) + ";\n"
)
print("Exported 1176 consecutive held-out hours (49 days).")
