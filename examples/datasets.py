"""Explicitly downloaded datasets and deterministic offline teaching tasks."""

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"


def _fetch(name, cache, download, sha256=None):
    path = Path(cache) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path}: pass --download to fetch the dataset explicitly")
        temp = path.with_suffix(".part")
        try:
            with (
                urllib.request.urlopen(BASE + name, timeout=60) as response,
                temp.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"Checksum mismatch: {path}; remove the invalid cached file")
    return path


def images(name="mnist", cache="data", download=False):
    """Return official train/test pairs, NCHW float32 in [0,1], int32 labels."""
    if name == "mnist":
        path = _fetch(
            "mnist.npz",
            cache,
            download,
            "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
        )
        with np.load(path, allow_pickle=False) as f:
            pairs = [(f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])]
    elif name == "fashion_mnist":
        pairs = []
        for prefix in ("train", "t10k"):
            labels = _fetch(prefix + "-labels-idx1-ubyte.gz", Path(cache) / name, download)
            pixels = _fetch(prefix + "-images-idx3-ubyte.gz", Path(cache) / name, download)
            with gzip.open(labels) as f:
                raw = f.read()
            if int.from_bytes(raw[:4], "big") != 2049:
                raise ValueError("Invalid IDX label file")
            y = np.frombuffer(raw, dtype=np.uint8, offset=8)
            if len(y) != int.from_bytes(raw[4:8], "big"):
                raise ValueError("Invalid IDX label count")
            with gzip.open(pixels) as f:
                raw = f.read()
            header = np.frombuffer(raw[:16], dtype=">u4")
            if tuple(header) != (2051, len(y), 28, 28):
                raise ValueError("Invalid IDX image header")
            x = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(len(y), 28, 28)
            pairs.append((x, y))
    else:
        raise ValueError("name must be mnist or fashion_mnist")
    return tuple((x[:, None].astype(np.float32) / 255, y.astype(np.int32)) for x, y in pairs)


def sine_sequences(samples=512, length=24, seed=0):
    """Independent sine trajectories, rather than overlapping train/test windows."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, (samples, 1))
    frequency = rng.uniform(0.08, 0.3, (samples, 1))
    values = np.sin(phase + frequency * np.arange(length + 1)[None]).astype(np.float32)
    return values[:, :-1, None], values[:, -1:]


def character_sequences(text, vocabulary, length=32, stride=32):
    lookup = {c: i for i, c in enumerate(vocabulary)}
    tokens = np.array([lookup[c] for c in text], dtype=np.int32)
    windows = [tokens[i : i + length + 1] for i in range(0, len(tokens) - length, stride)]
    if not windows:
        raise ValueError("Text is too short for the context length")
    windows = np.stack(windows)
    return windows[:, :-1], windows[:, 1:]


def titanic(cache="data", download=False, seed=0):
    """Training-only preprocessing; exclude outcome-leaking boat/body fields."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import fetch_openml
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    path = Path(cache) / "titanic-40945.csv"
    if not path.exists():
        if not download:
            raise FileNotFoundError("Pass --download to fetch OpenML Titanic 40945")
        frame = fetch_openml(data_id=40945, as_frame=True).frame
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    numeric = ["age", "sibsp", "parch", "fare"]
    categorical = ["pclass", "sex", "embarked"]
    x = frame[numeric + categorical].copy()
    y = frame["survived"].astype(np.int32).to_numpy()
    indices = np.arange(len(frame))
    train, holdout = train_test_split(indices, test_size=0.3, random_state=seed, stratify=y)
    val, test = train_test_split(holdout, test_size=0.5, random_state=seed, stratify=y[holdout])
    preprocess = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ]
    )
    preprocess.fit(x.iloc[train])
    pairs = tuple(
        (preprocess.transform(x.iloc[ids]).astype(np.float32), y[ids]) for ids in (train, val, test)
    )
    num = preprocess.named_transformers_["numeric"]
    cat = preprocess.named_transformers_["categorical"]
    metadata = {
        "source": "https://www.openml.org/d/40945",
        "seed": seed,
        "train_ids": train.tolist(),
        "val_ids": val.tolist(),
        "test_ids": test.tolist(),
        "features": preprocess.get_feature_names_out().tolist(),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "medians": num[0].statistics_.tolist(),
        "means": num[1].mean_.tolist(),
        "scales": num[1].scale_.tolist(),
        "categorical_fill": cat[0].statistics_.tolist(),
        "categories": [v.tolist() for v in cat[1].categories_],
        "test_sex": frame.iloc[test]["sex"].tolist(),
    }
    return pairs, metadata
