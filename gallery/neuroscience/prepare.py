"""Download and prepare participant-separated brain MRI datasets locally."""

import argparse
import hashlib
import io
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path("data/neuroscience")
ROOT.mkdir(parents=True, exist_ok=True)


def download(url, path, md5=None):
    if not path.exists():
        temp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, temp)
        temp.replace(path)
    if md5 and hashlib.md5(path.read_bytes()).hexdigest() != md5:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def normalize(image, size):
    image = np.maximum(image, 0)
    hi = np.percentile(image[image > 0], 99.5)
    image = np.clip(image / max(hi, 1e-8), 0, 1)
    h, w = image.shape
    pad = max(h, w)
    canvas = np.zeros((pad, pad), np.float32)
    canvas[(pad - h) // 2 : (pad - h) // 2 + h, (pad - w) // 2 : (pad - w) // 2 + w] = image
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.Resampling.BILINEAR))


def cheng():
    manifest = json.load(urllib.request.urlopen("https://api.figshare.com/v2/articles/1512427"))
    (ROOT / "cheng-source.json").write_text(json.dumps(manifest, indent=2))
    files = [f for f in manifest["files"] if f["name"].endswith(".zip")]
    with ThreadPoolExecutor(4) as pool:
        archives = list(
            pool.map(
                lambda f: download(f["download_url"], ROOT / f["name"], f["computed_md5"]), files
            )
        )
    images, labels, patients, ids = [], [], [], []
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for name in sorted(z.namelist()):
                if not name.endswith(".mat"):
                    continue
                with h5py.File(io.BytesIO(z.read(name))) as f:
                    d = f["cjdata"]
                    images.append(normalize(d["image"][:].T, 128))
                    labels.append(int(d["label"][0, 0]) - 1)
                    patients.append("".join(chr(int(c)) for c in d["PID"][:].ravel()))
                    ids.append(Path(name).stem)
    x, y, groups = np.array(images)[:, None], np.array(labels), np.array(patients)
    fold = np.zeros(len(y), int)
    for k, (_, indices) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups)
    ):
        fold[indices] = k
    np.savez_compressed(ROOT / "cnn.npz", images=x, labels=y, patients=groups, ids=ids, fold=fold)
    print("CNN prepared", x.shape, len(set(groups)), flush=True)


def lemon(limit):
    # Public OpenNeuro snapshot; source metadata declares CC0.
    prefix = "ds000221/"
    url = "https://s3.amazonaws.com/openneuro.org?list-type=2&prefix=" + prefix + "&max-keys=1000"
    keys = []
    while True:
        tree = ET.fromstring(urllib.request.urlopen(url).read())
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys.extend(
            n.text for n in tree.findall("s:Contents/s:Key", ns) if n.text.endswith("_T2w.nii.gz")
        )
        token = tree.find("s:NextContinuationToken", ns)
        if token is None:
            break
        from urllib.parse import quote

        url = (
            "https://s3.amazonaws.com/openneuro.org?list-type=2&prefix="
            + prefix
            + "&max-keys=1000&continuation-token="
            + quote(token.text, safe="")
        )
    keys = sorted(keys)[:limit]
    assert len(keys) >= 20, keys
    (ROOT / "lemon-source.json").write_text(
        json.dumps(
            {
                "keys": keys,
                "license": "CC0",
                "modality": "T2w",
                "dataset": "https://openneuro.org/datasets/ds000221",
            },
            indent=2,
        )
    )

    def process(key):
        path = download("https://s3.amazonaws.com/openneuro.org/" + key, ROOT / key.split("/")[-1])
        volume = nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
        # Central axial slab, uniformly sampled; no duplicate slices across splits.
        slices = [
            normalize(np.rot90(volume[:, :, k]), 64)
            for k in np.linspace(
                int(volume.shape[2] * 0.65), int(volume.shape[2] * 0.85), 24, dtype=int
            )
        ]
        print("MRI", path.name, flush=True)
        return np.array(slices), key.split("/")[1]

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(process, keys))
    images = np.concatenate([r[0] for r in results])[:, None]
    patients = np.repeat([r[1] for r in results], 24)
    subjects = np.unique(patients)
    np.random.default_rng(42).shuffle(subjects)
    test = np.isin(patients, subjects[: max(4, len(subjects) // 5)])
    np.savez_compressed(ROOT / "gan.npz", images=images, patients=patients, test=test)
    print("GAN prepared", images.shape, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["cnn", "gan"])
    p.add_argument("--subjects", type=int, default=96)
    a = p.parse_args()
    cheng() if a.task == "cnn" else lemon(a.subjects)
