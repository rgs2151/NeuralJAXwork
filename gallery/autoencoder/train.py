"""Train the additional gallery autoencoders locally on one GPU."""
import argparse
import io
import json
from pathlib import Path
import sys
import pyarrow.parquet as pq
from PIL import Image
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples import datasets
import neuraljaxlib as nj
from neuraljaxlib import nn, models, optim, training, checkpoint

class ImageAutoencoder(nn.Module):
    def __init__(self, shape, *, rngs):
        super().__init__()
        self.shape = shape
        features = int(np.prod(shape))
        self.encoder = models.MLP(features, [256, 128], 16, rngs=rngs)
        self.decoder = models.MLP(16, [128, 256], features, rngs=rngs)

    def forward(self, images):
        z = self.encoder(images.reshape(len(images), -1))
        return dict(latent=z, reconstruction=jax.nn.sigmoid(self.decoder(z)).reshape((-1, *self.shape)))


def animals():
    pairs = []
    for split in ('train', 'test'):
        rows = pq.read_table(ROOT / f'data/cifar10-{split}.parquet').to_pydict()
        labels = np.asarray(rows['label'], dtype=np.int32)
        selected = np.arange(len(labels))
        images = np.stack([np.asarray(Image.open(io.BytesIO(rows['img'][i]['bytes'])).convert('RGB')) for i in selected])
        pairs.append((images.transpose(0, 3, 1, 2).astype(np.float32)/255, labels[selected]))
    return tuple(pairs)


def traffic_signs():
    pairs = []
    for split in ('train', 'test'):
        rows = pq.read_table(ROOT / f'data/gtsrb-{split}.parquet').to_pydict()
        labels = np.asarray(rows['label'], dtype=np.int32)
        images = np.stack([
            np.asarray(Image.open(io.BytesIO(row['bytes'])).convert('RGB').resize((32, 32), Image.Resampling.LANCZOS))
            for row in rows['image']
        ])
        values = images.transpose(0, 3, 1, 2).astype(np.float32)/255
        mean = values.mean(axis=(1,2,3), keepdims=True)
        std = values.std(axis=(1,2,3), keepdims=True)
        values = np.clip(0.5 + (values-mean)/(6*np.maximum(std, 0.02)), 0, 1)
        pairs.append((values, labels))
    return tuple(pairs)


def run(name, classes, pairs):
    out = ROOT / 'runs/autoencoder-gallery' / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'predictions.npz').exists():
        print(name, 'cached; preserving existing run', flush=True)
        return
    (x, y), (test, labels) = pairs
    x = x[np.isin(y, classes)]
    test_ids = np.flatnonzero(np.isin(labels, classes))
    test, labels = test[test_ids], labels[test_ids]
    rng = np.random.default_rng(0)
    x = x[rng.permutation(len(x))]
    validation_count = min(2000, len(x)//5)
    val, x = x[:validation_count], x[validation_count:]
    model = ImageAutoencoder(tuple(x.shape[1:]), rngs=nj.Rngs(params=0))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    @nj.jit
    def evaluate(m, a):
        return m(a)
    history, best = [], float('inf')
    started = time.perf_counter()
    for epoch in range(25):
        model.train()
        losses = []
        for ids in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x)/256))):
            losses.append(float(training.autoencoder_step(model, optimizer, jnp.asarray(x[ids]))['loss']))
        model.eval()
        vl = np.mean([float(jnp.mean((evaluate(model,jnp.asarray(val[i:i+256]))['reconstruction']-val[i:i+256])**2)) for i in range(0,len(val),256)])
        history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), val_loss=float(vl)))
        if vl < best:
            best = vl
            checkpoint.save(out / 'best.npz', model=model)
        if (epoch+1)%5 == 0:
            print(name, history[-1], flush=True)
    checkpoint.restore(out / 'best.npz', model=model)
    model.eval()
    result = [evaluate(model,jnp.asarray(test[i:i+256])) for i in range(0,len(test),256)]
    latent = np.concatenate([np.asarray(r['latent']) for r in result])
    mse = np.mean([float(np.mean((np.asarray(r['reconstruction'])-test[i*256:(i+1)*256])**2)) for i,r in enumerate(result)])
    np.savez_compressed(out/'predictions.npz', targets=test, latent=latent, labels=labels, test_ids=test_ids)
    (out/'report.json').write_text(json.dumps(dict(dataset=name,classes=classes,epochs=25,train_samples=len(x),validation_samples=len(val),test_samples=len(test),test_mse=float(mse),seconds=time.perf_counter()-started,devices=[str(d) for d in jax.devices()],history=history),indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['fashion_mnist', 'gtsrb', 'cifar10_animals'], default=['fashion_mnist', 'gtsrb'])
    args = parser.parse_args()
    if 'fashion_mnist' in args.datasets:
        run('fashion_mnist', [0,1,5,8,9], datasets.images('fashion_mnist',ROOT/'data'))
    if 'gtsrb' in args.datasets:
        run('gtsrb_contrast', [6,12,13,14,38], traffic_signs())
    if 'cifar10_animals' in args.datasets:
        run('cifar10_animals', [2,3,4,5,7], animals())
