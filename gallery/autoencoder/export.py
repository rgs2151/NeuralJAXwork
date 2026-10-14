"""Export five class-centroid PCA endpoints per dataset for the autoencoder gallery."""
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import neuraljaxlib as nj
from neuraljaxlib import checkpoint, models

ROOT = Path(__file__).resolve().parents[2]


def project(name, images, latent, labels, ids, classes, checkpoint_path):
    projections, variance, scales = [], [], []
    for values in (images.reshape(len(images), -1), latent):
        pca = PCA(n_components=2, svd_solver='randomized', random_state=0)
        scores = pca.fit_transform(values)
        scale = float(np.sqrt(np.mean(np.sum(scores**2, axis=1))))
        projections.append(scores / scale)
        variance.append(pca.explained_variance_ratio_.tolist())
        scales.append(scale)
    # PCA signs are arbitrary; align signs without rotating the component axes.
    signs = np.where(np.sum(projections[0] * projections[1], axis=0) < 0, -1, 1)
    projections[1] *= signs
    samples, dots = [], []
    rng = np.random.default_rng(42)
    for class_index, (category, label) in enumerate(classes.items()):
        local = np.flatnonzero(labels == category)
        selected = rng.choice(local, size=min(100, len(local)), replace=False)
        dots.extend(dict(classIndex=class_index, testIndex=int(ids[i]),
                         start=projections[0][i].round(6).tolist(),
                         end=projections[1][i].round(6).tolist()) for i in selected)
        pixels = images[local].reshape(len(local), -1)
        representative = int(local[np.argmin(np.sum((pixels-pixels.mean(axis=0))**2, axis=1))])
        if images.shape[1] == 1:
            rgba = np.zeros((*images.shape[2:], 4), dtype=np.uint8)
            rgba[:, :, 3] = np.rint(images[representative, 0] * 255).astype(np.uint8)
            image = Image.fromarray(rgba)
        else:
            image = Image.fromarray(np.rint(images[representative].transpose(1,2,0)*255).astype(np.uint8))
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        samples.append(dict(label=label, testIndex=int(ids[representative]), count=len(local),
                            start=projections[0][local].mean(axis=0).tolist(),
                            end=projections[1][local].mean(axis=0).tolist(),
                            image='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()))
    rng.shuffle(dots)
    return dict(dataset=name, latentDimensions=latent.shape[1], samples=samples, dots=dots,
                checkpointSha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                explainedVariance=variance, rmsScales=scales, latentSigns=signs.tolist())


def main():
    run = ROOT/'runs/gallery/autoencoder'
    with np.load(ROOT/'data/mnist.npz') as source:
        images = source['x_test'][:,None].astype(np.float32)/255
        labels = source['y_test']
    model = models.Autoencoder(latent_size=8, width=128, rngs=nj.Rngs(params=0))
    checkpoint.restore(run/'best.npz',model=model)
    model.eval()
    # Recompute from the checkpoint to avoid device-precision differences in old caches.
    latent = np.concatenate([np.asarray(model.encode(images[i:i+256])) for i in range(0,len(images),256)])
    classes = {i:str(i) for i in [0,1,5,6,9]}
    ids = np.flatnonzero(np.isin(labels,list(classes)))
    batches = [project('MNIST',images[ids],latent[ids],labels[ids],ids,classes,run/'best.npz')]
    for name, classes in [('fashion_mnist',{0:'T-shirt',1:'Trouser',5:'Sandal',8:'Bag',9:'Ankle boot'}),('gtsrb_contrast',{6:'End of 80 km/h limit',12:'Priority road',13:'Yield',14:'Stop',38:'Keep right'})]:
        run = ROOT/'runs/autoencoder-gallery'/name
        with np.load(run/'predictions.npz') as data:
            batches.append(project(name,data['targets'],data['latent'],data['labels'],data['test_ids'],classes,run/'best.npz'))
    result = dict(schema=1,batches=batches,note='Independent input and bottleneck PCA on each five-class test subset; each projection normalized to unit RMS radius. Only PCA signs aligned. Means are not artificially separated. Interpolation is illustrative, not intermediate layer activations.')
    Path(__file__).with_name('samples.js').write_text('window.AE_SAMPLES = '+json.dumps(result,separators=(',',':'))+';\n')
    print('Exported three five-class PCA batches.')

if __name__ == '__main__':
    main()
