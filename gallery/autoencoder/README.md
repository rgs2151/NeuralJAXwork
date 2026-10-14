# Autoencoder

A compact, approximately 240×237 composition including its title shows 500 colored test-sample dots, with five representative images at class means, moving between two-dimensional input PCA and bottleneck PCA. The heading is `nj.Module.Autoencoder`; small PC1 and PC2 arrows identify the axes. Each transition uses one second of smoothstep easing, with a 0.3-second input hold and a 0.5-second bottleneck hold. Three datasets cycle in order:

- MNIST: 0, 1, 5, 6, and 9, using the existing eight-dimensional autoencoder checkpoint.
- Fashion-MNIST: T-shirt, trouser, sandal, bag, and ankle boot, using a separately trained 16-dimensional autoencoder.
- GTSRB traffic signs: end of the 80 km/h limit, priority road, yield, stop, and keep right, using another 16-dimensional autoencoder.

Open `index.html`, or load `samples.js` and `animation.js` and call `mountAutoencoder(element, AE_SAMPLES)`. The controller exposes `seek(milliseconds)`, `pause()`, `play()`, and `destroy()`. Playback pauses offscreen and in background tabs; reduced-motion mode displays a fixed bottleneck view.

## Interpretation

These are unsupervised reconstruction models, not five-way classifiers. Labels are used only to select the displayed classes and compute class means. Clothing and traffic-sign models train on their five selected classes; the existing MNIST model was trained on all ten digits.

PCA is fitted independently to input pixels and bottleneck features on the same held-out subset. Each projection is divided by its RMS radius so arbitrary feature units do not dominate the comparison. Corresponding component signs are aligned; axes are not rotated. A fixed viewport per dataset uses the pooled endpoint coordinates’ 2nd–98th percentiles, trimming peripheral outliers without changing point positions or scale during the transition.

Each displayed image is the real test image nearest its class's mean in pixel space, placed at that class's PCA centroid. One hundred held-out samples per class are selected deterministically (seed 42), with the same sample identities at both endpoints. Classes have distinct colors; each representative image carries a matching color dot. Drawing order is shuffled to avoid consistently placing one class above the others. The plot clips peripheral dots at its edges; all 500 samples remain in the data. No synthetic displacement or label-dependent separation is added. Positions between endpoints are illustrative interpolation, not actual intermediate activations. The colored clouds expose within-class spread and between-class overlap that class means alone conceal. Greater separation is not guaranteed by an autoencoder. Traffic signs are resized to 32×32 and contrast-normalized before training, PCA, and display. For each RGB image, one shared mean and standard deviation across all pixels and channels maps values to `clip(0.5 + (x - mean) / (6 * max(std, 0.02)), 0, 1)`. This reduces lighting variation while preserving differences between color channels.

## Reproduction

`export.py` reloads the original MNIST checkpoint and computes its bottleneck features. The two new runs are saved under the ignored `runs/autoencoder-gallery` directory. Both use reconstruction MSE, Adam, 25 epochs, a held-out validation split, and the checkpoint with lowest validation loss. Training never runs in documentation CI.

The sign images come from the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html), using the uncorrupted train/test splits distributed by [tanganke on Hugging Face](https://huggingface.co/datasets/tanganke/gtsrb). This mirror provides 26,640 training images and 12,630 test images before class filtering. A random 20% of the selected training images (capped at 2,000) is reserved for validation; it is not a track-grouped validation split. The test partition remains separate. This gallery is an exploratory representation display, not a classifier benchmark.

Download `data/{train,test}-00000-of-00001.parquet` from that mirror to `data/gtsrb-train.parquet` and `data/gtsrb-test.parquet`. The training script requires `pyarrow` in addition to the example dependencies. Fashion-MNIST and MNIST use the existing local cache. The traffic-sign run is cached under `runs/autoencoder-gallery/gtsrb_contrast`. The previous CIFAR-10 animal cache is retained for comparison but is not displayed.

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python gallery/autoencoder/train.py
JAX_PLATFORMS=cpu .venv/bin/python gallery/autoencoder/export.py
python3 gallery/autoencoder/build_preview.py /absolute/output/autoencoder.html
```

`samples.js` includes checkpoint hashes, explained-variance ratios, PCA normalization scales, class counts, representative test indices, and both endpoints. The presentation label does not introduce a new package import path. The additional model is composed from the package's existing modules in `train.py`.
