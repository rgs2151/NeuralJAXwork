# Brain MRI gallery

Additional CNN and GAN examples with the approved gallery geometry. The original
MNIST, weather, autoencoder, Fashion-MNIST and transformer assets remain unchanged.

- [Composite animation](gallery.gif): 1248 × 270, five 240px panels, 12px gaps.
- [CNN animation](../cnn_brain_mri/animation.gif): three held-out tumor examples.
- [GAN animation](../gan_brain_mri/animation.gif): three batches of four generated slices.
- [Interactive composite](../assets/neuroscience.html): independent native animation clocks.

The composite uses `../composite/animation.js` directly, so its layout is shared
with the approved version. Typography, scan timings, panel spacing and the GAN's
relative tile gap are preserved. The CNN's final class names use 32px bold Arial
to fit the existing prediction area. No original assets are overwritten.

## CNN: tumor-type classification

An eight-convolution network with four pooling stages processes 128 × 128 images.
It was trained for 40 epochs on GPU 0 with AdamW, mild intensity/translation
augmentation and horizontal flips. The checkpoint was selected using validation
balanced accuracy at epoch 31, then evaluated once on the held-out test split.

| Split | Patients | Slices |
| --- | ---: | ---: |
| Training | 139 | 1,883 |
| Validation | 47 | 536 |
| Test | 47 | 645 |

Test accuracy is **93.33%**, with **92.13% balanced accuracy**. Splits use patient
IDs, not independent random slices. All three labels denote tumor types; this is
not a healthy-versus-tumor classifier. See [full results](cnn-results.json).

The animation shows the first correctly classified test image of each class.
This selection is for presentation; the reported metrics include mistakes.
Input images are resized to 64 × 64 for display, then pooled illustratively in
place through four passes. These are not intermediate CNN feature maps. Final
labels are actual cached predictions. The model sees 128 × 128 inputs.

Source: [Jun Cheng's brain tumor dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427),
3,064 contrast-enhanced T1 MRI slices from 233 patients; **CC BY 4.0**.
Images are intensity-normalized, square-padded and resized. The source includes
meningioma, glioma and pituitary tumors. Cite Cheng et al.,
[Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition](https://doi.org/10.1371/journal.pone.0140381), 2015.

## GAN: brain slice generation

A convolutional GAN maps 128-dimensional latent vectors to 64 × 64 T2-weighted
MRI slices. Its generator uses a 256-channel 4 × 4 projection and four upsampling
stages. Training ran for 150 epochs on GPU 0, using 2,472 slices from 103
participants; 600 slices from 25 other participants were held out.

The source is [MPI-Leipzig Mind-Brain-Body, OpenNeuro ds000221](https://openneuro.org/datasets/ds000221),
whose [dataset metadata](https://github.com/OpenNeuroDatasets/ds000221/blob/master/dataset_description.json)
declares **CC0**. This run uses the first 128 sorted available T2 scan keys and
24 axial slices per scan, between 65% and 85% of its canonical superior axis.
Images are percentile-normalized, square-padded and resized. T2 was selected
because the downloaded MP2RAGE T1 images had distracting background artifacts.
Cite Babayan et al., [A mind-brain-body dataset of MRI, EEG, cognition, emotion,
and peripheral physiology in young and old adults](https://doi.org/10.1038/sdata.2018.308), 2019,
and the OpenNeuro dataset.

The animation uses the first 12 seeded latent vectors, with no visual ranking or
rejection. Every saved frame comes from the generator at that training epoch.
Interpolation between checkpoints is a display effect: a GAN generates its image
in one forward pass, rather than progressively denoising it at inference time.
The first three early frames are omitted, as in the approved renderer.

Outputs have recognizable slice outlines and internal contrast, but remain softer
than real scans. The 32 fixed samples' nearest-training-image pixel RMSE ranges
from 0.062 to 0.081 on [0, 1] intensities. This is a limited diagnostic, not proof
of novelty, anatomical fidelity or a perceptual quality score. See
[full results](gan-results.json). These are research demonstrations, not diagnostic
models or validated synthetic clinical data.

## Reproduce locally

From the repository root, using the package's CUDA and examples dependencies:

```bash
.venv/bin/pip install -r gallery/neuroscience/requirements.txt
.venv/bin/python gallery/neuroscience/prepare.py cnn
.venv/bin/python gallery/neuroscience/prepare.py gan --subjects 128

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python gallery/neuroscience/train.py cnn --epochs 40 --output runs/neuroscience/cnn
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python gallery/neuroscience/train.py gan --epochs 150 --output runs/neuroscience/gan

.venv/bin/python gallery/neuroscience/export.py
JAX_PLATFORMS=cpu .venv/bin/python -m pytest gallery/neuroscience/test_smoke.py
```

Downloads and checkpoints stay under gitignored `data/neuroscience` and
`runs/neuroscience`. Downloads include the source manifest, CNN archive checksums,
and GAN source keys. The trained weights are not bundled into the HTML or GIFs.
Reproduction overwrites the specified run directory; use a different `--output`
for an independent experiment. Export expects the canonical run paths above.

`render_gif.cjs` captures the composite with Playwright and encodes GIFs with
ffmpeg. Set `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` if they are not installed in
the default locations. `geometry.json` records checked panel sizes and title
fonts. `check_layout.cjs` compares panel, heading and plot bounds against the
approved composite at full and half width. `verify_checkpoints.py` reloads both
models and checks their outputs against the cached results. The composite GIF
uses the same eight-frame end blend as the original gallery; individual GIFs
retain their native cycles. Set `GALLERY_PYTHON` for the Pillow environment.
No data downloads, training or notebook execution are added to CI.
