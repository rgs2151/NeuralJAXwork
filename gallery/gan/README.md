# Fashion-MNIST GAN

Open `index.html` for a 240×240 SVG animation, with `nj.Module.GAN` above it. Twelve fixed latent inputs cycle in three 2×2 batches through the generator's training progression. Four samples develop together in each batch. Playback starts at epoch 3, skipping the initial gray frames, and runs at twice the original speed. Each batch morphs through epochs 3–30 in approximately 1.53 seconds, then immediately switches to the next batch without a hold. The three-batch loop takes 4.59 seconds.

This shows **training progress**, not iterative denoising during inference. A trained GAN generates each output in one forward pass. The selected examples are fixed-noise grid indices 2, 4, 7, 12, 16, 17, 18, 20, 21, 22, 28, and 31, chosen for visual variety rather than a quantitative quality claim.

## Sources

The existing run is `runs/gallery/gan`. Its epoch plots preserve the same latent batch throughout training. The exporter reconstructs the plot layout and samples the 28×28 pixel centers for intermediate frames; these are display approximations because only rendered grids were cached. Registration is checked against the final raw arrays (mean absolute grayscale difference 4.31/255). Final frames use `samples.npz` directly. The initial frame is reconstructed using the same generator initialization seed and cached latent batch, with no training.

Frames blend linearly during playback. All final displayed pixels match the exported raw samples. `samples.js` records frame indices, source hash, and the recovery error.

## Size and embedding

The animation reuses 3,136 SVG rectangles and updates changed pixels at up to 30 frames per second. JavaScript plus all twelve sequences totals approximately **393 kB uncompressed / 204 kB gzip**. There are no framework, movie, image, or network dependencies during playback. The checked-in data uses base64-encoded grayscale bytes; it does not create a separate SVG for each frame.

Load `samples.js` and `animation.js`, then call `mountGAN(element, GAN_SAMPLES)`. The controller provides `seek(milliseconds)`, `pause()`, `play()`, and `destroy()`. Playback pauses offscreen and in background tabs. Reduced-motion users see a finished batch.

Regenerate from local cached artifacts:

```bash
JAX_PLATFORMS=cpu .venv/bin/python gallery/gan/export.py
python3 gallery/gan/build_preview.py /absolute/output/gan-fashion.html
```

`nj.Module.GAN` is a presentation label. The trained implementation is `neuraljaxlib.models.FashionGenerator`.
