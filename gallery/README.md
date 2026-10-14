# Gallery

Five approved SVG animations, implemented in dependency-free JavaScript. The canonical individual sources are frozen; `frozen.json` records their SHA-256 hashes. Composite layout and exports live separately under `composite/`.

Open `assets/composite.html` for the complete row, or reuse any of the self-contained `assets/{cnn,rnn,transformer,autoencoder,gan}.html` files. These are packaged copies of the same final versions, not alternative designs. The GitHub README uses `composite/gallery.gif`.

See [composite export instructions](composite/README.md) for layout, playback, and reproduction.

- [CNN](cnn/index.html): convolution and pooling.
- [RNN](rnn/README.md): a weather forecasting clock.
- [GAN](gan/README.md): Fashion-MNIST training progression.
- [Transformer](transformer/README.md): continuous text with mean causal attention.
- [Autoencoder](autoencoder/README.md): class-centroid PCA transitions for digits, clothing, and traffic signs.

## CNN

Open `cnn/index.html` in a browser. The default square is 240×240 pixels, with the centered Arial heading `nj.Module.CNN` above it. This is a presentation label, not a currently available import path; the trained implementation is `neuraljaxlib.models.DigitCNN`. It can scale from 200 to 300 pixels by changing the container width.

A bold red window scans the handwritten digit continuously, alternating direction between rows. Each completed 2×2 patch is replaced **in the same place** by one larger pixel containing its maximum value. Four passes reduce the padded image from 32×32 to 16×16 to 8×8 to 4×4 to 2×2 while maintaining the square's physical size. The four scans take 5.6, 3.2, 1.8, and 1.2 seconds, with a 0.35-second hold after each pass. Finer grids receive more viewing time, while the final four pooling steps remain readable. The predicted digit flashes in bold white Arial on black for 0.6 seconds, without an outline, fade, or zoom, then the next image appears immediately. A 14.5-second cycle repeats for digits 7, 3, and 8.

This is an illustrative pooling sequence applied to the input image, not a trace of the trained network's intermediate activations. The final predictions come from the existing cached CNN, which performs convolution before each of its four pools. The four-stage model was retrained locally on GPU 0; its checkpoint is in `runs/gallery-four-layer/cnn`. The old two-stage checkpoint is not compatible with this architecture.

`cnn/samples.js` contains the source images, predictions, activation exports, checkpoint hash, and test indices. Regenerate from the existing local cache without training:

```bash
JAX_PLATFORMS=cpu .venv/bin/python gallery/cnn/export.py
```

To embed, load `samples.js` and `animation.js`, then call `mountCNN(element, CNN_SAMPLES)`. The controller exposes `pause()`, `play()`, `seek(milliseconds)`, and `destroy()`. Instances are independent. Reduced-motion users see the final numeral; playback freezes offscreen or in a hidden tab.

Create a self-contained chat preview from the same files:

```bash
python3 gallery/cnn/build_preview.py /absolute/output/cnn.html
```
