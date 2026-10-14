# Composite gallery

The row contains CNN, RNN, Transformer, Autoencoder, and GAN, in that order. All panels have a 240-pixel overall height, 18-pixel Arial bold titles, and 6-pixel horizontal gaps. Widths follow each plot's proportions; the complete row is approximately 1149×240 pixels. A single outer SVG contains the five mounted SVG renderers. Its width scales responsively while retaining the horizontal arrangement.

`animation.js` mounts the canonical renderers and arranges their output without editing their source files. The RNN legend is reproduced in SVG so the entire row can be captured consistently. The live composite uses the original timings. `window.gallery.seek(milliseconds)` provides deterministic capture.

## Files

- `index.html`: live composite, loading the canonical sources.
- `gallery.gif`: 18-second, 15-fps looping excerpt for the GitHub README.
- `../assets/composite.html`: self-contained live composite.
- `../assets/{cnn,rnn,transformer,autoencoder,gan}.html`: self-contained approved individual animations.
- `../frozen.json`: hashes of the frozen individual sources, checked before packaging.

The GIF preserves the original playback speeds. Its last eight frames blend into its opening frame to soften the excerpt's loop boundary. That blend belongs only to the GIF export; individual assets and the live composite retain their approved behavior. The GIF uses a 96-color palette without dithering to reduce size and avoid speckling.

## Reproduction

Packaging uses Python's standard library. Browser capture additionally needs Node.js, Playwright with Chromium, ffmpeg, and Python with Pillow.

```bash
python3 gallery/composite/package_assets.py
node gallery/composite/render_gif.cjs
python3 gallery/composite/build_preview.py /absolute/output/gallery.html
```

Set `PLAYWRIGHT_MODULE`, `CHROMIUM_PATH`, or `GALLERY_PYTHON` when those dependencies are outside their default locations. Captured frames are temporary and removed after successful encoding. Regeneration never retrains a model or modifies the frozen renderers. Only the current approved standalone bundles and current GIF are retained as reusable assets.
