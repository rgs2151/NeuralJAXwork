# Composite Gallery

The row contains CNN, RNN, Autoencoder, GAN and Transformer, in that order.
It mounts the five standalone renderers intact in HTML grid panels, not nested
SVG viewports. Geometry, cached data and playback timing stay in the individual
renderers. RNN retains its original legend instead of a separately redrawn copy.

The design frame is 1248 x 300: five 240px-wide panels with 12px gaps. Each panel
has a 24px heading row, a 240px plot row, and a 24px legend row, with 6px vertical
gaps. Plots retain their native aspect ratios and are centered in the plot row.
Headings use 18px bold Arial with aligned baselines. RNN labels/legend,
transformer words and PCA axis labels use the same 18px design scale. Only the
composite enlarges the standalone PCA labels; its underlying figure is unchanged.

The whole row scales once from its design width. Do not mount it in a 300px
design canvas and enlarge it: that magnifies non-scaling SVG strokes and makes
the CNN scan window and RNN lines excessively thick. The RNN source now uses
`clientWidth` for label sizing, so ancestor transforms cannot inflate "now".

`mountGallery(root)` exposes `width`, `height`, `seek(milliseconds)`, `pause()`,
`play()` and `destroy()`. Playback respects visibility and reduced motion.

## Files

- `index.html`: live row loading the canonical renderers.
- `animation.js`: shared layout and playback orchestration.
- `gallery.gif`: the current 18-second looping GIF for the README and docs landing page.
- `../assets/composite.html`: self-contained corrected row.
- `../assets/{cnn,rnn,autoencoder,gan,transformer}.html`: standalone assets.
- `../frozen.json`: approved individual-source hashes.

The current GIF is captured from this corrected layout at 1248 × 300 pixels,
15 frames per second, with the original playback speeds. Its last eight frames
blend into the opening frame. The README and documentation display the GIF
directly; the interactive bundle remains available separately.

## Reproduction

```bash
python3 gallery/composite/package_assets.py
python3 gallery/composite/build_preview.py /absolute/output/gallery.html
```

The optional `render_gif.cjs` exporter uses the corrected 1248 x 300 viewport.
It takes browser screenshots; run it only when that capture is explicitly wanted.
Packaging the HTML does not require screenshots, training or network access.
