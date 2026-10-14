# Transformer

A 240×240 SVG scrolls words continuously beneath `nj.Module.Transformer`. Each new word enters from the right as previous words slide left. The current word is bold and dark red, with the text line close beneath the title. One set of black semicircles below the text shows the mean attention of both heads in transformer layer 3; stronger weights produce thicker arcs.

Words advance every 0.85 seconds, with a 0.45-second eased slide. The viewport starts filled with prior context. Eighty words from the cached generation form a continuous loop, including the transition from the last word back to the first. That boundary has been checked to be pixel-identical before and after wraparound. There is no sentence-clearing reset.

## Attention source

The checkpoint at `runs/gallery/transformer/best.npz` is a character language model with width 128, three blocks, two heads per block, and a 128-character context. The exporter uses the body of its existing `generation.txt`, normalizes whitespace, and repeats that text to supply rolling context across the loop boundary. Each word is replayed with its preceding 128-character window ending at that word's final character.

Characters are grouped into words for readability. Each step uses the query row of the current word's **last character**, summing key weights over each word's characters and following whitespace, then averaging the two heads. This is word-grouped character attention, not a word-tokenized model. Current-word attention is omitted from arcs without renormalizing previous-word weights. The causal mask prevents attention to future characters.

During the slide, arc positions and weights interpolate between successive query rows. At the settled position, their widths match the exported mean attention exactly through `0.8 + 11.2 × weight` SVG units. Arcs to words outside the viewport are omitted, and arcs fade only at the left clipping edge. The text is a replay, not live browser inference; the loop's joining context is illustrative and differs from the original generation context. The presentation label does not introduce a new package API.

## Embedding and reproduction

Open `index.html`, or load `samples.js` and `animation.js`, then call `mountTransformer(element, TRANSFORMER_SAMPLES)`. The controller supports `seek(milliseconds)`, `pause()`, `play()`, and `destroy()`. Playback freezes offscreen and in background tabs. Reduced-motion users see a settled word and static attention arcs.

```bash
JAX_PLATFORMS=cpu .venv/bin/python gallery/transformer/export.py
python3 gallery/transformer/build_preview.py /absolute/output/transformer.html
```

The exporter validates attention normalization, zero future attention, and agreement between instrumented and ordinary logits within float32 tolerance. Data includes the checkpoint hash, selected layer and heads, per-head grouped weights, and their mean. No retraining or network access is needed for playback.
