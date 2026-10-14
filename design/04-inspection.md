# Inspection, visualization, and showcase notebooks

Status: proposed contracts. Build the metadata foundations early; plotting and
notebooks follow working training. Models must work without notebook or plotting
dependencies installed.

## Three views of an architecture

| View | What it describes | Limitation |
| --- | --- | --- |
| Module hierarchy | Named ownership, layer configs, parameter/buffer counts, aliases | Does not show every computation in `forward` |
| Execution trace | Calls, tensor shapes, branches, residual edges, reused modules for example inputs | Describes traced paths/signatures, not every possible execution |
| Model specification | Registered type IDs, constructor config, schema versions, child and alias structure | Custom Python code must remain available |

These are separate schemas. A tree is useful for summaries; it is not sufficient
for a residual network schematic. Weights alone do not reconstruct Python control
flow. Inspection must state its scope and disclose opaque or untraced regions.

## Proposed public API

```python
from neuraljaxwork import inspection

summary = inspection.summary(model)
print(summary)
spec = inspection.model_spec(model)
spec.write_json("runs/mnist/model.json")

trace = inspection.trace(
    model, example_inputs=(example_images,), mode="eval",
)
trace.write_json("runs/mnist/architecture-trace.json")

# Later optional rendering layer:
# visualization.architecture(trace).save("architecture.svg")
```

Inspection operates on a snapshot, with a separate RNG snapshot when required.
It must not alter training mode, buffers, parameters, runtime RNG counters, or
optimizer state on the live model. Trace validates this noninterference contract.
Mode, input shapes/dtypes, source revision, and trace coverage are recorded.
Array values and dataset examples are excluded from architecture JSON by default.

A trace node records schema version, node ID, canonical module path, invocation
index, registered type, config, input/output pytree shape and dtype specifications,
parameter references, and buffer references. Edges connect specific output and
input ports. Reusing a layer produces separate invocation nodes pointing to the
same parameter identity. Parameter totals deduplicate shared weights.

A later tracing backend combines module boundaries with JAX computation metadata
to represent operations such as residual addition and concatenation. Until that
backend exists, module-call traces must label unobserved functional operations;
they must not pretend to be complete computation graphs. Recurrent scans and
conditional regions should remain structured nodes with nested bodies instead
of thousands of unrolled boxes. No promise of universal Python source recovery.

## Intermediates and model diagnostics

```python
outputs, captured = inspection.capture(
    model, example_images,
    paths=["conv1", "conv2"], mode="eval", max_bytes=8_000_000,
)
```

Capture is opt-in, bounded, and snapshot-based. Repeated calls to a shared module
are keyed by invocation index. Under JIT, selected intermediates are explicit
auxiliary outputs, not Python hooks that append tensors during tracing. Requested
paths and capture settings are static compilation configuration. Failure to meet
the memory budget raises an informative error or uses an explicitly selected
sampling policy; it never silently captures the entire training graph.

Initial diagnostics: parameter counts, shapes, dtypes, gradient/weight norms,
non-finite checks, and named outputs. Later consumers may render activation maps,
attention maps, latent embeddings, or attribution. These outputs are descriptive;
attention weights and latent projections are not automatically causal explanations.

## Experiment record

```python
from neuraljaxwork.experiments import Run

run = Run("runs/mnist/cnn-seed42", config=config)
run.log_metrics(step=100, split="train", loss=0.18, accuracy=0.95)
run.log_artifact("architecture", "runs/mnist/architecture-trace.json")
```

The initial implementation is local files, with no mandatory external service.
Stable records should make later dashboards interchangeable consumers.

```text
runs/<experiment>/<run-id>/
  manifest.json             # schema, config, versions, source revision, seeds
  model.json                # ModelSpec and parameter alias map
  architecture-trace.json   # optional shape-specialized execution trace
  data.json                 # dataset versions, splits, class/feature mappings
  metrics.jsonl             # step, epoch, split, name, value, count, wall time
  predictions/              # optional IDs, targets, scores, split provenance
  checkpoints/              # atomic snapshots and checkpoint manifests
  artifacts/                # plots, reconstructions, sample galleries, reports
```

Metrics distinguish update step from microbatch, epoch, and elapsed time.
Use stable names such as `train/loss`, `val/accuracy`, `gan/g_loss`, and
`gan/d_loss`, with numerator/denominator metadata where appropriate. Artifact
records include type, relative path, generating checkpoint, selection policy,
and relevant sample IDs. Prediction storage is opt-in and bounded. Embedding
projections record method, seed, and input sample selection.

A failed or interrupted run is labeled as such. “Best” means a named validation
metric and direction; final test results are identified separately. A plot must
be regenerable from saved records rather than a hidden notebook variable.

## Showcase notebook roadmap

Every notebook follows the same narrative: question and baseline, provenance and
splits, preprocessing, model definition, architecture summary, explicit training,
validation, selected-checkpoint evaluation, visual diagnostics, limitations, and
artifact export. Provide a short CPU smoke configuration and a longer recommended
configuration, with hardware and runtime reported after measurement.

| Notebook | Model and objective | Evidence and later visualizations |
| --- | --- | --- |
| `01_xor.ipynb` | Tiny MLP, binary logits loss | Loss curve, predictions, decision boundary |
| `02_titanic.ipynb` | Tabular MLP; compare simple baseline | Stratified splits, accuracy/AUROC, confusion matrix, calibration, error groups |
| `03_mnist_dense.ipynb` | Dense multiclass classifier | Learning curves, confusion matrix, correct/incorrect examples |
| `04_mnist_cnn.ipynb` | CNN on same MNIST split | Dense/CNN comparison, parameter counts, filters and activation samples |
| `05_cats_dogs.ipynb` | Small CNN binary classifier | Held-out accuracy/AUROC, class examples, errors, augmentation previews |
| `06_sequence_rnn_lstm.ipynb` | RNN/LSTM on controlled sequence prediction | Same splits and task, learning curves, forecast/target overlays, long-context behavior |
| `07_tiny_transformer.ipynb` | Character next-token LM | Validation cross-entropy/perplexity, fixed-prompt generations, optional attention views |
| `08_mnist_autoencoder.ipynb` | Dense then convolutional reconstruction | Input/reconstruction pairs, reconstruction error, latent projections/interpolations |
| `09_mnist_gan.ipynb` | Generator/discriminator pair | Both losses, fixed-noise grids over time, diversity and failure examples |

Dataset choices, downloads, licenses, and access instructions are finalized when
each notebook is implemented. Titanic's competition test set may lack public
labels: score a labeled holdout and do not invent test accuracy. For cats/dogs,
avoid duplicate or related-image leakage across splits. Token vocabularies and
normalizers are fitted only on training data. GAN image quality must not be
inferred from declining loss; any later quantitative metric needs a documented
feature extractor, sample count, preprocessing, and baseline.

Notebook code imports reusable experiment functions from `examples/`; it does not
contain the only implementation of a training loop. Saved plots include units,
split, checkpoint, and configuration context. Gallery selection is reproducible,
with both successes and failures represented. Executed notebook outputs are
generated only after real runs; the design provides no fabricated scores.
