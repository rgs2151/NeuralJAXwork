# State, checkpoints, and reproducibility

## Save and restore the next update

```python
from jaxsaw import checkpoint

checkpoint.save(
    'runs/experiment/last.npz',
    model=model, optimizer=optimizer, rngs=runtime_rngs,
    progress={'next_epoch': epoch, 'next_update': update},
    data_state=loader.state_dict(),
    metadata={'seed': 42},
)

restored = checkpoint.restore(
    'runs/experiment/last.npz',
    model=model, optimizer=optimizer, rngs=runtime_rngs,
)
loader.load_state_dict(restored.data_state)
```

Construct the same architecture, optimizer configuration, and named RNG streams
before restoring. Save all participants together so the model and optimizer share
the same parameter identities. For GANs include both models and both optimizers.
The NPZ archive includes a JSON manifest, canonical graph paths, variable types,
arrays, and model modes. Typed JAX random keys are stored as integer key data and
restored using the matching initialized key implementation.

No pickle or embedded Python executes on load. Schema, paths, architecture
configuration, optimizer settings, shapes, and dtypes are checked before state
updates. Files are written to a temporary sibling and atomically replaced.
Compatible CPU/backend runs reproduce the next update in tests; bitwise reproduction across devices is not guaranteed. Custom modules must expose configuration
as ordinary public scalar/container fields for meaningful compatibility checks.
Opaque custom static objects are not fully validated by the architecture schema.

`state_dict` is a lighter weights/buffers snapshot for an already constructed
model; it does not include optimizer/RNG state or enforce alias metadata. Full
checkpoints include architecture hierarchy/aliases. Neither file can recover
arbitrary Python model source. Keep model code, preprocessing, and environment
versions alongside experiment records.

## Data-loader cursor

`DataLoader.iter_epoch(epoch)` resumes from its saved cursor when called for the
same epoch; a different epoch resets the cursor and derives its shuffle from
`(seed, epoch)`. The cursor advances before yielding a batch. Save after a completed
optimizer update, not in the middle of processing a yielded batch. To intentionally
repeat a completed epoch, create a new loader or restore a cursor at zero.

Loader restore checks batch size, dataset length, seed, shuffle, and drop-last
settings. It cannot prove that a same-length dataset has identical content: save
split IDs and dataset provenance. The loader is single process. There is no hidden
prefetch queue or distributed sampler to reconstruct.

## Inspection

```python
from jaxsaw import inspection

print(inspection.summary(model))
inspection.write_json(model, 'architecture.json')
shapes = inspection.output_spec(model, example_input)
```

The summary deduplicates parameters. JSON records module ownership, static fields,
variable shape/dtype, and alias references. `output_spec` runs an evaluation clone
and returns shapes/dtypes without changing the live module's mode or buffers.
It does not provide a full computation graph or capture intermediate activations.
