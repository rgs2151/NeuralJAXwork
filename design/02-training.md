# Training, data, and experiments

Status: proposed API. The explicit loop is primary; orchestration is optional.

## A complete supervised step

This example uses `DigitCNN` from the architecture cookbook. Constructor RNGs
initialize weights; the separately owned runtime stream supplies dropout.

```python
import neuraljaxwork as nj
from neuraljaxwork import optim, metrics
from neuraljaxwork.nn import functional as F

model = DigitCNN(rngs=nj.Rngs(params=42))
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
runtime_rngs = nj.Rngs(dropout=43)

@nj.jit
def train_step(model, optimizer, batch, rngs):
    images, targets = batch

    def objective(candidate):
        logits = candidate(images, rngs=rngs)
        loss = F.cross_entropy(logits, targets)
        return loss, {"correct": (logits.argmax(-1) == targets).sum()}

    (loss, aux), grads = nj.value_and_grad(objective, has_aux=True)(model)
    grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
    optimizer.step(grads)
    return {"loss": loss, "correct": aux["correct"],
            "count": targets.size, "grad_norm": grad_norm}

@nj.jit
def eval_step(model, batch):
    images, targets = batch
    logits = model(images)  # eval dropout requires no RNG
    return {"loss": F.cross_entropy(logits, targets),
            "correct": (logits.argmax(-1) == targets).sum(),
            "count": targets.size}

for epoch in range(epochs):
    model.train()
    train_metrics = metrics.ClassificationAccumulator()
    for batch in train_loader.iter_epoch(epoch):
        result = train_step(model, optimizer, batch, runtime_rngs)
        train_metrics.update(result)

    model.eval()
    val_metrics = metrics.ClassificationAccumulator()
    for batch in val_loader.iter_epoch(epoch):
        val_metrics.update(eval_step(model, batch))

    run.log_metrics(epoch=epoch, split="train", **train_metrics.compute())
    run.log_metrics(epoch=epoch, split="val", **val_metrics.compute())
```

`ClassificationAccumulator` weights each mean batch loss by `count`, sums correct
predictions and counts, and reports dataset loss and accuracy. For padded/masked
tasks, report the number of valid items, not array capacity. Gradient norm is a
diagnostic aggregated separately, not weighted as a prediction loss. Host-side
logging and aggregation occur outside `nj.jit`; conversion/synchronization is
explicit and can be amortized. Evaluation never updates model buffers or RNGs
unless a caller explicitly opts into stochastic inference.

## Loss contracts

Functions live in `nn.functional`; optional `nn.CrossEntropyLoss`-style modules
hold configuration and call the same functions.

| Function | Contract |
| --- | --- |
| `cross_entropy(logits, targets, class_axis=-1, mask=None, reduction="mean")` | Stable log-softmax; integer target indices; one item per non-class location |
| `binary_cross_entropy_with_logits(logits, targets, reduction="mean")` | Stable logits formulation; floating targets with exactly matching shape |
| `mse_loss(pred, target, reduction="mean")` | Mean over elements, not only the batch dimension |
| `l1_loss(pred, target, reduction="mean")` | Mean absolute error over elements |
| `smooth_l1_loss(pred, target, beta=1.0, reduction="mean")` | Explicit piecewise smooth-L1 definition; distinct from the legacy smooth MAE |

`reduction` is `none`, `sum`, or `mean`. `mean` divides by valid element count.
An all-masked batch produces zero loss and zero gradient using a safe denominator;
its valid count is zero and aggregators skip it. Shape mismatch is an error rather
than accidental broadcasting. Classifier modules return logits; prediction helpers
apply sigmoid or softmax outside the training objective as needed.

## Data contracts

```python
from neuraljaxwork import data

train_data = data.TensorDataset(train_x, train_y)
train_loader = data.DataLoader(
    train_data, batch_size=64, shuffle=True, seed=101, drop_last=True,
)
val_loader = data.DataLoader(
    data.TensorDataset(val_x, val_y), batch_size=128, shuffle=False,
)
```

`Dataset` defines `__len__` and `__getitem__`; samples and batches can be pytrees.
The default collator stacks compatible leaves. A custom collator handles padded
sequences and returns explicit lengths/masks. Numeric batches are converted to
JAX arrays; string IDs remain host metadata and do not enter compiled steps.

`iter_epoch(epoch)` derives a reproducible shuffle from `(seed, epoch)` and supports
saved cursor/permutation state for resumption. Stochastic transforms receive
keys derived from seed, epoch, and sample identity; no untracked global RNGs.
First version uses a single-process loader. Parallel loading, prefetch, and device
placement are extensions with separate reproducibility contracts.

The final smaller batch is valid but may compile separately. Choose `drop_last`,
padding plus masks, or size buckets explicitly. Validation defaults to retaining
all samples. Do not silently discard evaluation data for compilation convenience.

Splits and preprocessing live in experiment code: fit scalers, imputers, and
vocabularies only on training data, then persist them. Dataset download adapters
record origin, license, version/hash, split indices, and class mapping. The core
library never silently downloads a dataset or requires credentials.

## Optimization controls

Initial optimizers: SGD with optional momentum, Adam, and AdamW with decoupled
weight decay. Public optimizer configuration is serializable. All updates use
JAX autodiff; layers do not implement handwritten backward methods.

Parameter groups can select named subsets for distinct learning rates and decay:

```python
optimizer = optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-4},
    {"params": model.head.parameters(), "lr": 1e-3},
], weight_decay=0.01)
```

Groups cannot overlap. Gradient selection uses the same stable parameter
identities. Frozen parameters are omitted before creating groups.

Gradient accumulation adds explicit gradient trees, weighted by valid sample
count, and divides by total count before clipping and stepping. Optimizer counters
advance once per update, not once per microbatch. BatchNorm statistics still update
per forward, so accumulation does not promise large-batch BatchNorm equivalence.
Schedulers specify whether they advance per update or epoch; save their state.
Non-finite diagnostics must identify the failing step and allow an explicit
abort/skip policy; silent updates with invalid gradients are unacceptable.

## Checkpoint and resume

```python
nj.checkpoint.save(
    "runs/mnist/checkpoints/last",
    model=model, optimizer=optimizer, rngs=runtime_rngs,
    progress={"epoch": epoch, "update": update},
    data_state=train_loader.state_dict(),
    metadata={"config": config, "source_revision": revision},
)

restored = nj.checkpoint.restore(
    "runs/mnist/checkpoints/last",
    model=model, optimizer=optimizer, rngs=runtime_rngs,
)
train_loader.load_state_dict(restored.data_state)
```

Save snapshots outside compiled calls at an update boundary with completed device
work; use an atomic manifest commit so partial writes are not valid checkpoints.
Restore validates all participants before committing any of them. Progress and
loader cursor specify the **next** batch/update to execute. Include scheduler,
accumulation, and task state if active. Initial exact-resume support is single
process on a compatible backend; other configurations must disclose limitations.
An inference export omits optimizer and sampler state but retains model config,
weights, class labels, and preprocessing references.

## Optional Lightning-inspired orchestration

`training.Task` owns experiment logic; `nn.Module` owns architecture. The trainer
calls the same standalone train/eval steps shown above, so adopting it does not
change differentiation, model semantics, or numerical behavior.

```python
class ClassificationTask(training.Task):
    def __init__(self, model, *, lr=3e-4, seed=43):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.rngs = nj.Rngs(dropout=seed)

    def training_step(self, batch):
        return train_step(self.model, self.optimizer, batch, self.rngs)

    def validation_step(self, batch):
        return eval_step(self.model, batch)

task = ClassificationTask(model)
trainer = training.Trainer(
    max_epochs=20,
    callbacks=[training.ModelCheckpoint(monitor="val/loss", mode="min")],
    run=run,
)
trainer.fit(task, train_loader=train_loader, val_loader=val_loader)
```

In this proposal, `training_step` performs the update and returns metrics; the
trainer **never automatically differentiates or steps an optimizer**. This differs
deliberately from Lightning's automatic optimization. GANs use this same manual
contract. Task registers its models, optimizers, RNGs, and other checkpointable
state through typed fields inspected at construction; unregistered state causes
a checkpoint validation error.

Trainer handles mode changes for registered models, epochs, validation, weighted
metric reducers, logging, callbacks, and resume. A task can override phase mode
policy for adversarial or frozen models. Custom metrics declare reducers; the
trainer must not blindly average arbitrary scalars. Callbacks run on the host
after steps and cannot change compiled optimizer math implicitly. Early stopping
and best-checkpoint selection use validation data; final test evaluation stays a
separate explicit action. The trainer comes after the manual loop is stable.
