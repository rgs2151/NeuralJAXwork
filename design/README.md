# NeuralJAXwork: proposed API design

This folder preserves the original API proposal from 2026-10-14. Code blocks
describe proposed contracts and may include unsupported interfaces. For release
behavior, see the [API guide](../docs/api.md) and
[implementation status](IMPLEMENTATION.md).

## Design goals

Build a JAX-backed framework that feels familiar to a PyTorch user: subclass
`nn.Module`, compose layers in `__init__`, write computation in `forward`, and
control training with an ordinary Python loop. The same module system should
support dense networks, CNNs, RNNs, LSTMs, transformers, autoencoders, and GANs.
Residual connections, shared weights, multiple inputs and outputs, and custom
losses should be ordinary composition, not special model categories.

Use JAX arrays and
automatic differentiation, explicit random streams, and graph-aware transforms.
Autodifferentiation remains the responsibility of JAX.

Make experiments inspectable from the start. Stable parameter paths, structured
outputs, reproducible checkpoints, and an architecture description will later
support schematics, learning curves, confusion matrices, reconstructions, and
sample galleries. Plotting and notebooks remain optional consumers of this data.

## Read in this order

1. [Core module and state contract](01-core.md): inheritance, registration,
   randomness, differentiation, compilation, and serialization.
2. [Training and data](02-training.md): the primary training loop, optimization,
   evaluation, checkpointing, and optional Lightning-inspired orchestration.
3. [Architecture cookbook](03-architectures.md): proposed APIs for every model
   family, including recurrent state, attention masks, and GAN update boundaries.
4. [Inspection and experiment artifacts](04-inspection.md): contracts that make
   future visualization possible without coupling it to training.
5. [Implementation roadmap and decisions](05-roadmap.md): milestones, acceptance
   criteria, proposed source layout, unresolved decisions, and references.

## API responsibilities

| Concept | Responsibility |
| --- | --- |
| `nn.Module` | Composable computation, parameters, buffers, child modules |
| `nn.Parameter` | A trainable JAX array with stable identity |
| `nn.Buffer` | Non-parameter state, such as running statistics |
| `nj.Rngs` | Explicit named random streams and checkpointable counters |
| `nj.value_and_grad` | Differentiate selected parameters; return gradients |
| `optim.Optimizer` | Update parameters and optimizer state from explicit gradients |
| `nj.jit` | Transform graph-aware code into pure JAX computation |
| `data.DataLoader` | Produce batches outside the compiled computation |
| `training.Task` / `Trainer` | Optional experiment orchestration around the same step |
| `inspection` / `Run` | Describe models and persist structured experiment evidence |

Examples use `neuraljaxwork` with the alias `nj`, as adopted in version 0.1.0.
The uppercase prototype is archived under `legacy/`.

## Model and training API

```python
import neuraljaxwork as nj
from neuraljaxwork import nn, optim
from neuraljaxwork.nn import functional as F

class Classifier(nn.Module):
    def __init__(self, in_features, classes, *, rngs):
        super().__init__()
        self.hidden = nn.Linear(in_features, 128, rngs=rngs)
        self.output = nn.Linear(128, classes, rngs=rngs)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

model = Classifier(784, 10, rngs=nj.Rngs(params=42))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

@nj.jit
def train_step(model, optimizer, batch):
    x, y = batch

    def objective(candidate):
        logits = candidate(x)
        return F.cross_entropy(logits, y)

    loss, grads = nj.value_and_grad(objective)(model)
    optimizer.step(grads)
    return {"loss": loss}

model.train()
for batch in train_loader:  # prepared separately; see 02-training.md
    metrics = train_step(model, optimizer, batch)

model.eval()
logits = model(test_images)
```

Model definitions and control flow resemble PyTorch;
gradients are explicit and compilation understands module state. `model.train()`
sets mode, while the loop performs training. There is no required `model.fit()`.

## Scope boundaries

The first release targets correctness on CPU and one accelerator, float32,
supervised learning, and explicit custom loops. Distributed execution, mixed
precision, export formats, pretrained model hubs, a GUI, and automatic architecture
layout are outside the initial scope.

“Any architecture” means extensible composition of supported differentiable
operations, subject to JAX's tracing and shape constraints. It does not mean that
every architecture, arbitrary Python side effect, or dynamic-shape program will
automatically compile. New primitives can be added without changing the base model
abstraction.
