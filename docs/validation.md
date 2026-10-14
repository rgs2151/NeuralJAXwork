# Validation record

Local validation on 2026-10-14 used Python 3.10.20, JAX/JAXlib 0.5.3, Flax 0.10.4,
Optax 0.2.4, NumPy 2.2.6, and CPU execution.

| Check | Result |
| --- | --- |
| Default pytest suite | 14 passed in 9.76 seconds; no downloads |
| Ruff check and formatting | Passed for package, examples, and tests |
| Sphinx with `-W --keep-going` | Tutorial book built with all seven notebooks executed; figures/downloads checked |
| Source distribution and wheel | Built successfully with isolated build environments |
| Wheel import outside checkout | Imported from isolated install target and ran a compiled classifier update |
| Notebook execution | All seven notebooks executed in offline smoke mode |
| Example integration | All seven runners completed tiny offline runs |
| Real dataset plumbing | Titanic, MNIST CNN/autoencoder, and Fashion-MNIST GAN completed one-epoch checks |

Real-data checks used the complete 916-row Titanic training split and a cap of
32 training examples for image tasks, with batch size 16. These runs validate
I/O, preprocessing, training, checkpointing, and plot production. They are not
convergence studies, competitive scores, or evidence of useful GAN sample quality.
No GPU validation was performed locally. The documentation workflow separately
executes the notebooks and publishes successful main-branch builds; its status is
available in GitHub Actions.

Timings are observations on this machine, not enforced performance assertions.
The notebooks' synthetic outputs were executed for validation but are not committed
as if they were real-dataset model results. New full runs should retain their own
configurations, reports, and artifacts.
