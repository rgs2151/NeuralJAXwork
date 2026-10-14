# Notebooks

Notebook outputs are generated locally and committed for display in the Sphinx
documentation. CI **never executes notebooks or trains models**. Downloaded data,
checkpoints, and full run directories are gitignored.

API tutorials cover modules/parameters, explicit training and evaluation, data
batching, checkpoint/RNG restoration, and inspection/custom layers. Model examples
cover Titanic, MNIST CNNs, RNN/LSTM forecasting, character transformers, MNIST
reconstruction, and convolutional Fashion-MNIST generation.

Install `.[notebooks]`, select the intended environment as the notebook kernel,
and launch Jupyter from the repository root or this directory. For GPU execution,
set `CUDA_VISIBLE_DEVICES=0` before starting Jupyter. Follow each notebook's run
configuration; full training is not a smoke test. Save outputs after execution.
