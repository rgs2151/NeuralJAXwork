# Notebooks

Notebook outputs are generated locally and committed for display in the Sphinx
documentation. CI renders saved outputs without executing cells. Downloaded data,
checkpoints, and full run directories are gitignored.

API tutorials cover modules/parameters, explicit training and evaluation, data
batching, checkpoint/RNG restoration, and inspection/custom layers. Architecture tutorials
cover MLP Titanic classification, CNN MNIST classification, RNN temperature
forecasting, LSTM market volatility, transformer language modeling, autoencoder
MNIST reconstruction, and GAN Fashion-MNIST generation.

Install `.[notebooks]`, select the intended environment as the notebook kernel,
and launch Jupyter from the repository root or this directory. For GPU execution,
set `CUDA_VISIBLE_DEVICES=0` before starting Jupyter. Follow each notebook's run
configuration and save outputs after execution.
