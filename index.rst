NeuralJAXlib
=============

NeuralJAXlib is a GPU-accelerated, lightweight machine-learning framework in JAX.
PyTorch-style modules, automatic differentiation, JIT compilation, and explicit
training loops provide a common API for dense networks, CNNs, recurrent networks,
transformers, autoencoders, and GANs.

Model structure, gradients, optimizer updates, and random state remain visible
in your code, making it straightforward to inspect and adapt each part of a
training pipeline. The tutorials use this approach to connect neural-network
concepts with working implementations, real datasets, and visual evaluation.

Start with installation and the user guide, follow an architecture tutorial to
train a model, or explore the API tutorials to build your own components.
The API reference documents individual interfaces and their behavior.

.. toctree::
   :maxdepth: 2
   :titlesonly:

   docs/quickstart
   docs/guide
   docs/learning
   docs/api_tutorials
   docs/api_reference
