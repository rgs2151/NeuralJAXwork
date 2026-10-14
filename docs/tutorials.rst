Architecture tutorials
======================

These notebook chapters are part of this Sphinx book, not external links to a
separate collection. Each chapter includes its explanation, model summary,
executable Python, generated report, and plots. Use the page download menu or the
source links below to work locally in Jupyter.

Every published chapter runs in offline smoke mode. Set ``SMOKE=False`` and allow
``DOWNLOAD=True`` where needed to train on the named dataset. Keep full training
outside documentation CI; the :doc:`examples` guide provides larger-run commands,
data provenance, and interpretation caveats.

.. toctree::
   :maxdepth: 1
   :numbered:

   ../notebooks/01_titanic_dense
   ../notebooks/02_mnist_cnn
   ../notebooks/03_rnn_forecasting
   ../notebooks/04_lstm_forecasting
   ../notebooks/05_character_transformer
   ../notebooks/06_mnist_autoencoder
   ../notebooks/07_fashion_mnist_gan

Download notebook sources
-------------------------

* :download:`Titanic dense classifier <../notebooks/01_titanic_dense.ipynb>`
* :download:`MNIST convolutional classifier <../notebooks/02_mnist_cnn.ipynb>`
* :download:`RNN forecasting <../notebooks/03_rnn_forecasting.ipynb>`
* :download:`LSTM forecasting <../notebooks/04_lstm_forecasting.ipynb>`
* :download:`Character transformer <../notebooks/05_character_transformer.ipynb>`
* :download:`MNIST autoencoder <../notebooks/06_mnist_autoencoder.ipynb>`
* :download:`Fashion-MNIST GAN <../notebooks/07_fashion_mnist_gan.ipynb>`
