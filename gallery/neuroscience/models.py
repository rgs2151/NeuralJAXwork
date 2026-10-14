"""Brain MRI experiment models built from NeuralJAXlib layers."""

import jax
import jax.numpy as jnp

from neuraljaxlib import nn


class BrainCNN(nn.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.blocks = []
        previous = 1
        for width in (24, 48, 96, 128):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(previous, width, 3, padding=1, rngs=rngs),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.Conv2d(width, width, 3, padding=1, rngs=rngs),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2),
                )
            )
            previous = width
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128, rngs=rngs),
            nn.ReLU(),
            nn.Linear(128, 3, rngs=rngs),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class BrainGenerator(nn.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.project = nn.Linear(128, 256 * 4 * 4, rngs=rngs)
        self.norm = nn.BatchNorm2d(256)
        self.layers = []
        previous = 256
        for width in (128, 64, 32):
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(previous, width, 4, stride=2, padding=1, rngs=rngs),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                )
            )
            previous = width
        self.out = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, rngs=rngs)

    def forward(self, z):
        x = jax.nn.relu(self.norm(self.project(z).reshape(-1, 256, 4, 4)))
        for layer in self.layers:
            x = layer(x)
        return jnp.tanh(self.out(x))


class BrainDiscriminator(nn.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.layers = []
        previous = 1
        for width in (32, 64, 128, 256):
            self.layers.append(nn.Conv2d(previous, width, 4, stride=2, padding=1, rngs=rngs))
            previous = width
        self.head = nn.Linear(256 * 4 * 4, 1, rngs=rngs)

    def forward(self, x):
        for layer in self.layers:
            x = jax.nn.leaky_relu(layer(x), 0.2)
        return self.head(x.reshape(len(x), -1))
