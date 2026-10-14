"""Small reference architectures. These are teaching models, not pretrained weights."""

import jax.numpy as jnp

from . import nn
from .nn import functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features, *, rngs):
        super().__init__()
        widths = [in_features, *hidden_sizes]
        self.hidden = nn.ModuleList(
            [nn.Linear(a, b, rngs=rngs) for a, b in zip(widths[:-1], widths[1:], strict=True)]
        )
        self.head = nn.Linear(widths[-1], out_features, rngs=rngs)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.head(x)


class DigitCNN(nn.Module):
    """Four convolution/pooling stages for MNIST NCHW images.

    Zero padding expands 28x28 to 32x32 before four reductions to 16, 8,
    4, and 2 pixels per side. Earlier two-stage checkpoints are incompatible.
    """

    def __init__(self, width=16, classes=10, *, rngs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, 3, padding=1, rngs=rngs)
        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1, rngs=rngs)
        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, padding=1, rngs=rngs)
        self.conv4 = nn.Conv2d(width * 4, width * 8, 3, padding=1, rngs=rngs)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(width * 8 * 2 * 2, classes, rngs=rngs)

    def forward(self, x, *, rngs=None):
        x = jnp.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)))
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = self.pool(F.relu(conv(x)))
        return self.head(self.dropout(x.reshape(x.shape[0], -1), rngs=rngs))


class PetCNN(nn.Module):
    """RGB image classifier with global pooling and running normalization."""

    def __init__(self, width=16, *, rngs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, padding=1, rngs=rngs)
        self.norm1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1, rngs=rngs)
        self.norm2 = nn.BatchNorm2d(width * 2)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d()
        self.head = nn.Linear(width * 2, 2, rngs=rngs)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        return self.head(self.global_pool(x).reshape(x.shape[0], -1))


class SequenceRegressor(nn.Module):
    """Next-value forecasting with an RNN or LSTM and explicit returned carry."""

    def __init__(self, features=1, hidden=32, outputs=1, kind="rnn", *, rngs):
        super().__init__()
        if kind not in ("rnn", "lstm"):
            raise ValueError("kind must be rnn or lstm")
        self.kind = kind
        self.recurrent = (nn.RNN if kind == "rnn" else nn.LSTM)(features, hidden, rngs=rngs)
        self.head = nn.Linear(hidden, outputs, rngs=rngs)

    def forward(self, x, state=None, *, lengths=None):
        _, carry = self.recurrent(x, state, lengths=lengths)
        h = carry[0] if self.kind == "lstm" else carry
        return {"prediction": self.head(h[-1]), "state": carry}


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, classes, embedding_dim=32, hidden=32, *, rngs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, rngs=rngs)
        self.lstm = nn.LSTM(embedding_dim, hidden, rngs=rngs)
        self.head = nn.Linear(hidden, classes, rngs=rngs)

    def forward(self, tokens, state=None, *, lengths=None):
        _, carry = self.lstm(self.embedding(tokens), state, lengths=lengths)
        return {"logits": self.head(carry[0][-1]), "state": carry}


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, *, rngs):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, rngs=rngs)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4, rngs=rngs),
            nn.GELU(),
            nn.Linear(width * 4, width, rngs=rngs),
        )

    def forward(self, x, *, mask=None):
        z = self.norm1(x)
        x = x + self.attention(z, z, z, mask=mask)
        return x + self.ff(self.norm2(x))


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, max_length, width=32, depth=2, heads=4, *, rngs):
        super().__init__()
        self.max_length = max_length
        self.token = nn.Embedding(vocab_size, width, rngs=rngs)
        self.position = nn.Embedding(max_length, width, rngs=rngs)
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, heads, rngs=rngs) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, rngs=rngs)

    def forward(self, tokens, *, valid=None):
        if tokens.shape[1] > self.max_length:
            raise ValueError("Token sequence exceeds max_length")
        length = tokens.shape[1]
        x = self.token(tokens) + self.position(jnp.arange(length))[None]
        mask = nn.attention.causal_mask(length)
        if valid is not None:
            mask = mask & nn.attention.padding_mask(valid)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.head(self.norm(x))


class Autoencoder(nn.Module):
    """MNIST autoencoder with named reconstruction and latent outputs."""

    def __init__(self, latent_size=16, width=128, *, rngs):
        super().__init__()
        self.encoder = MLP(784, [width], latent_size, rngs=rngs)
        self.decoder = MLP(latent_size, [width], 784, rngs=rngs)

    def encode(self, images):
        return self.encoder(images.reshape(images.shape[0], 784))

    def decode(self, latent):
        return F.sigmoid(self.decoder(latent)).reshape(-1, 1, 28, 28)

    def forward(self, images):
        z = self.encode(images)
        return {"reconstruction": self.decode(z), "latent": z}


class Generator(nn.Module):
    """Fashion-MNIST generator; output range [-1, 1]."""

    def __init__(self, latent_size=64, width=128, *, rngs):
        super().__init__()
        self.net = MLP(latent_size, [width, width], 784, rngs=rngs)

    def forward(self, z):
        return jnp.tanh(self.net(z)).reshape(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self, width=128, *, rngs):
        super().__init__()
        self.net = MLP(784, [width, width], 1, rngs=rngs)

    def forward(self, images):
        return self.net(images.reshape(images.shape[0], 784))


class ConvAutoencoder(nn.Module):
    """28x28 reconstruction with a spatial latent tensor [B, channels, 7, 7]."""

    def __init__(self, channels=8, *, rngs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 4, stride=2, padding=1, rngs=rngs),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1, rngs=rngs),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, rngs=rngs),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, 1, 4, stride=2, padding=1, rngs=rngs),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return {"latent": z, "reconstruction": self.decode(z)}


class FashionGenerator(nn.Module):
    """Convolutional Fashion-MNIST generator, with explicit BatchNorm state."""

    def __init__(self, latent_size=64, width=64, *, rngs):
        super().__init__()
        self.width = width
        self.project = nn.Linear(latent_size, 2 * width * 7 * 7, rngs=rngs)
        self.norm1 = nn.BatchNorm2d(2 * width)
        self.up1 = nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1, rngs=rngs)
        self.norm2 = nn.BatchNorm2d(width)
        self.up2 = nn.ConvTranspose2d(width, 1, 4, stride=2, padding=1, rngs=rngs)

    def forward(self, z):
        x = self.project(z).reshape(-1, 2 * self.width, 7, 7)
        x = F.relu(self.norm1(x))
        x = F.relu(self.norm2(self.up1(x)))
        return jnp.tanh(self.up2(x))


class FashionDiscriminator(nn.Module):
    """Convolutional discriminator; outputs logits without a sigmoid."""

    def __init__(self, width=32, *, rngs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, 4, stride=2, padding=1, rngs=rngs)
        self.conv2 = nn.Conv2d(width, 2 * width, 4, stride=2, padding=1, rngs=rngs)
        self.head = nn.Linear(2 * width * 7 * 7, 1, rngs=rngs)

    def forward(self, images):
        x = self.conv1(images)
        x = jnp.where(x >= 0, x, 0.2 * x)
        x = self.conv2(x)
        x = jnp.where(x >= 0, x, 0.2 * x)
        return self.head(x.reshape(x.shape[0], -1))
