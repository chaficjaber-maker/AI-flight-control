from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


@dataclass
class MLPWeights:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def copy(self) -> "MLPWeights":
        return MLPWeights(
            w1=self.w1.copy(),
            b1=self.b1.copy(),
            w2=self.w2.copy(),
            b2=self.b2.copy(),
        )


class MLP:
    """Two-layer MLP with tanh activations (NumPy-only for embedded-friendly dev)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weights = MLPWeights(
            w1=rng.normal(0, 0.2, (input_dim, hidden_dim)),
            b1=np.zeros(hidden_dim),
            w2=rng.normal(0, 0.2, (hidden_dim, output_dim)),
            b2=np.zeros(output_dim),
        )
        self._last_x: np.ndarray | None = None
        self._last_h: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        h = _tanh(x @ self.weights.w1 + self.weights.b1)
        y = h @ self.weights.w2 + self.weights.b2
        self._last_x = x
        self._last_h = h
        return y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.weights.w1,
            b1=self.weights.b1,
            w2=self.weights.w2,
            b2=self.weights.b2,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )

    @classmethod
    def load(cls, path: Path) -> "MLP":
        data = np.load(path)
        net = cls(int(data["input_dim"]), int(data["hidden_dim"]), int(data["output_dim"]))
        net.weights = MLPWeights(
            w1=data["w1"],
            b1=data["b1"],
            w2=data["w2"],
            b2=data["b2"],
        )
        return net


def train_supervised(
    net: MLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int = 64,
) -> list[float]:
    """Simple MSE regression with manual backprop through tanh."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    losses: list[float] = []

    for _ in range(epochs):
        idx = np.random.default_rng().permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            preds = np.stack([net.forward(row) for row in xb])
            loss = np.mean((preds - yb) ** 2)
            epoch_loss += loss
            for i, row in enumerate(xb):
                pred = net.forward(row)
                err = pred - yb[i]
                h = net._last_h
                assert h is not None
                grad_y = 2.0 * err / len(xb)
                grad_w2 = np.outer(h, grad_y)
                grad_b2 = grad_y
                grad_h = grad_y @ net.weights.w2.T
                grad_h *= 1.0 - h * h
                grad_w1 = np.outer(row, grad_h)
                grad_b1 = grad_h
                net.weights.w2 -= learning_rate * grad_w2
                net.weights.b2 -= learning_rate * grad_b2
                net.weights.w1 -= learning_rate * grad_w1
                net.weights.b1 -= learning_rate * grad_b1
        losses.append(float(epoch_loss / max(1, n // batch_size)))
    return losses
