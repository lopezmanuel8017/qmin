"""End-to-end training demo: hybrid quantum-classical classifier on MNIST.

Trains a minimal hybrid model on downscaled MNIST (0 vs 1):
  Flatten(8x8) → Linear(64, 4) → Quantum(4 qubits) → Linear(4, 2)

Proves that gradient flow through the quantum circuit via parameter-shift
actually leads to learning — loss decreases, accuracy improves.

Usage:
    python experiments/training_demo.py
    python experiments/training_demo.py --epochs 25 --train-size 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classical.layers import Linear
from classical.loss import CrossEntropyLoss
from classical.optim import AdamW
from pipeline.hybrid_layer import HybridQuantumClassicalLayer
from quantum.ansatz import CNOTLadder
from quantum.encoding import AngleEncoder


def load_data(
    class_a: int = 4,
    class_b: int = 9,
    train_size: int = 50,
    test_size: int = 20,
    img_size: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 2-class MNIST downscaled to img_size x img_size.

    Returns flattened images: (N, img_size*img_size).
    """
    try:
        from data.mnist import load_mnist
        x_all, y_all, x_test_all, y_test_all = load_mnist()
    except Exception:
        print("  MNIST unavailable, using synthetic data")
        return _synthetic(train_size, test_size, img_size, seed)

    rng = np.random.default_rng(seed)

    tr_mask = (y_all == class_a) | (y_all == class_b)
    te_mask = (y_test_all == class_a) | (y_test_all == class_b)
    x_tr, y_tr = x_all[tr_mask], (y_all[tr_mask] == class_b).astype(int)
    x_te, y_te = x_test_all[te_mask], (y_test_all[te_mask] == class_b).astype(int)

    idx_a = np.where(y_tr == 0)[0]
    idx_b = np.where(y_tr == 1)[0]
    half = train_size // 2
    tr_idx = np.concatenate([
        rng.choice(idx_a, half, replace=False),
        rng.choice(idx_b, half, replace=False),
    ])
    rng.shuffle(tr_idx)

    idx_a_te = np.where(y_te == 0)[0]
    idx_b_te = np.where(y_te == 1)[0]
    half_te = test_size // 2
    te_idx = np.concatenate([
        rng.choice(idx_a_te, half_te, replace=False),
        rng.choice(idx_b_te, half_te, replace=False),
    ])

    x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]
    x_te, y_te = x_te[te_idx], y_te[te_idx]

    x_tr = _downscale(x_tr, img_size)
    x_te = _downscale(x_te, img_size)

    return x_tr, y_tr, x_te, y_te


def _downscale(images: np.ndarray, size: int) -> np.ndarray:
    """(N, 1, 28, 28) → (N, size*size) via block averaging."""
    N = images.shape[0]
    imgs = images[:, 0]
    block = 28 // size
    result = np.zeros((N, size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            result[:, i, j] = imgs[
                :, i*block:(i+1)*block, j*block:(j+1)*block
            ].mean(axis=(1, 2))
    return result.reshape(N, -1)


def _synthetic(
    train_size: int, test_size: int, img_size: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic binary classification data."""
    rng = np.random.default_rng(seed)
    dim = img_size * img_size

    def make_batch(n):
        x = rng.uniform(0, 0.2, (n, dim)).astype(np.float32)
        y = np.zeros(n, dtype=int)
        for i in range(n):
            if i % 2 == 0:
                x[i, :dim//2] += rng.uniform(0.5, 1.0, dim//2)
            else:
                x[i, dim//2:] += rng.uniform(0.5, 1.0, dim - dim//2)
                y[i] = 1
        return x, y

    x_tr, y_tr = make_batch(train_size)
    x_te, y_te = make_batch(test_size)
    return x_tr, y_tr, x_te, y_te


class MinimalHybridClassifier:
    """Flatten → Linear → ReLU → Linear → Quantum → Linear.

    Minimal architecture to demonstrate quantum layer training.
    """

    def __init__(self, input_dim: int, num_qubits: int = 4,
                 num_layers: int = 1) -> None:
        self.proj = Linear(input_dim, num_qubits, bias=False)
        self.proj.weight[:] = np.random.default_rng(42).normal(
            0, 0.01, self.proj.weight.shape,
        )

        ansatz = CNOTLadder(num_qubits, num_layers)
        encoder = AngleEncoder(num_qubits)
        self.quantum = HybridQuantumClassicalLayer(
            num_qubits, ansatz, encoder,
            compute_input_grad=True,
            init_strategy="uniform",
        )
        self.head = Linear(num_qubits, 2)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = self.proj.forward(x)
        out = self.quantum.forward(out)
        return self.head.forward(out)

    def backward(self) -> None:
        grad = self.loss_fn.backward()
        grad = self.head.backward(grad)
        grad = self.quantum.backward(grad)
        self.quantum.sync_grads_to_array()
        self.proj.backward(grad)

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = self.proj.parameters()
        params += self.quantum.parameters()
        params += self.head.parameters()
        return params


def train(
    num_epochs: int = 20,
    train_size: int = 50,
    test_size: int = 20,
    num_qubits: int = 4,
    num_layers: int = 1,
    lr: float = 0.01,
) -> dict:
    """Train and return history."""

    print("Loading data...")
    x_train, y_train, x_test, y_test = load_data(
        train_size=train_size, test_size=test_size,
    )
    input_dim = x_train.shape[1]
    print(f"  Train: {x_train.shape} ({np.bincount(y_train)} per class)")
    print(f"  Test:  {x_test.shape} ({np.bincount(y_test)} per class)")

    print(f"\nModel: Linear({input_dim}→{num_qubits}) "
          f"→ Quantum({num_qubits}q, {num_layers}L) → Linear({num_qubits}→2)")

    model = MinimalHybridClassifier(input_dim, num_qubits, num_layers)

    _ = model.forward(x_train[0])
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.001)

    history: dict[str, list] = {
        "train_loss": [], "test_loss": [], "test_acc": [],
    }

    correct = 0
    test_loss_0 = 0.0
    for i in range(len(x_test)):
        logits = model.forward(x_test[i])
        test_loss_0 += model.loss_fn.forward(logits, y_test[i:i+1])
        if np.argmax(logits) == y_test[i]:
            correct += 1
    test_loss_0 /= len(x_test)
    test_acc_0 = correct / len(x_test)
    history["train_loss"].append(float("nan"))
    history["test_loss"].append(test_loss_0)
    history["test_acc"].append(test_acc_0)

    print(f"\nTraining for {num_epochs} epochs...\n")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Test Loss':>9}  "
          f"{'Accuracy':>8}  {'Time':>6}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*6}")
    print(f"  {'0':>5}  {'---':>10}  {test_loss_0:9.4f}  "
          f"{test_acc_0:7.1%}  {'---':>6}")

    for epoch in range(num_epochs):
        t0 = time.time()
        epoch_loss = 0.0
        indices = np.random.default_rng(epoch).permutation(len(x_train))

        for idx in indices:
            logits = model.forward(x_train[idx])
            loss = model.loss_fn.forward(logits, y_train[idx:idx+1])
            model.backward()
            optimizer.step()
            model.quantum.sync_from_array()
            optimizer.zero_grad()
            epoch_loss += loss

        avg_loss = epoch_loss / len(x_train)

        correct = 0
        test_loss = 0.0
        for i in range(len(x_test)):
            logits = model.forward(x_test[i])
            test_loss += model.loss_fn.forward(logits, y_test[i:i+1])
            if np.argmax(logits) == y_test[i]:
                correct += 1

        test_loss /= len(x_test)
        test_acc = correct / len(x_test)
        dt = time.time() - t0

        history["train_loss"].append(avg_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"  {epoch+1:5d}  {avg_loss:10.4f}  {test_loss:9.4f}  "
              f"{test_acc:7.1%}  {dt:5.1f}s")

    print(f"\n  Final test accuracy: {history['test_acc'][-1]:.1%}")

    return history


def plot_training(
    history: dict,
    output_path: str = "experiments/training_demo.png",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    plt.style.use("dark_background")
    rcParams["font.family"] = "monospace"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")

    epochs = range(0, len(history["train_loss"]))

    train_epochs = [e for e, v in zip(epochs, history["train_loss"]) if not np.isnan(v)]
    train_vals = [v for v in history["train_loss"] if not np.isnan(v)]
    ax1.plot(train_epochs, train_vals, color="#ff4757", linewidth=2.5,
             marker="o", markersize=5, label="Train")
    ax1.plot(list(epochs), history["test_loss"], color="#ffa502", linewidth=2.5,
             marker="s", markersize=5, label="Test", linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=13, color="white")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=13, color="white")
    ax1.set_title("Loss", fontsize=14, color="white", fontweight="bold")
    ax1.legend(fontsize=11, framealpha=0.3)
    ax1.grid(True, alpha=0.12, linestyle="--")
    ax1.tick_params(colors="white", labelsize=10)

    accs = [a * 100 for a in history["test_acc"]]
    ax2.plot(list(epochs), accs, color="#2ed573", linewidth=2.5,
             marker="D", markersize=5)
    ax2.axhline(y=50, color="gray", linestyle=":", alpha=0.5,
                label="Random chance (50%)")
    ax2.set_xlabel("Epoch", fontsize=13, color="white")
    ax2.set_ylabel("Test Accuracy (%)", fontsize=13, color="white")
    ax2.set_title("Accuracy", fontsize=14, color="white", fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=11, framealpha=0.3, loc="lower right")
    ax2.grid(True, alpha=0.12, linestyle="--")
    ax2.tick_params(colors="white", labelsize=10)

    final_acc = history["test_acc"][-1] * 100
    ax2.annotate(
        f"{final_acc:.0f}%",
        xy=(len(accs) - 1, final_acc),
        xytext=(len(accs) - 5, max(final_acc - 18, 10)),
        fontsize=18, color="#2ed573", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#2ed573", lw=2),
    )

    fig.suptitle(
        "End-to-End Hybrid Quantum-Classical Training\n"
        "MNIST (4 vs 9)  ·  4 qubits  ·  parameter-shift backprop",
        fontsize=14, color="white", fontweight="bold", y=1.02,
    )

    fig.text(
        0.99, 0.01,
        "quantum-pipeline-from-scratch  ·  zero dependencies",
        ha="right", fontsize=8, color="gray", alpha=0.4,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    history = train(
        num_epochs=args.epochs,
        train_size=args.train_size,
        test_size=args.test_size,
        num_qubits=args.qubits,
        num_layers=args.layers,
        lr=args.lr,
    )
    plot_training(history)


if __name__ == "__main__":
    main()
