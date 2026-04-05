"""Animated quantum state evolution — probability bars morph as gates are applied.

Shows the probability distribution evolving step-by-step through a quantum
circuit. Each frame adds one gate, and the bars animate to show how
superposition, interference, and entanglement reshape the quantum state.

Two circuits are shown:
  1. GHZ state build-up: H → CNOT chain → maximally entangled state
  2. Interference circuit: complex rotations + entanglement → rich patterns

Usage:
    python experiments/state_evolution.py          # static frames
    python experiments/state_evolution.py --gif    # animated GIF
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsim.circuit import Circuit
from qsim.statevector import Statevector


def build_interference_circuit(n: int = 4) -> list[tuple[str, Circuit]]:
    """Build a circuit step-by-step, returning (label, circuit_so_far) at each step.

    Creates a visually dramatic sequence: uniform superposition → rotations →
    entanglement → interference patterns.
    """
    steps: list[tuple[str, Circuit]] = []

    qc = Circuit(n, name="evolution")
    steps.append(("|0000\u27e9", _copy_circuit(qc, n)))

    qc.h(0)
    steps.append(("H on q\u2080", _copy_circuit(qc, n)))

    qc.h(1)
    steps.append(("H on q\u2081", _copy_circuit(qc, n)))

    qc.h(2)
    steps.append(("H on q\u2082", _copy_circuit(qc, n)))

    qc.h(3)
    steps.append(("H on q\u2083  \u2014  uniform superposition", _copy_circuit(qc, n)))

    qc.ry(np.pi / 3, 0)
    steps.append(("Ry(\u03c0/3) on q\u2080", _copy_circuit(qc, n)))

    qc.rz(np.pi / 4, 1)
    steps.append(("Rz(\u03c0/4) on q\u2081", _copy_circuit(qc, n)))

    qc.rx(np.pi / 5, 2)
    steps.append(("Rx(\u03c0/5) on q\u2082", _copy_circuit(qc, n)))

    qc.cx(0, 1)
    steps.append(("CNOT(q\u2080, q\u2081)  \u2014  entangle!", _copy_circuit(qc, n)))

    qc.cx(1, 2)
    steps.append(("CNOT(q\u2081, q\u2082)", _copy_circuit(qc, n)))

    qc.cx(2, 3)
    steps.append(("CNOT(q\u2082, q\u2083)  \u2014  chain entanglement", _copy_circuit(qc, n)))

    qc.h(0)
    steps.append(("H on q\u2080  \u2014  interference", _copy_circuit(qc, n)))

    qc.ry(2 * np.pi / 3, 3)
    steps.append(("Ry(2\u03c0/3) on q\u2083", _copy_circuit(qc, n)))

    qc.cz(0, 3)
    steps.append(("CZ(q\u2080, q\u2083)  \u2014  phase kick", _copy_circuit(qc, n)))

    qc.h(2)
    steps.append(("H on q\u2082  \u2014  final interference", _copy_circuit(qc, n)))

    return steps


def _copy_circuit(qc: Circuit, n: int) -> Circuit:
    """Create an independent copy of a circuit."""
    new = Circuit(n, name=qc.name)
    for inst in qc.instructions:
        new._instructions.append(inst)
    return new


def create_gif(
    output_path: str = "experiments/state_evolution.gif",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.style.use("dark_background")
    rcParams["font.family"] = "monospace"

    n_qubits = 4
    n_states = 2**n_qubits
    steps = build_interference_circuit(n_qubits)

    all_probs = []
    for _, circ in steps:
        sv = Statevector.from_circuit(circ)
        all_probs.append(sv.probabilities())

    base_colors = plt.cm.cool(np.linspace(0.1, 0.9, n_states))

    n_data = len(steps)
    n_hold = 8
    n_frames = n_data + n_hold

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]

    title_text = fig.suptitle("", fontsize=16, color="white",
                               fontweight="bold", y=0.96)
    gate_text = ax.text(0.5, 0.95, "", transform=ax.transAxes,
                        fontsize=13, color="#ffd93d", ha="center",
                        va="top", fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  fc="#1a1a2e", ec="#ffd93d", alpha=0.9))

    step_counter = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                           fontsize=11, color="gray", va="top")

    bars = ax.bar(range(n_states), np.zeros(n_states),
                  color=base_colors, width=0.7, edgecolor="none")

    def init():
        ax.set_xlim(-0.5, n_states - 0.5)
        ax.set_ylim(0, 0.65)
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right",
                           color="white")
        ax.set_ylabel("Probability  |amplitude|\u00b2", fontsize=13,
                       color="white", labelpad=10)
        ax.tick_params(colors="white", labelsize=10)
        ax.grid(True, axis="y", alpha=0.1, linestyle="--")
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")

        title_text.set_text("Quantum State Evolution  \u2014  4 Qubits")
        fig.text(0.99, 0.01, "quantum-pipeline-from-scratch",
                 ha="right", fontsize=8, color="gray", alpha=0.4)
        return list(bars) + [gate_text, step_counter]

    def update(frame):
        idx = min(frame, n_data - 1)
        probs = all_probs[idx]
        label = steps[idx][0]

        for bar, p in zip(bars, probs):
            bar.set_height(p)

        max_p = max(probs.max(), 1e-10)
        for bar, p in zip(bars, probs):
            intensity = p / max_p
            r = 0.1 + 0.5 * intensity
            g = 0.7 * intensity + 0.1
            b = 0.9 * intensity + 0.3
            bar.set_color((r, g, min(b, 1.0), 0.6 + 0.4 * intensity))
            bar.set_edgecolor(
                (r, g, min(b, 1.0), 0.9) if intensity > 0.3 else "none"
            )

        gate_text.set_text(f"Gate {idx}/{n_data - 1}:  {label}")
        step_counter.set_text(f"Step {idx}")

        y_max = max(probs.max() * 1.3, 0.15)
        ax.set_ylim(0, y_max)

        return list(bars) + [gate_text, step_counter]

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=1500, blit=False)
    writer = PillowWriter(fps=0.7)
    anim.save(output_path, writer=writer, dpi=150,
              savefig_kwargs={"facecolor": "#0d1117"})
    print(f"Saved: {output_path}")
    plt.close()


def create_static(
    output_path: str = "experiments/state_evolution.png",
) -> None:
    """Create a multi-panel static image showing key moments."""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    plt.style.use("dark_background")
    rcParams["font.family"] = "monospace"

    n_qubits = 4
    n_states = 2**n_qubits
    steps = build_interference_circuit(n_qubits)
    labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]

    key_indices = [0, 4, 8, 10, 13, len(steps) - 1]
    key_indices = [i for i in key_indices if i < len(steps)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Quantum State Evolution  \u2014  4 Qubits, 15 Gates",
                 fontsize=18, color="white", fontweight="bold", y=0.98)

    for ax_idx, step_idx in enumerate(key_indices):
        ax = axes.flat[ax_idx]
        ax.set_facecolor("#0d1117")

        sv = Statevector.from_circuit(steps[step_idx][1])
        probs = sv.probabilities()
        label = steps[step_idx][0]

        max_p = max(probs.max(), 1e-10)
        colors = []
        for p in probs:
            intensity = p / max_p
            r = 0.1 + 0.5 * intensity
            g = 0.7 * intensity + 0.1
            b = 0.9 * intensity + 0.3
            colors.append((r, g, min(b, 1.0), 0.6 + 0.4 * intensity))

        ax.bar(range(n_states), probs, color=colors, width=0.7)
        ax.set_title(f"Step {step_idx}: {label}", fontsize=10,
                     color="#ffd93d", pad=8)
        ax.set_ylim(0, max(probs.max() * 1.3, 0.15))
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right",
                           color="gray")
        ax.tick_params(colors="gray", labelsize=8)
        ax.grid(True, axis="y", alpha=0.08, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")

    fig.text(0.99, 0.01, "quantum-pipeline-from-scratch",
             ha="right", fontsize=8, color="gray", alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    print("Building quantum state evolution visualization...\n")
    create_static()
    if args.gif:
        create_gif()


if __name__ == "__main__":
    main()
