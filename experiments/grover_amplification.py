"""Grover's algorithm probability amplification — the most visceral quantum viz.

Shows the "probability pump" effect: over successive Grover iterations, the
target state's probability grows while all others shrink. The bars visually
pulse as quantum amplitude amplification does its work.

This is quantum computing's most photogenic algorithm.

Usage:
    python experiments/grover_amplification.py          # static
    python experiments/grover_amplification.py --gif    # animated GIF
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsim.circuit import Circuit
from qsim.statevector import Statevector


def _toffoli(qc: Circuit, c0: int, c1: int, tgt: int) -> None:
    """Toffoli (CCX) decomposition from Nielsen & Chuang, Fig 4.9.

    Uses Phase(±π/4) for T/T† gates — NOT Rz, which differs by a global
    phase that doesn't cancel in the decomposition.
    """
    qc.h(tgt)
    qc.cx(c1, tgt)
    qc.phase(-np.pi / 4, tgt)
    qc.cx(c0, tgt)
    qc.phase(np.pi / 4, tgt)
    qc.cx(c1, tgt)
    qc.phase(-np.pi / 4, tgt)
    qc.cx(c0, tgt)
    qc.phase(np.pi / 4, c1)
    qc.phase(np.pi / 4, tgt)
    qc.h(tgt)
    qc.cx(c0, c1)
    qc.phase(np.pi / 4, c0)
    qc.phase(-np.pi / 4, c1)
    qc.cx(c0, c1)


def _ccz(qc: Circuit, q0: int, q1: int, q2: int) -> None:
    """Controlled-controlled-Z via CCZ = H · Toffoli · H on target."""
    qc.h(q2)
    _toffoli(qc, q0, q1, q2)
    qc.h(q2)


def _mcz(qc: Circuit, qubits: list[int]) -> None:
    """Multi-controlled Z on the given qubits (flips phase of |11...1>).

    Decomposed recursively into Toffoli + ancilla-free gates.
    """
    n = len(qubits)
    if n == 2:
        qc.cz(qubits[0], qubits[1])
    elif n == 3:
        _ccz(qc, qubits[0], qubits[1], qubits[2])
    elif n == 4:
        qc.h(qubits[3])
        _toffoli(qc, qubits[0], qubits[1], qubits[3])
        qc.cx(qubits[2], qubits[3])
        _toffoli(qc, qubits[0], qubits[1], qubits[3])
        qc.cx(qubits[2], qubits[3])
        qc.h(qubits[3])
    else:
        raise NotImplementedError(f"MCZ not implemented for {n} qubits")


def grover_oracle(qc: Circuit, target_bits: str) -> None:
    """Oracle: flip phase of |target_bits⟩.

    Strategy: X gates map target to |11...1⟩, MCZ flips its phase, undo X.
    """
    n = qc.num_qubits
    for i, bit in enumerate(target_bits):
        if bit == "0":
            qc.x(i)

    _mcz(qc, list(range(n)))

    for i, bit in enumerate(target_bits):
        if bit == "0":
            qc.x(i)


def grover_diffusion(qc: Circuit) -> None:
    """Diffusion operator: 2|s⟩⟨s| - I where |s⟩ = H^n|0⟩.

    Implementation: H^n · X^n · MCZ · X^n · H^n
    """
    n = qc.num_qubits
    for i in range(n):
        qc.h(i)
    for i in range(n):
        qc.x(i)

    _mcz(qc, list(range(n)))

    for i in range(n):
        qc.x(i)
    for i in range(n):
        qc.h(i)


def run_grover_steps(
    n_qubits: int = 3,
    target: str = "101",
    n_iterations: int | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Run Grover's algorithm, capturing probability distribution at each step.

    Returns: [(label, probabilities)] for each step.
    """
    if n_iterations is None:
        n_iterations = max(1, int(np.round(np.pi / 4 * np.sqrt(2**n_qubits))))

    target_idx = int(target, 2)
    results: list[tuple[str, np.ndarray]] = []

    sv = Statevector(n_qubits)
    results.append(("|" + "0" * n_qubits + "\u27e9  initial", sv.probabilities()))

    qc_init = Circuit(n_qubits)
    for i in range(n_qubits):
        qc_init.h(i)
    sv = Statevector.from_circuit(qc_init)
    p_uniform = sv.probabilities()[target_idx]
    results.append((
        f"H\u2297{n_qubits}  \u2014  all states equal at {p_uniform:.1%}",
        sv.probabilities(),
    ))

    for it in range(1, n_iterations + 1):
        qc = Circuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        for _ in range(it):
            grover_oracle(qc, target)
            grover_diffusion(qc)

        sv = Statevector.from_circuit(qc)
        probs = sv.probabilities()
        target_prob = probs[target_idx]
        results.append((
            f"Iteration {it}:  P(|{target}\u27e9) = {target_prob:.1%}",
            probs,
        ))

    return results


def create_gif(
    output_path: str = "experiments/grover_amplification.gif",
    n_qubits: int = 3,
    target: str = "101",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.style.use("dark_background")
    rcParams["font.family"] = "monospace"

    steps = run_grover_steps(n_qubits, target)
    n_states = 2**n_qubits
    labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]
    target_idx = int(target, 2)

    n_data = len(steps)
    n_hold = 12
    n_frames = n_data + n_hold

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    title_text = fig.suptitle("", fontsize=17, color="white",
                               fontweight="bold", y=0.97)
    step_text = ax.text(0.5, 0.93, "", transform=ax.transAxes,
                        fontsize=14, ha="center", va="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  fc="#1a1a2e", ec="#ffd93d", alpha=0.9),
                        color="#ffd93d")

    bars = ax.bar(range(n_states), np.zeros(n_states), width=0.7)

    def init():
        ax.set_xlim(-0.5, n_states - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(
            [f"|{label}\u27e9" for label in labels],
            fontsize=11, color="white",
        )
        ax.set_ylabel("Probability", fontsize=14, color="white", labelpad=10)
        ax.tick_params(colors="white", labelsize=11)
        ax.grid(True, axis="y", alpha=0.1, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")

        title_text.set_text(
            f"Grover's Search  \u2014  finding |{target}\u27e9 "
            f"among {n_states} states"
        )

        ax.axvline(x=target_idx, color="#ff6348", alpha=0.12,
                   linewidth=30, zorder=0)
        ax.text(target_idx, 1.01, "\u2193 target", ha="center",
                fontsize=11, color="#ff6348", alpha=0.8)

        fig.text(0.99, 0.01, "quantum-pipeline-from-scratch",
                 ha="right", fontsize=8, color="gray", alpha=0.4)

        return list(bars) + [step_text]

    def update(frame):
        idx = min(frame, n_data - 1)
        label, probs = steps[idx]

        for i, (bar, p) in enumerate(zip(bars, probs)):
            bar.set_height(p)
            if i == target_idx:
                intensity = min(p * 2.5, 1.0)
                bar.set_color((1.0, 0.35 + 0.55 * intensity, 0.05, 0.9))
                bar.set_edgecolor("#ff6348")
                bar.set_linewidth(2)
            else:
                bar.set_color((0.2, 0.35, 0.75, 0.3 + 0.5 * min(p * 8, 1.0)))
                bar.set_edgecolor("none")
                bar.set_linewidth(0)

        step_text.set_text(label)
        return list(bars) + [step_text]

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=1200, blit=False)
    writer = PillowWriter(fps=1)
    anim.save(output_path, writer=writer, dpi=150,
              savefig_kwargs={"facecolor": "#0d1117"})
    print(f"Saved: {output_path}")
    plt.close()


def create_mp4(
    output_path: str = "experiments/grover_amplification.mp4",
    n_qubits: int = 3,
    target: str = "101",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    plt.style.use("dark_background")
    rcParams["font.family"] = "monospace"

    steps = run_grover_steps(n_qubits, target)
    n_states = 2**n_qubits
    labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]
    target_idx = int(target, 2)

    n_data = len(steps)
    n_hold = 30
    n_frames = n_data + n_hold

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    title_text = fig.suptitle("", fontsize=17, color="white",
                               fontweight="bold", y=0.97)
    step_text = ax.text(0.5, 0.93, "", transform=ax.transAxes,
                        fontsize=14, ha="center", va="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  fc="#1a1a2e", ec="#ffd93d", alpha=0.9),
                        color="#ffd93d")

    bars = ax.bar(range(n_states), np.zeros(n_states), width=0.7)

    def init():
        ax.set_xlim(-0.5, n_states - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(
            [f"|{label}\u27e9" for label in labels],
            fontsize=11, color="white",
        )
        ax.set_ylabel("Probability", fontsize=14, color="white", labelpad=10)
        ax.tick_params(colors="white", labelsize=11)
        ax.grid(True, axis="y", alpha=0.1, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")

        title_text.set_text(
            f"Grover's Search  \u2014  finding |{target}\u27e9 "
            f"among {n_states} states"
        )

        ax.axvline(x=target_idx, color="#ff6348", alpha=0.12,
                   linewidth=30, zorder=0)
        ax.text(target_idx, 1.01, "\u2193 target", ha="center",
                fontsize=11, color="#ff6348", alpha=0.8)

        fig.text(0.99, 0.01, "quantum-pipeline-from-scratch",
                 ha="right", fontsize=8, color="gray", alpha=0.4)

        return list(bars) + [step_text]

    def update(frame):
        idx = min(frame, n_data - 1)
        label, probs = steps[idx]

        for i, (bar, p) in enumerate(zip(bars, probs)):
            bar.set_height(p)
            if i == target_idx:
                intensity = min(p * 2.5, 1.0)
                bar.set_color((1.0, 0.35 + 0.55 * intensity, 0.05, 0.9))
                bar.set_edgecolor("#ff6348")
                bar.set_linewidth(2)
            else:
                bar.set_color((0.2, 0.35, 0.75, 0.3 + 0.5 * min(p * 8, 1.0)))
                bar.set_edgecolor("none")
                bar.set_linewidth(0)

        step_text.set_text(label)
        return list(bars) + [step_text]

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=1200, blit=False)
    writer = FFMpegWriter(fps=1, metadata={"title": "Grover's Algorithm"})
    anim.save(output_path, writer=writer, dpi=200,
              savefig_kwargs={"facecolor": "#0d1117"})
    print(f"Saved: {output_path}")
    plt.close()


def create_static(
    output_path: str = "experiments/grover_amplification.png",
    n_qubits: int = 3,
    target: str = "101",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    plt.style.use("dark_background")
    rcParams["font.family"] = "monospace"

    steps = run_grover_steps(n_qubits, target)
    n_states = 2**n_qubits
    labels = [f"|{format(i, f'0{n_qubits}b')}\u27e9" for i in range(n_states)]
    target_idx = int(target, 2)

    n_panels = len(steps)
    cols = min(n_panels, 4)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"Grover's Algorithm  \u2014  Searching for |{target}\u27e9 among {n_states} states",
        fontsize=16, color="white", fontweight="bold", y=0.99,
    )

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for ax_idx in range(len(axes)):
        ax = axes[ax_idx]
        ax.set_facecolor("#0d1117")

        if ax_idx >= n_panels:
            ax.set_visible(False)
            continue

        label, probs = steps[ax_idx]

        colors = []
        for i, p in enumerate(probs):
            if i == target_idx:
                intensity = min(p * 2.5, 1.0)
                colors.append((1.0, 0.35 + 0.55 * intensity, 0.05, 0.9))
            else:
                colors.append((0.2, 0.35, 0.75, 0.3 + 0.5 * min(p * 8, 1.0)))

        ax.bar(range(n_states), probs, color=colors, width=0.7)
        ax.set_title(label, fontsize=10, color="#ffd93d", pad=8)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(labels, fontsize=7, color="gray")
        ax.tick_params(colors="gray", labelsize=8)
        ax.grid(True, axis="y", alpha=0.08, linestyle="--")
        ax.axvline(x=target_idx, color="#ff6348", alpha=0.1, linewidth=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("gray")
        ax.spines["bottom"].set_color("gray")

    fig.text(0.99, 0.01, "quantum-pipeline-from-scratch",
             ha="right", fontsize=8, color="gray", alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--target", type=str, default="101")
    args = parser.parse_args()

    print(f"Running Grover's algorithm ({args.qubits} qubits, "
          f"target=|{args.target}>)...\n")

    create_static(n_qubits=args.qubits, target=args.target)
    if args.gif:
        create_gif(n_qubits=args.qubits, target=args.target)


if __name__ == "__main__":
    main()
