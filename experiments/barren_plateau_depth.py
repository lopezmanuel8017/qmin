"""Barren plateau visualization — the real one.

Shows gradient variance vs qubit count for DEEP circuits (depth = n) vs
SHALLOW circuits (depth = 1). Deep circuits approach a 2-design where
McClean et al. (2018) proved Var[∂C/∂θ] ~ O(2^{-n}).

The shallow curve decays gently. The deep curve crashes exponentially.
That's the barren plateau.

For large qubit counts (>12), only a random subset of parameters is sampled
to keep computation tractable. The theorem applies to ANY parameter, so
subset sampling is statistically valid.

Usage:
    python experiments/barren_plateau_depth.py              # PNG only
    python experiments/barren_plateau_depth.py --gif        # + animated GIF
    python experiments/barren_plateau_depth.py --fast       # fewer samples
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsim.circuit import Circuit
from qsim.measurement import expectation_value
from qsim.observables import Observable
from qsim.parameters import Parameter
from qsim.statevector import Statevector
from quantum.ansatz import CNOTLadder

SHIFT = np.pi / 2


def _single_param_gradient(
    circuit: Circuit,
    observable: Observable,
    param_values: dict[Parameter, float],
    target_param: Parameter,
) -> float:
    """Parameter-shift gradient for a single parameter. 2 circuit evals."""
    theta = param_values[target_param]

    shifted_plus = dict(param_values)
    shifted_plus[target_param] = theta + SHIFT
    bound_plus = circuit.bind_parameters(shifted_plus)
    sv_plus = Statevector.from_circuit(bound_plus)
    exp_plus = expectation_value(sv_plus, observable)

    shifted_minus = dict(param_values)
    shifted_minus[target_param] = theta - SHIFT
    bound_minus = circuit.bind_parameters(shifted_minus)
    sv_minus = Statevector.from_circuit(bound_minus)
    exp_minus = expectation_value(sv_minus, observable)

    return (exp_plus - exp_minus) / 2.0


def compute_variance(
    num_qubits: int,
    num_layers: int,
    n_samples: int = 30,
    n_param_subset: int | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute mean gradient variance.

    When n_param_subset is set, only samples that many random parameters
    instead of all parameters. Valid because the barren plateau theorem
    applies to any individual parameter.

    Returns (mean_variance, std_variance).
    """
    ansatz = CNOTLadder(num_qubits, num_layers)
    circuit = ansatz.circuit()
    all_params = ansatz.parameters
    obs = Observable.z(0)

    rng = np.random.default_rng(seed + 999)

    if n_param_subset is not None and n_param_subset < len(all_params):
        param_indices = rng.choice(len(all_params), size=n_param_subset, replace=False)
        params = [all_params[i] for i in param_indices]
    else:
        params = all_params

    all_grads = np.zeros((n_samples, len(params)))
    for i in range(n_samples):
        pv = ansatz.init_params(seed=seed + i * 137, strategy="uniform")
        for j, p in enumerate(params):
            all_grads[i, j] = _single_param_gradient(circuit, obs, pv, p)

    variances = np.var(all_grads, axis=0)
    return float(np.mean(variances)), float(np.std(variances))


def compute_all_data(
    qubit_range: list[int],
    n_samples: int = 30,
    n_param_subset: int | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Compute gradient variance for deep and shallow circuits."""
    results: dict[str, list[tuple[float, float]]] = {
        "deep": [], "shallow": [],
    }

    total = len(qubit_range) * 2
    step = 0

    for n in qubit_range:
        n_sub_deep = n_param_subset if n > 10 else None

        step += 1
        print(f"  [{step}/{total}] {n} qubits, depth=1...",
              end="", flush=True)
        t0 = time.time()
        mv, sv = compute_variance(n, 1, n_samples, None)
        dt = time.time() - t0
        results["shallow"].append((mv, sv))
        print(f" var={mv:.2e}  ({dt:.1f}s)")

        step += 1
        tag = f" (subset={n_sub_deep})" if n_sub_deep else ""
        print(f"  [{step}/{total}] {n} qubits, depth={n}{tag}...",
              end="", flush=True)
        t0 = time.time()
        mv, sv = compute_variance(n, n, n_samples, n_sub_deep)
        dt = time.time() - t0
        results["deep"].append((mv, sv))
        print(f" var={mv:.2e}  ({dt:.1f}s)")

    return results


def create_static(
    data: dict,
    qubit_range: list[int],
    output_path: str = "experiments/barren_plateau_depth.png",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    plt.style.use("dark_background")
    rcParams["font.family"] = "monospace"

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    deep_means = [d[0] for d in data["deep"]]
    deep_stds = [d[1] for d in data["deep"]]
    shallow_means = [d[0] for d in data["shallow"]]
    shallow_stds = [d[1] for d in data["shallow"]]

    ax.semilogy(
        qubit_range, deep_means,
        color="#ff4757", marker="o", markersize=10, linewidth=3,
        label="Deep circuit (depth = n)  \u2014  barren plateau",
        zorder=5,
    )
    deep_lo = [max(m - s, 1e-18) for m, s in zip(deep_means, deep_stds)]
    deep_hi = [m + s for m, s in zip(deep_means, deep_stds)]
    ax.fill_between(qubit_range, deep_lo, deep_hi,
                    color="#ff4757", alpha=0.12, zorder=2)

    ax.semilogy(
        qubit_range, shallow_means,
        color="#2ed573", marker="s", markersize=9, linewidth=3,
        label="Shallow circuit (depth = 1)  \u2014  trainable",
        zorder=5,
    )
    shallow_lo = [max(m - s, 1e-18) for m, s in zip(shallow_means, shallow_stds)]
    shallow_hi = [m + s for m, s in zip(shallow_means, shallow_stds)]
    ax.fill_between(qubit_range, shallow_lo, shallow_hi,
                    color="#2ed573", alpha=0.12, zorder=2)

    log_deep = np.log2([m for m in deep_means if m > 0])
    n_arr = np.array(qubit_range[:len(log_deep)], dtype=float)
    slope, intercept = np.polyfit(n_arr, log_deep, 1)
    fit_x = np.linspace(qubit_range[0], qubit_range[-1], 200)
    fit_y = 2.0 ** (slope * fit_x + intercept)
    ax.semilogy(
        fit_x, fit_y,
        color="white", linewidth=1.5, linestyle="--", alpha=0.4,
        label=f"Fit: O(2^{{{slope:.2f}n}})",
        zorder=3,
    )

    d_last = deep_means[-1]
    s_last = shallow_means[-1]
    if d_last > 0 and s_last > 0:
        gap = s_last / d_last
        mid_y = np.sqrt(d_last * s_last)
        ax.annotate(
            f"{gap:.0f}x",
            xy=(qubit_range[-1] + 0.3, mid_y),
            fontsize=24, color="white", fontweight="bold", va="center",
        )
        ax.plot(
            [qubit_range[-1] + 0.15] * 2,
            [d_last * 1.5, s_last * 0.7],
            color="white", alpha=0.35, lw=1.5, linestyle="--",
        )

    collapse = deep_means[0] / d_last if d_last > 0 else 0
    ax.annotate(
        f"Collapsed {collapse:.0f}x",
        xy=(qubit_range[-1], d_last),
        xytext=(qubit_range[-1] - 5, d_last * 0.04),
        fontsize=12, color="#ff4757", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#ff4757", lw=2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e",
                  ec="#ff4757", alpha=0.85),
    )

    ax.annotate(
        "Still trainable \u2714",
        xy=(qubit_range[-1], s_last),
        xytext=(qubit_range[-1] - 5, s_last * 15),
        fontsize=12, color="#2ed573", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#2ed573", lw=2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e",
                  ec="#2ed573", alpha=0.85),
    )

    extrap_n = 50
    extrap_val = 2.0 ** (slope * extrap_n + intercept)
    ax.text(
        0.02, 0.05,
        f"At 50 qubits: gradient variance ~ {extrap_val:.1e}\n"
        f"Your optimizer sees a flat desert in every direction.",
        transform=ax.transAxes, fontsize=10, color="gray",
        alpha=0.7, va="bottom", fontfamily="monospace",
    )

    ax.set_xlabel("Number of Qubits", fontsize=14, color="white", labelpad=10)
    ax.set_ylabel("Gradient Variance (log scale)", fontsize=14,
                  color="white", labelpad=10)
    ax.set_title(
        "The Barren Plateau\n"
        "Deep quantum circuits have exponentially vanishing gradients",
        fontsize=16, color="white", fontweight="bold", pad=18,
    )
    ax.set_xticks(qubit_range)
    ax.tick_params(colors="white", labelsize=11)
    ax.grid(True, alpha=0.12, linestyle="--")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.4,
              fancybox=True, edgecolor="gray")

    fig.text(
        0.99, 0.01,
        "quantum-pipeline-from-scratch  \u00b7  parameter-shift gradients"
        "  \u00b7  zero dependencies",
        ha="right", fontsize=8, color="gray", alpha=0.5,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {output_path}")
    plt.close()


def create_gif(
    data: dict,
    qubit_range: list[int],
    output_path: str = "experiments/barren_plateau_depth.gif",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.style.use("dark_background")
    rcParams["font.family"] = "monospace"

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    n_data = len(qubit_range)
    n_hold = 15
    n_frames = n_data + n_hold

    deep_means = [d[0] for d in data["deep"]]
    shallow_means = [d[0] for d in data["shallow"]]

    all_vals = [v for v in deep_means + shallow_means if v > 0]
    ymin = min(all_vals) * 0.1
    ymax = max(all_vals) * 10

    lines = {}
    dots = {}

    def init():
        ax.clear()
        ax.set_facecolor("#0d1117")
        ax.set_xlim(qubit_range[0] - 0.5, qubit_range[-1] + 2)
        ax.set_ylim(ymin, ymax)
        ax.set_yscale("log")
        ax.set_xticks(qubit_range)
        ax.set_xlabel("Number of Qubits", fontsize=14, color="white",
                      labelpad=10)
        ax.set_ylabel("Gradient Variance", fontsize=14, color="white",
                      labelpad=10)
        ax.set_title(
            "The Barren Plateau\nDeep circuits kill your gradients",
            fontsize=16, color="white", fontweight="bold", pad=15,
        )
        ax.grid(True, alpha=0.12, linestyle="--")
        ax.tick_params(colors="white", labelsize=11)

        line_d, = ax.plot([], [], color="#ff4757", linewidth=3,
                          label="Deep (depth = n)", marker="o", markersize=10)
        dot_d, = ax.plot([], [], color="#ff4757", marker="o",
                         markersize=12, linestyle="none")
        line_s, = ax.plot([], [], color="#2ed573", linewidth=3,
                          label="Shallow (depth = 1)", marker="s",
                          markersize=9)
        dot_s, = ax.plot([], [], color="#2ed573", marker="s",
                         markersize=11, linestyle="none")
        lines["deep"] = line_d
        dots["deep"] = dot_d
        lines["shallow"] = line_s
        dots["shallow"] = dot_s

        ax.legend(fontsize=12, loc="upper right", framealpha=0.4)
        fig.text(0.99, 0.01, "quantum-pipeline-from-scratch",
                 ha="right", fontsize=8, color="gray", alpha=0.5)
        return [line_d, dot_d, line_s, dot_s]

    annotated = [False]

    def update(frame):
        n_pts = min(frame + 1, n_data)
        xs = qubit_range[:n_pts]

        for key, means in [("deep", deep_means), ("shallow", shallow_means)]:
            ys = means[:n_pts]
            lines[key].set_data(xs, ys)
            if n_pts > 0:
                dots[key].set_data([xs[-1]], [ys[-1]])

        if n_pts == n_data and not annotated[0]:
            d_last = deep_means[-1]
            s_last = shallow_means[-1]

            ax.annotate(
                "BARREN\nPLATEAU",
                xy=(qubit_range[-1], d_last),
                xytext=(qubit_range[-1] - 4, d_last * 0.03),
                fontsize=15, color="#ff4757", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#ff4757", lw=2.5),
            )
            ax.annotate(
                "TRAINABLE",
                xy=(qubit_range[-1], s_last),
                xytext=(qubit_range[-1] - 4, s_last * 15),
                fontsize=15, color="#2ed573", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#2ed573", lw=2.5),
            )

            if d_last > 0 and s_last > 0:
                gap = s_last / d_last
                mid_y = np.sqrt(d_last * s_last)
                ax.text(qubit_range[-1] + 0.5, mid_y, f"{gap:.0f}x",
                        fontsize=24, color="white", fontweight="bold",
                        va="center")

            annotated[0] = True

        return list(lines.values()) + list(dots.values())

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=800, blit=False)
    writer = PillowWriter(fps=1.3)
    anim.save(output_path, writer=writer, dpi=150,
              savefig_kwargs={"facecolor": "#0d1117"})
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-qubits", type=int, default=12)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--param-subset", type=int, default=5,
                        help="Number of parameters to sample for large circuits")
    args = parser.parse_args()

    if args.samples is not None:
        n_samples = args.samples
    elif args.fast:
        n_samples = 8
    else:
        n_samples = 15

    qubit_range = list(range(2, args.max_qubits + 1, 2))

    print("Computing barren plateau data...")
    print(f"  Qubits: {qubit_range}")
    print(f"  Samples per point: {n_samples}")
    print(f"  Param subset for n>10: {args.param_subset}")
    print("  Deep: depth = n  |  Shallow: depth = 1")
    print()

    t0 = time.time()
    data = compute_all_data(
        qubit_range, n_samples=n_samples,
        n_param_subset=args.param_subset,
    )
    total_time = time.time() - t0
    print(f"\nTotal computation time: {total_time:.1f}s ({total_time/60:.1f} min)")

    create_static(data, qubit_range)
    if args.gif:
        create_gif(data, qubit_range)


if __name__ == "__main__":
    main()
