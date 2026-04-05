"""Run experiments on ideal simulator, noisy simulator, and IBM Quantum hardware.

Three-way comparison:
  GREEN  — Ideal simulator (statevector, zero noise)
  ORANGE — Noisy simulator (density matrix + realistic noise model)
  RED    — Real IBM Quantum hardware

The noisy simulator uses the project's own DensityMatrix engine with
depolarizing noise calibrated to approximate IBM hardware error rates.

Setup:
    1. pip install qiskit qiskit-ibm-runtime
    2. Get your API token from https://quantum.ibm.com
    3. Export it:  export IBM_QUANTUM_TOKEN="your_token_here"

Usage:
    python experiments/ibm_runner.py --run all --local-only
    python experiments/ibm_runner.py --run grover --shots 4096
    python experiments/ibm_runner.py --run all
    python experiments/ibm_runner.py --compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsim.circuit import Circuit
from qsim.density_matrix import DensityMatrix
from qsim.measurement import sample
from qsim.noise import NoiseModel, ReadoutError, depolarizing
from qsim.statevector import Statevector
from qsim.utils import index_to_bitstring

RESULTS_DIR = Path(__file__).parent / "ibm_results"


def ibm_like_noise_model(
    p_1q: float = 0.001,
    p_2q: float = 0.01,
    p_readout: float = 0.02,
) -> tuple[NoiseModel, ReadoutError]:
    """Create a noise model approximating IBM Quantum hardware.

    Default error rates are conservative estimates for IBM Eagle/Heron
    processors (127-156 qubits, ~2024-2025 calibration data):
      - Single-qubit gate error: ~0.1% (depolarizing)
      - Two-qubit gate error:    ~1%   (depolarizing)
      - Readout error:           ~2%   (bit-flip)

    Reference: IBM Quantum system properties, publicly available at
    quantum.ibm.com for each backend.
    """
    noise = NoiseModel()

    dep_1q = depolarizing(p_1q)
    for gate_name in ["H", "X", "Y", "Z", "S", "T", "Rx", "Ry", "Rz", "Phase"]:
        noise.add_gate_noise(gate_name, dep_1q)

    dep_2q = depolarizing(p_2q)
    for gate_name in ["CNOT", "CZ", "SWAP", "CRx", "CRy", "CRz"]:
        noise.add_gate_noise(gate_name, dep_2q)

    readout = ReadoutError(p_readout)
    return noise, readout


def make_bell_circuit() -> Circuit:
    qc = Circuit(2, name="bell")
    qc.h(0).cx(0, 1)
    return qc


def make_ghz_circuit(n: int = 3) -> Circuit:
    qc = Circuit(n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def make_grover_circuit() -> Circuit:
    from experiments.grover_amplification import grover_oracle, grover_diffusion
    qc = Circuit(3, name="grover_101")
    for i in range(3):
        qc.h(i)
    for _ in range(2):
        grover_oracle(qc, "101")
        grover_diffusion(qc)
    return qc


def make_state_evolution_circuit() -> Circuit:
    qc = Circuit(4, name="state_evolution")
    for i in range(4):
        qc.h(i)
    qc.ry(np.pi / 3, 0).rz(np.pi / 4, 1).rx(np.pi / 5, 2)
    qc.cx(0, 1).cx(1, 2).cx(2, 3)
    qc.h(0).ry(2 * np.pi / 3, 3).cz(0, 3).h(2)
    return qc


EXPERIMENT_REGISTRY: dict[str, list[tuple[str, Circuit]]] = {
    "bell": [("Bell |Φ+⟩", make_bell_circuit())],
    "ghz": [("GHZ₃", make_ghz_circuit(3)), ("GHZ₅", make_ghz_circuit(5))],
    "grover": [("Grover |101⟩", make_grover_circuit())],
    "state_evolution": [("Interference 4q", make_state_evolution_circuit())],
}


def run_ideal(circuit: Circuit, shots: int = 4096, seed: int = 42) -> Counter:
    """Ideal statevector simulator — zero noise."""
    sv = Statevector.from_circuit(circuit)
    return sample(sv, shots=shots, seed=seed)


def run_noisy(
    circuit: Circuit,
    shots: int = 4096,
    seed: int = 42,
    p_1q: float = 0.001,
    p_2q: float = 0.01,
    p_readout: float = 0.02,
) -> Counter:
    """Noisy density matrix simulator with IBM-calibrated noise model."""
    noise_model, readout_error = ibm_like_noise_model(p_1q, p_2q, p_readout)

    dm = DensityMatrix.from_circuit(circuit, noise_model=noise_model)
    probs = dm.probabilities()

    probs = np.maximum(probs, 0)
    probs /= probs.sum()

    rng = np.random.default_rng(seed)
    n = circuit.num_qubits
    indices = rng.choice(2**n, size=shots, p=probs)

    counts: Counter = Counter()
    for idx in indices:
        bs = index_to_bitstring(int(idx), n)
        bs = readout_error.apply(bs, rng)
        counts[bs] += 1

    return counts


def run_ibm(
    circuit: Circuit,
    token: str | None = None,
    backend_name: str | None = None,
    shots: int = 4096,
) -> tuple[Counter, str]:
    """Submit to IBM Quantum hardware. Returns (counts, backend_name)."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    from qsim.qasm_export import to_qasm2

    token = token or os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise ValueError(
            "No IBM Quantum token. Set IBM_QUANTUM_TOKEN or pass --token."
        )

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token,
    )

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(
            min_num_qubits=circuit.num_qubits, operational=True,
        )

    actual_backend = backend.name
    print(f"    Backend: {actual_backend} ({backend.num_qubits} qubits)")

    qasm_str = to_qasm2(circuit, classical_register=True)
    qk_circuit = QuantumCircuit.from_qasm_str(qasm_str)
    qk_circuit.measure_all()

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pm.run(qk_circuit)
    print(f"    Transpiled depth: {transpiled.depth()}")

    sampler = Sampler(mode=backend)
    print(f"    Submitting {shots} shots...", end="", flush=True)
    t0 = time.time()
    job = sampler.run([transpiled], shots=shots)
    result = job.result()
    dt = time.time() - t0
    print(f" done ({dt:.1f}s)")

    raw_counts = result[0].data.meas.get_counts()

    counts = Counter({bs[::-1]: c for bs, c in raw_counts.items()})
    return counts, actual_backend


@dataclass
class ExperimentResult:
    name: str
    num_qubits: int
    num_gates: int
    shots: int
    ideal_counts: dict[str, int]
    noisy_counts: dict[str, int]
    ibm_counts: dict[str, int] | None = None
    ibm_backend: str | None = None
    noise_params: dict[str, float] | None = None
    timestamp: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_gates": self.num_gates,
            "shots": self.shots,
            "ideal_counts": self.ideal_counts,
            "noisy_counts": self.noisy_counts,
            "ibm_counts": self.ibm_counts,
            "ibm_backend": self.ibm_backend,
            "noise_params": self.noise_params,
            "timestamp": self.timestamp,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ExperimentResult:
        d = json.loads(path.read_text())
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def create_comparison(
    results: list[ExperimentResult],
    output_path: str = "experiments/ibm_comparison.png",
) -> None:
    """Three-panel comparison: ideal (green) | noisy sim (orange) | IBM (red)."""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    plt.style.use("dark_background")
    rcParams["font.family"] = "monospace"

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.8 * n))
    fig.patch.set_facecolor("#0d1117")

    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Ideal Simulator  vs  Noisy Simulator  vs  Real Quantum Hardware",
        fontsize=17, color="white", fontweight="bold", y=0.99,
    )

    panel_configs = [
        ("ideal_counts", "#2ed573", "Ideal Simulator"),
        ("noisy_counts", "#ffa502", "Noisy Simulator"),
        ("ibm_counts", "#ff4757", "IBM Quantum"),
    ]

    for row_idx, res in enumerate(results):
        all_states = set()
        for attr, _, _ in panel_configs:
            counts = getattr(res, attr, None)
            if counts:
                all_states.update(counts.keys())
        if not all_states:
            nq = res.num_qubits
            all_states = {format(i, f"0{nq}b") for i in range(2**nq)}
        all_states = sorted(all_states)

        ideal_total = sum(res.ideal_counts.values())
        ideal_probs = {s: res.ideal_counts.get(s, 0) / ideal_total
                       for s in all_states}

        y_max = 0
        for col_idx, (attr, color, source_label) in enumerate(panel_configs):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("#0d1117")

            counts = getattr(res, attr, None)

            if counts:
                total = sum(counts.values())
                probs = {s: counts.get(s, 0) / total for s in all_states}
                vals = [probs[s] for s in all_states]
                y_max = max(y_max, max(vals) * 1.3)

                ax.bar(range(len(all_states)), vals,
                       color=color, alpha=0.85, width=0.7,
                       edgecolor=color, linewidth=0.5)

                if attr == "ibm_counts" and res.ibm_backend:
                    source_label = res.ibm_backend

                if attr != "ideal_counts":
                    fidelity = sum(
                        np.sqrt(ideal_probs[s] * probs[s])
                        for s in all_states
                    )
                    ax.text(
                        0.97, 0.93,
                        f"Fidelity: {fidelity:.3f}",
                        transform=ax.transAxes, fontsize=9,
                        color="white", ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="#1a1a2e", ec=color, alpha=0.85),
                    )

                ax.set_title(f"{res.name}  —  {source_label}",
                             fontsize=10, color=color, pad=8)
            else:
                ax.text(
                    0.5, 0.5,
                    "Awaiting IBM results\n\nSet IBM_QUANTUM_TOKEN\nand re-run",
                    transform=ax.transAxes, fontsize=11,
                    color="gray", ha="center", va="center",
                )
                ax.set_title(f"{res.name}  —  {source_label} (pending)",
                             fontsize=10, color="gray", pad=8)

            ax.set_xticks(range(len(all_states)))
            ax.set_xticklabels(
                [f"|{s}⟩" for s in all_states],
                fontsize=6, color="gray", rotation=45, ha="right",
            )
            if col_idx == 0:
                ax.set_ylabel("Probability", fontsize=9, color="white")
            ax.tick_params(colors="gray", labelsize=7)
            ax.grid(True, axis="y", alpha=0.08, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if y_max > 0:
            for col_idx in range(3):
                axes[row_idx][col_idx].set_ylim(0, y_max)

    fig.text(
        0.99, 0.005,
        "quantum-pipeline-from-scratch  ·  zero dependencies  ·  "
        "density matrix noise model",
        ha="right", fontsize=7, color="gray", alpha=0.4,
    )

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="IBM Quantum experiment runner")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--run", nargs="+",
                        choices=["all", "bell", "ghz", "grover",
                                 "state_evolution"])
    parser.add_argument("--local-only", action="store_true",
                        help="Skip IBM hardware (ideal + noisy sim only)")
    parser.add_argument("--compare", action="store_true",
                        help="Regenerate comparison from saved results")
    parser.add_argument("--p-1q", type=float, default=0.001,
                        help="Single-qubit depolarizing error rate")
    parser.add_argument("--p-2q", type=float, default=0.01,
                        help="Two-qubit depolarizing error rate")
    parser.add_argument("--p-readout", type=float, default=0.02,
                        help="Readout bit-flip error rate")
    args = parser.parse_args()

    if args.compare:
        result_files = sorted(RESULTS_DIR.glob("*.json"))
        if not result_files:
            print("No saved results. Run experiments first.")
            return
        results = [ExperimentResult.load(f) for f in result_files]
        create_comparison(results)
        return

    if not args.run:
        parser.print_help()
        return

    if "all" in args.run:
        experiment_names = list(EXPERIMENT_REGISTRY.keys())
    else:
        experiment_names = args.run

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[ExperimentResult] = []
    noise_params = {"p_1q": args.p_1q, "p_2q": args.p_2q,
                    "p_readout": args.p_readout}

    for exp_name in experiment_names:
        circuits = EXPERIMENT_REGISTRY[exp_name]
        for label, circuit in circuits:
            print(f"\n{'='*60}")
            print(f"  {label}  ({circuit.num_qubits} qubits, "
                  f"{len(circuit)} gates)")
            print(f"{'='*60}")

            print("  [green]  Ideal simulator...", end="", flush=True)
            t0 = time.time()
            ideal_counts = run_ideal(circuit, shots=args.shots)
            print(f" done ({time.time()-t0:.3f}s)")
            for bs, c in ideal_counts.most_common(3):
                print(f"    |{bs}⟩: {c/args.shots:.1%}")

            print(f"  [orange] Noisy simulator (p_1q={args.p_1q}, "
                  f"p_2q={args.p_2q}, readout={args.p_readout})...",
                  end="", flush=True)
            t0 = time.time()
            noisy_counts = run_noisy(
                circuit, shots=args.shots,
                p_1q=args.p_1q, p_2q=args.p_2q,
                p_readout=args.p_readout,
            )
            print(f" done ({time.time()-t0:.2f}s)")
            for bs, c in noisy_counts.most_common(3):
                print(f"    |{bs}⟩: {c/args.shots:.1%}")

            ibm_counts = None
            ibm_backend = None
            if not args.local_only:
                try:
                    print("  [red]    IBM Quantum...")
                    ibm_counts, ibm_backend = run_ibm(
                        circuit, token=args.token,
                        backend_name=args.backend, shots=args.shots,
                    )
                    for bs, c in ibm_counts.most_common(3):
                        print(f"    |{bs}⟩: {c/args.shots:.1%}")
                except Exception as e:
                    print(f"    IBM error: {e}")

            from datetime import datetime
            result = ExperimentResult(
                name=label,
                num_qubits=circuit.num_qubits,
                num_gates=len(circuit),
                shots=args.shots,
                ideal_counts=dict(ideal_counts),
                noisy_counts=dict(noisy_counts),
                ibm_counts=dict(ibm_counts) if ibm_counts else None,
                ibm_backend=ibm_backend,
                noise_params=noise_params,
                timestamp=datetime.now().isoformat(),
            )
            safe = label.lower().replace(" ", "_")
            for ch in "|⟩+₃₅":
                safe = safe.replace(ch, "")
            result.save(RESULTS_DIR / f"{safe}.json")
            all_results.append(result)

    if all_results:
        create_comparison(all_results)


if __name__ == "__main__":
    main()
