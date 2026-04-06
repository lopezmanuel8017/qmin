# qmin

![Ideal Simulator vs Noisy Simulator vs IBM Quantum Hardware](experiments/ibm_comparison.png)

## Project structure

```
qmin/
├── qsim/                        # Quantum simulator core
│   ├── statevector.py           #   State vector simulator (tensor contraction)
│   ├── density_matrix.py        #   Density matrix simulator (mixed states + noise)
│   ├── gates.py                 #   Gate definitions (H, X, Y, Z, CNOT, Toffoli, etc.)
│   ├── circuit.py               #   Circuit construction and execution
│   ├── measurement.py           #   Measurement and sampling
│   ├── noise.py                 #   Noise channels (depolarizing, readout error)
│   ├── gradient.py              #   Parameter-shift gradient computation
│   ├── observables.py           #   Observable expectation values
│   ├── parameters.py            #   Trainable parameter management
│   ├── qasm_export.py           #   OpenQASM export (for IBM hardware)
│   └── utils.py                 #   Fidelity, partial trace, helpers
│
├── quantum/                     # Quantum ML components
│   ├── ansatz.py                #   Variational circuit ansatze
│   ├── attention.py             #   Quantum attention mechanism
│   ├── encoding.py              #   Classical-to-quantum data encoding
│   ├── kernel.py                #   Quantum kernel methods
│   └── diagnostics.py           #   Barren plateau analysis, expressibility
│
├── classical/                   # Classical ML components (from-scratch)
│   ├── layers.py                #   Linear, Conv2d (im2col), BatchNorm
│   ├── loss.py                  #   Cross-entropy, MSE
│   ├── optim.py                 #   SGD, Adam
│   └── detection_head.py        #   Object detection head
│
├── pipeline/                    # Hybrid quantum-classical pipeline
│   ├── hybrid_layer.py          #   Quantum layer with classical interface
│   ├── classifier.py            #   MNIST hybrid classifier
│   ├── detector.py              #   Object detection pipeline
│   ├── distillation.py          #   Knowledge distillation
│   ├── quantum_reranker.py      #   Quantum re-ranking module
│   └── trainer.py               #   Training loop
│
├── data/                        # Data loading
│   ├── mnist.py                 #   MNIST loader
│   └── kitti.py                 #   KITTI dataset loader
│
├── experiments/                 # Reproducible experiments & visualizations
│   ├── ibm_runner.py            #   Run circuits on IBM Quantum hardware
│   ├── barren_plateau_depth.py  #   Barren plateau gradient variance study
│   ├── grover_amplification.py  #   Grover's algorithm visualization
│   ├── state_evolution.py       #   Quantum state evolution animation
│   └── training_demo.py         #   Hybrid classifier training demo
│
├── tests/                       # 589 tests, 100% branch coverage
├── pyproject.toml
└── README.md
```
