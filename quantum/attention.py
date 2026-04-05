"""Quantum self-attention mechanism (HQViT-style).

Implements quantum self-attention where attention coefficients are computed
via parameterized quantum circuits with O(log₂N) qubits. This replaces
the O(n²) classical attention weight matrix with quantum inner products.

Architecture:
  1. Input features are projected to query/key/value
  2. Query-key attention scores are computed via quantum circuits
  3. Scores are used to weight values (classical weighted sum)

Encoding modes:
  - "reuploading": Data re-uploading (Perez-Salinas et al. 2020). Multiple layers
    of encoding interleaved with variational layers. Encodes all feature dimensions.
  - "projection": Learned linear compression W_compress: (feature_dim,) -> (n_qubits,).
  - "truncate": Legacy mode. Truncates interaction to first num_qubits elements.

Reference:
  - HQViT: Hybrid Quantum Vision Transformer (2025)
  - Perez-Salinas et al., Data re-uploading for a universal quantum classifier (2020)

For N features, we use ceil(log₂(N)) qubits to encode the attention pattern.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from qsim.circuit import Circuit
from qsim.gradient import parameter_shift_gradient
from qsim.measurement import expectation_value
from qsim.observables import Observable
from qsim.parameters import Parameter
from qsim.statevector import Statevector

_VALID_ENCODING_MODES = ("reuploading", "projection", "truncate")


class QuantumSelfAttention:
    """Quantum self-attention layer.

    Uses parameterized quantum circuits to compute attention weights.
    For sequence_length N, uses ceil(log₂(N)) qubits.
    """

    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        num_heads: int = 1,
        init_strategy: str = "uniform",
        init_epsilon: float = 0.01,
        encoding_mode: str = "reuploading",
        max_reuploading_layers: int = 8,
    ) -> None:
        if encoding_mode not in _VALID_ENCODING_MODES:
            raise ValueError(
                f"Unknown encoding_mode: {encoding_mode!r}. "
                f"Choose from {_VALID_ENCODING_MODES}"
            )

        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.encoding_mode = encoding_mode

        self.num_qubits = max(2, math.ceil(math.log2(sequence_length)))
        n = self.num_qubits

        rng = np.random.default_rng(42)
        scale = np.sqrt(2.0 / feature_dim)
        self.W_q = rng.normal(0, scale, (feature_dim, feature_dim))
        self.W_k = rng.normal(0, scale, (feature_dim, feature_dim))
        self.W_v = rng.normal(0, scale, (feature_dim, feature_dim))
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)

        if encoding_mode == "reuploading":
            self.num_reuploading_layers = min(
                math.ceil(feature_dim / n), max_reuploading_layers,
            )
            self._num_encoding_dims = self.num_reuploading_layers * n
        elif encoding_mode == "projection":
            self.num_reuploading_layers = 1
            self._num_encoding_dims = n
            self.W_compress = rng.normal(0, scale, (n, feature_dim))
            self.grad_W_compress = np.zeros_like(self.W_compress)
        else:
            self.num_reuploading_layers = 1
            self._num_encoding_dims = n

        n_quantum_params = self.num_reuploading_layers * n * 2
        self._quantum_params = [
            Parameter(f"qattn_L{layer}_q{qubit}_{gate}")
            for layer in range(self.num_reuploading_layers)
            for qubit in range(n)
            for gate in ("ry", "rz")
        ]
        assert len(self._quantum_params) == n_quantum_params

        if init_strategy == "uniform":
            self._param_values = {p: rng.uniform(0, 2 * np.pi) for p in self._quantum_params}
        elif init_strategy == "identity_block":
            self._param_values = {p: rng.normal(0, init_epsilon) for p in self._quantum_params}
        elif init_strategy == "small_random":
            s = np.pi / (2 * np.sqrt(n_quantum_params))
            self._param_values = {p: rng.normal(0, s) for p in self._quantum_params}
        else:
            raise ValueError(f"Unknown init strategy: {init_strategy!r}")

        self._param_array = np.array([self._param_values[p] for p in self._quantum_params])
        self._grad_array = np.zeros(n_quantum_params)

        self._encoding_params = [
            Parameter(f"enc_{i}") for i in range(self._num_encoding_dims)
        ]

        self._score_circuit = self._build_score_circuit()
        self._obs = Observable.z(0)

        self._input: Optional[np.ndarray] = None
        self._queries: Optional[np.ndarray] = None
        self._keys: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._attn_weights: Optional[np.ndarray] = None

    def _build_score_circuit(self) -> Circuit:
        """Build a fully parameterized circuit for attention scoring.

        For reuploading: multiple layers of (encode + variational + entangle).
        For truncate/projection: single layer.
        """
        n = self.num_qubits
        qc = Circuit(n, name="attn_score")

        enc_idx = 0
        var_idx = 0
        for _layer in range(self.num_reuploading_layers):
            for i in range(n):
                qc.ry(self._encoding_params[enc_idx], i)
                enc_idx += 1

            for i in range(n):
                qc.ry(self._quantum_params[var_idx], i)
                qc.rz(self._quantum_params[var_idx + 1], i)
                var_idx += 2

            for i in range(n - 1):
                qc.cx(i, i + 1)

        return qc

    def _interaction_vector(self, query: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Compute the query-key interaction vector for encoding.

        Returns an array of length self._num_encoding_dims.
        """
        n = self.num_qubits

        if self.encoding_mode == "projection":
            interaction = query * key
            return self.W_compress @ interaction

        if self.encoding_mode == "truncate":
            if len(query) >= n:
                return query[:n] * key[:n]
            interaction = query * key
            return np.pad(interaction, (0, max(0, n - len(interaction))))[:n]

        interaction = query * key
        target_len = self._num_encoding_dims
        if len(interaction) < target_len:
            interaction = np.pad(interaction, (0, target_len - len(interaction)))
        return interaction[:target_len]

    def _quantum_attention_score(
        self, query: np.ndarray, key: np.ndarray
    ) -> float:
        """Compute attention score between query and key using quantum circuit."""
        interaction = self._interaction_vector(query, key)

        bindings: dict[Parameter, float] = {}
        for i, enc_p in enumerate(self._encoding_params):
            bindings[enc_p] = float(interaction[i])
        bindings.update(self._param_values)

        bound = self._score_circuit.bind_parameters(bindings)
        sv = Statevector.from_circuit(bound)
        return expectation_value(sv, self._obs)

    def _quantum_score_gradients(
        self, query: np.ndarray, key: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradients of a single attention score via parameter-shift.

        Returns:
            d_score_d_interaction: (num_encoding_dims,) gradient w.r.t. encoding angles
            d_score_d_params: (n_quantum_params,) gradient w.r.t. variational params
        """
        interaction = self._interaction_vector(query, key)

        bindings: dict[Parameter, float] = {}
        for i, enc_p in enumerate(self._encoding_params):
            bindings[enc_p] = float(interaction[i])
        bindings.update(self._param_values)

        grads = parameter_shift_gradient(self._score_circuit, self._obs, bindings)

        d_score_d_interaction = np.array([
            grads[p] for p in self._encoding_params
        ])
        d_score_d_params = np.array([
            grads[p] for p in self._quantum_params
        ])
        return d_score_d_interaction, d_score_d_params

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        x: (sequence_length, feature_dim) -- single sample, no batch dim.
        Returns: (sequence_length, feature_dim)
        """
        self._input = x
        seq_len = x.shape[0]

        self._queries = x @ self.W_q.T
        self._keys = x @ self.W_k.T
        self._values = x @ self.W_v.T

        attn_scores = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                attn_scores[i, j] = self._quantum_attention_score(
                    self._queries[i], self._keys[j]
                )

        shifted = attn_scores - attn_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self._attn_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        output = self._attn_weights @ self._values

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass with full gradient flow through quantum circuit.

        Computes gradients for:
          - W_v: through value path (classical)
          - W_q, W_k: through attention score path via parameter-shift
          - quantum params: through attention score path via parameter-shift
          - W_compress: if encoding_mode == "projection"
        """
        seq_len = self._attn_weights.shape[0]
        n = self.num_qubits

        grad_values = self._attn_weights.T @ grad_output
        grad_attn = grad_output @ self._values.T

        grad_scores = np.zeros_like(self._attn_weights)
        for i in range(seq_len):
            p = self._attn_weights[i]
            dp = grad_attn[i]
            grad_scores[i] = p * (dp - np.sum(dp * p))

        self.grad_W_v = grad_values.T @ self._input

        grad_queries = np.zeros_like(self._queries)
        grad_keys = np.zeros_like(self._keys)
        self._grad_array[:] = 0.0
        if self.encoding_mode == "projection":
            self.grad_W_compress = np.zeros_like(self.W_compress)

        for i in range(seq_len):
            for j in range(seq_len):
                gs = grad_scores[i, j]
                if abs(gs) < 1e-12:
                    continue

                d_score_d_enc, d_score_d_params = self._quantum_score_gradients(
                    self._queries[i], self._keys[j]
                )

                self._grad_array += gs * d_score_d_params

                if self.encoding_mode == "projection":
                    interaction_raw = self._queries[i] * self._keys[j]
                    grad_enc = gs * d_score_d_enc
                    grad_raw = self.W_compress.T @ grad_enc
                    grad_queries[i] += grad_raw * self._keys[j]
                    grad_keys[j] += grad_raw * self._queries[i]
                    self.grad_W_compress += np.outer(grad_enc, interaction_raw)
                elif self.encoding_mode == "truncate":
                    grad_interaction = gs * d_score_d_enc
                    grad_queries[i, :n] += grad_interaction * self._keys[j, :n]
                    grad_keys[j, :n] += grad_interaction * self._queries[i, :n]
                else:
                    n_enc = self._num_encoding_dims
                    fd = self.feature_dim
                    grad_enc_full = gs * d_score_d_enc
                    effective = min(fd, n_enc)
                    grad_queries[i, :effective] += grad_enc_full[:effective] * self._keys[j, :effective]
                    grad_keys[j, :effective] += grad_enc_full[:effective] * self._queries[i, :effective]

        self.grad_W_q = grad_queries.T @ self._input
        self.grad_W_k = grad_keys.T @ self._input

        grad_input = (
            grad_values @ self.W_v
            + grad_queries @ self.W_q
            + grad_keys @ self.W_k
        )
        return grad_input

    def sync_from_array(self) -> None:
        """Update internal parameter dict from the persistent flat array."""
        for i, p in enumerate(self._quantum_params):
            self._param_values[p] = float(self._param_array[i])

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = [
            (self.W_q, self.grad_W_q),
            (self.W_k, self.grad_W_k),
            (self.W_v, self.grad_W_v),
            (self._param_array, self._grad_array),
        ]
        if self.encoding_mode == "projection":
            params.append((self.W_compress, self.grad_W_compress))
        return params
