"""Tests for quantum/kernel.py — quantum kernel similarity."""

import numpy as np

from quantum.kernel import QuantumKernel


class TestQuantumKernel:
    def test_identical_features_fidelity_one(self):
        """K(x, x) = |<psi(x)|psi(x)>|^2 = 1.0."""
        kernel = QuantumKernel(num_qubits=2)
        x = np.array([0.5, 1.2])
        val = kernel.compute_entry(x, x)
        np.testing.assert_allclose(val, 1.0, atol=1e-10)

    def test_symmetry(self):
        """K(x, y) = K(y, x)."""
        kernel = QuantumKernel(num_qubits=2)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2,))
        y = rng.normal(size=(2,))
        np.testing.assert_allclose(
            kernel.compute_entry(x, y),
            kernel.compute_entry(y, x),
            atol=1e-10,
        )

    def test_bounded_zero_to_one(self):
        """Fidelity is always in [0, 1]."""
        kernel = QuantumKernel(num_qubits=3)
        rng = np.random.default_rng(1)
        for _ in range(10):
            x = rng.normal(size=(3,))
            y = rng.normal(size=(3,))
            val = kernel.compute_entry(x, y)
            assert 0.0 - 1e-10 <= val <= 1.0 + 1e-10

    def test_matrix_shape(self):
        kernel = QuantumKernel(num_qubits=2)
        features = np.random.default_rng(2).normal(size=(4, 2))
        K = kernel.compute_matrix(features)
        assert K.shape == (4, 4)

    def test_matrix_diagonal_is_one(self):
        kernel = QuantumKernel(num_qubits=2)
        features = np.random.default_rng(3).normal(size=(3, 2))
        K = kernel.compute_matrix(features)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_matrix_symmetric(self):
        kernel = QuantumKernel(num_qubits=2)
        features = np.random.default_rng(4).normal(size=(4, 2))
        K = kernel.compute_matrix(features)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_matrix_positive_semi_definite(self):
        """Kernel matrix should be PSD (all eigenvalues >= 0)."""
        kernel = QuantumKernel(num_qubits=2)
        features = np.random.default_rng(5).normal(size=(5, 2))
        K = kernel.compute_matrix(features)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8)

    def test_sync_from_array(self):
        kernel = QuantumKernel(num_qubits=2)
        kernel._param_array[:] = 0.0
        kernel.sync_from_array()
        for p in kernel._trainable_params:
            assert kernel._param_values[p] == 0.0

    def test_parameters(self):
        kernel = QuantumKernel(num_qubits=2)
        params = kernel.parameters()
        assert len(params) == 1
        assert params[0][0] is kernel._param_array
