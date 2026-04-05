"""Tests for quantum/diagnostics.py — barren plateau detection."""

from quantum.ansatz import CNOTLadder
from quantum.diagnostics import estimate_gradient_variance


class TestEstimateGradientVariance:
    def test_runs_and_returns_structure(self):
        ansatz = CNOTLadder(2, 1)
        result = estimate_gradient_variance(ansatz, n_samples=5, seed=0)
        assert "mean_grad_variance" in result
        assert "per_param_variance" in result
        assert "num_samples" in result
        assert result["num_samples"] == 5
        assert len(result["per_param_variance"]) == ansatz.num_parameters

    def test_variance_is_non_negative(self):
        ansatz = CNOTLadder(2, 1)
        result = estimate_gradient_variance(ansatz, n_samples=10, seed=0)
        assert result["mean_grad_variance"] >= 0
        for var in result["per_param_variance"].values():
            assert var >= 0

    def test_variance_decreases_with_qubit_count_uniform(self):
        """Uniform init: gradient variance should decrease with more qubits.

        This is the signature of barren plateaus (McClean et al. 2018).
        """
        var_2q = estimate_gradient_variance(
            CNOTLadder(2, 2), strategy="uniform", n_samples=30, seed=0,
        )["mean_grad_variance"]
        var_5q = estimate_gradient_variance(
            CNOTLadder(5, 2), strategy="uniform", n_samples=30, seed=0,
        )["mean_grad_variance"]
        assert var_5q < var_2q

    def test_identity_block_gradients_are_consistent(self):
        """Identity-block produces low-variance (consistent) gradients."""
        ansatz = CNOTLadder(3, 2)
        result = estimate_gradient_variance(
            ansatz, strategy="identity_block", n_samples=20, seed=0,
        )
        assert result["mean_grad_variance"] < 0.01

    def test_custom_observable(self):
        from qsim.observables import Observable
        ansatz = CNOTLadder(2, 1)
        obs = Observable.x(0)
        result = estimate_gradient_variance(ansatz, observable=obs, n_samples=5, seed=0)
        assert result["mean_grad_variance"] >= 0

    def test_different_seeds_give_different_results(self):
        ansatz = CNOTLadder(2, 1)
        r1 = estimate_gradient_variance(ansatz, n_samples=10, seed=0)
        r2 = estimate_gradient_variance(ansatz, n_samples=10, seed=100)
        assert r1["mean_grad_variance"] != r2["mean_grad_variance"]
