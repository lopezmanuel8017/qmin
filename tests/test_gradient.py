"""Tests for qsim/gradient.py — parameter-shift rule and gradient validation."""

import numpy as np

from qsim.circuit import Circuit
from qsim.gradient import (
    compute_cost_and_gradient,
    numerical_gradient,
    parameter_shift_gradient,
)
from qsim.observables import Observable
from qsim.parameters import Parameter, ParameterVector


class TestParameterShiftGradient:
    def test_rx_gradient_analytic(self):
        """For Rx(θ)|0>, d<Z>/dθ = -sin(θ).

        Derivation:
          |ψ(θ)> = Rx(θ)|0> = cos(θ/2)|0> - i·sin(θ/2)|1>
          <Z> = cos²(θ/2) - sin²(θ/2) = cos(θ)
          d<Z>/dθ = -sin(θ)
        """
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        obs = Observable.z(0)

        for angle in [0.0, 0.3, np.pi / 4, np.pi / 2, np.pi, 1.7]:
            grad = parameter_shift_gradient(qc, obs, {theta: angle})
            expected = -np.sin(angle)
            assert np.isclose(grad[theta], expected, atol=1e-10), (
                f"At θ={angle}: got {grad[theta]}, expected {expected}"
            )

    def test_ry_gradient_analytic(self):
        """For Ry(θ)|0>, d<Z>/dθ = -sin(θ).

        |ψ(θ)> = cos(θ/2)|0> + sin(θ/2)|1>
        <Z> = cos²(θ/2) - sin²(θ/2) = cos(θ)
        d<Z>/dθ = -sin(θ)
        """
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.ry(theta, 0)
        obs = Observable.z(0)

        grad = parameter_shift_gradient(qc, obs, {theta: np.pi / 3})
        expected = -np.sin(np.pi / 3)
        assert np.isclose(grad[theta], expected, atol=1e-10)

    def test_rz_gradient_z_observable(self):
        """For Rz(θ)|0>, <Z> = 1 (constant), so d<Z>/dθ = 0.

        Rz only adds phase to |0> and |1>. Since |0> has amplitude 1,
        the probabilities don't change, and <Z> = 1 always.
        """
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rz(theta, 0)
        obs = Observable.z(0)

        grad = parameter_shift_gradient(qc, obs, {theta: 1.5})
        assert np.isclose(grad[theta], 0.0, atol=1e-10)


class TestMultiParameterGradient:
    def test_two_param_circuit(self):
        """Two independent rotations: gradient of each is independent."""
        a = Parameter("a")
        b = Parameter("b")
        qc = Circuit(2)
        qc.rx(a, 0).ry(b, 1)

        obs_a = Observable.z(0)
        grad_a = parameter_shift_gradient(qc, obs_a, {a: 0.5, b: 1.0})
        assert np.isclose(grad_a[a], -np.sin(0.5), atol=1e-10)
        assert np.isclose(grad_a[b], 0.0, atol=1e-10)

    def test_entangled_circuit(self):
        """Gradient through CNOT: both parameters affect the output."""
        a = Parameter("a")
        b = Parameter("b")
        qc = Circuit(2)
        qc.ry(a, 0).cx(0, 1).ry(b, 1)
        obs = Observable.z(1)

        vals = {a: 0.7, b: 1.2}
        ps_grad = parameter_shift_gradient(qc, obs, vals)
        num_grad = numerical_gradient(qc, obs, vals)

        for param in [a, b]:
            assert np.isclose(ps_grad[param], num_grad[param], atol=1e-5), (
                f"Param {param.name}: PS={ps_grad[param]}, Num={num_grad[param]}"
            )

    def test_parameter_vector(self):
        pv = ParameterVector("theta", 3)
        qc = Circuit(3)
        for i in range(3):
            qc.ry(pv[i], i)
        obs = Observable.z(0) + Observable.z(1) + Observable.z(2)

        vals = {pv[i]: 0.5 * (i + 1) for i in range(3)}
        ps_grad = parameter_shift_gradient(qc, obs, vals)
        num_grad = numerical_gradient(qc, obs, vals)

        for i in range(3):
            assert np.isclose(ps_grad[pv[i]], num_grad[pv[i]], atol=1e-5)


class TestNumericalGradient:
    def test_rx_matches_analytic(self):
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        obs = Observable.z(0)

        grad = numerical_gradient(qc, obs, {theta: 1.0})
        expected = -np.sin(1.0)
        assert np.isclose(grad[theta], expected, atol=1e-5)


class TestComputeCostAndGradient:
    def test_returns_both(self):
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        obs = Observable.z(0)

        angle = np.pi / 4
        cost, grad = compute_cost_and_gradient(qc, obs, {theta: angle})

        assert np.isclose(cost, np.cos(angle), atol=1e-10)
        assert np.isclose(grad[theta], -np.sin(angle), atol=1e-10)

    def test_consistency(self):
        """Cost from compute_cost_and_gradient matches standalone evaluation."""
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.ry(theta, 0)
        obs = Observable.z(0)

        cost, _ = compute_cost_and_gradient(qc, obs, {theta: 0.8})

        from qsim.measurement import expectation_value
        from qsim.statevector import Statevector

        bound = qc.bind_parameters({theta: 0.8})
        sv = Statevector.from_circuit(bound)
        standalone_cost = expectation_value(sv, obs)

        assert np.isclose(cost, standalone_cost)


class TestParameterShiftVsNumerical:
    """Comprehensive comparison across various circuit structures."""

    def test_deep_circuit(self):
        """Multi-layer circuit: parameter-shift must match numerical."""
        params = [Parameter(f"p{i}") for i in range(6)]
        qc = Circuit(2)
        qc.ry(params[0], 0).ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 0).ry(params[3], 1)
        qc.cx(1, 0)
        qc.ry(params[4], 0).ry(params[5], 1)

        obs = Observable.z(0) + Observable.z(1)
        vals = {p: np.random.default_rng(42 + i).uniform(0, 2 * np.pi)
                for i, p in enumerate(params)}

        ps_grad = parameter_shift_gradient(qc, obs, vals)
        num_grad = numerical_gradient(qc, obs, vals)

        for p in params:
            assert np.isclose(ps_grad[p], num_grad[p], atol=1e-5), (
                f"{p.name}: PS={ps_grad[p]:.8f}, Num={num_grad[p]:.8f}"
            )

    def test_repeated_parameter(self):
        """Same parameter used in multiple gates."""
        theta = Parameter("theta")
        qc = Circuit(2)
        qc.ry(theta, 0).ry(theta, 1)
        obs = Observable.z(0) + Observable.z(1)

        vals = {theta: 1.0}
        ps_grad = parameter_shift_gradient(qc, obs, vals)
        num_grad = numerical_gradient(qc, obs, vals)

        assert np.isclose(ps_grad[theta], num_grad[theta], atol=1e-5)
