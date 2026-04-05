"""Tests for qsim/parameters.py — symbolic parameters."""

import pytest

from qsim.parameters import Parameter, ParameterVector


class TestParameter:
    def test_name(self):
        p = Parameter("theta")
        assert p.name == "theta"

    def test_repr(self):
        p = Parameter("phi")
        assert repr(p) == "Parameter('phi')"

    def test_two_same_name_are_distinct(self):
        """UUID-based identity: same name doesn't mean same parameter."""
        p1 = Parameter("theta")
        p2 = Parameter("theta")
        assert p1 != p2
        assert hash(p1) != hash(p2)

    def test_self_equality(self):
        p = Parameter("x")
        assert p == p

    def test_hash_is_consistent(self):
        p = Parameter("x")
        assert hash(p) == hash(p)

    def test_inequality_with_non_parameter(self):
        p = Parameter("x")
        assert p != "x"
        assert p != 42

    def test_usable_as_dict_key(self):
        p1 = Parameter("a")
        p2 = Parameter("b")
        d = {p1: 1.0, p2: 2.0}
        assert d[p1] == 1.0
        assert d[p2] == 2.0

    def test_usable_in_set(self):
        p1 = Parameter("a")
        p2 = Parameter("a")
        s = {p1, p2}
        assert len(s) == 2


class TestParameterVector:
    def test_length(self):
        pv = ParameterVector("theta", 4)
        assert len(pv) == 4

    def test_indexing(self):
        pv = ParameterVector("theta", 3)
        assert pv[0].name == "theta[0]"
        assert pv[1].name == "theta[1]"
        assert pv[2].name == "theta[2]"

    def test_iteration(self):
        pv = ParameterVector("x", 3)
        names = [p.name for p in pv]
        assert names == ["x[0]", "x[1]", "x[2]"]

    def test_elements_are_distinct(self):
        pv = ParameterVector("x", 3)
        assert pv[0] != pv[1]
        assert pv[1] != pv[2]

    def test_name_property(self):
        pv = ParameterVector("weights", 2)
        assert pv.name == "weights"

    def test_repr(self):
        pv = ParameterVector("w", 5)
        assert repr(pv) == "ParameterVector('w', 5)"

    def test_zero_length(self):
        pv = ParameterVector("empty", 0)
        assert len(pv) == 0
        assert list(pv) == []

    def test_negative_length_raises(self):
        with pytest.raises(ValueError, match="Length must be >= 0"):
            ParameterVector("bad", -1)

    def test_out_of_range_raises(self):
        pv = ParameterVector("x", 2)
        with pytest.raises(IndexError):
            _ = pv[5]
