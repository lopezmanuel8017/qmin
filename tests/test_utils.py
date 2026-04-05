"""Tests for qsim/utils.py — qubit ordering and validation."""

import pytest

from qsim.utils import (
    QUBIT_ORDER,
    computational_basis_index,
    index_to_bitstring,
    validate_qubit_indices,
)


class TestQubitOrder:
    def test_convention_is_big_endian(self):
        assert QUBIT_ORDER == "big-endian"


class TestValidateQubitIndices:
    def test_valid_single_qubit(self):
        validate_qubit_indices((0,), 1)

    def test_valid_multiple_qubits(self):
        validate_qubit_indices((0, 1, 2), 4)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_qubit_indices((3,), 3)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_qubit_indices((-1,), 2)

    def test_duplicate_indices_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            validate_qubit_indices((1, 1), 3)

    def test_zero_qubits_raises(self):
        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            validate_qubit_indices((0,), 0)

    def test_empty_qubits_tuple_valid(self):
        validate_qubit_indices((), 2)


class TestComputationalBasisIndex:
    def test_single_zero(self):
        assert computational_basis_index("0") == 0

    def test_single_one(self):
        assert computational_basis_index("1") == 1

    def test_multi_bit(self):
        assert computational_basis_index("101") == 5

    def test_all_zeros(self):
        assert computational_basis_index("000") == 0

    def test_all_ones(self):
        assert computational_basis_index("111") == 7

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            computational_basis_index("")

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError, match="Invalid bitstring"):
            computational_basis_index("012")

    def test_two_bit(self):
        assert computational_basis_index("10") == 2
        assert computational_basis_index("01") == 1
        assert computational_basis_index("11") == 3


class TestIndexToBitstring:
    def test_zero_one_qubit(self):
        assert index_to_bitstring(0, 1) == "0"

    def test_one_one_qubit(self):
        assert index_to_bitstring(1, 1) == "1"

    def test_five_three_qubits(self):
        assert index_to_bitstring(5, 3) == "101"

    def test_zero_three_qubits(self):
        assert index_to_bitstring(0, 3) == "000"

    def test_seven_three_qubits(self):
        assert index_to_bitstring(7, 3) == "111"

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            index_to_bitstring(-1, 2)

    def test_index_too_large_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            index_to_bitstring(4, 2)

    def test_roundtrip(self):
        """computational_basis_index and index_to_bitstring are inverses."""
        for n in range(1, 5):
            for idx in range(2**n):
                bs = index_to_bitstring(idx, n)
                assert computational_basis_index(bs) == idx
