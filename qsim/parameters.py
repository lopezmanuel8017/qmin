"""Symbolic parameters for parameterized quantum circuits.

Parameters use UUID-based identity, not name-based. Two Parameter("theta")
objects are distinct, preventing collisions when composing circuits from
different sources. This matches the pattern used by Qiskit.
"""

from __future__ import annotations

import uuid
from typing import Union


class Parameter:
    """A named symbolic parameter for parameterized circuits."""

    __slots__ = ("_name", "_uuid")

    def __init__(self, name: str) -> None:
        self._name = name
        self._uuid = uuid.uuid4()

    @property
    def name(self) -> str:
        return self._name

    def __hash__(self) -> int:
        return hash(self._uuid)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parameter):
            return NotImplemented
        return self._uuid == other._uuid

    def __repr__(self) -> str:
        return f"Parameter('{self._name}')"


class ParameterVector:
    """A named vector of Parameters, for convenience.

    Usage:
        theta = ParameterVector("theta", 4)
        circuit.ry(theta[0], qubit=0)
    """

    __slots__ = ("_name", "_params")

    def __init__(self, name: str, length: int) -> None:
        if length < 0:
            raise ValueError(f"Length must be >= 0, got {length}")
        self._name = name
        self._params = tuple(Parameter(f"{name}[{i}]") for i in range(length))

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, index: int) -> Parameter:
        return self._params[index]

    def __len__(self) -> int:
        return len(self._params)

    def __iter__(self):
        return iter(self._params)

    def __repr__(self) -> str:
        return f"ParameterVector('{self._name}', {len(self._params)})"


ParameterValueType = Union[float, Parameter]
