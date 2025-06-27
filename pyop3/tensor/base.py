from __future__ import annotations

import abc
import dataclasses
import numbers
from collections.abc import Iterable
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from pyop3 import utils
from pyop3.axtree import ContextAware
from pyop3.axtree.tree import AbstractAxisTree, Expression
from pyop3.exceptions import InvalidIndexCountException
from pyop3.lang import FunctionArgument, ArrayAssignment


class Tensor(ContextAware, FunctionArgument, Expression, abc.ABC):

    DEFAULT_PREFIX: ClassVar[str] = "array"

    def __init__(self, name: str | None=None, *, prefix: str | None=None, parent: Array|None=None) -> None:
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parent", parent)

    def __getitem__(self, indices):
        # Handle the fact that 'obj[123]' sets 'indices' to '123' (not a tuple)
        # but 'obj[123, 456]' sets it to '(123, 456)' (a tuple).
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != self.dim:
            raise InvalidIndexCountException(
                f"Wrong number of indices provided during indexing. Expected {self.dim} but got {len(indices)}.")
        return self.getitem(*indices, strict=False)

    # Since __getitem__ is implemented, this class is implicitly considered
    # to be iterable (which it's not). This avoids some confusing behaviour.
    __iter__ = None

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def parent(self) -> Array | None:
        pass

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def buffer(self) -> Any:
        pass

    @abc.abstractmethod
    def getitem(self, *indices, strict=False):
        pass

    # TODO: remove these
    @abc.abstractmethod
    def with_context(self):
        pass

    @property
    @abc.abstractmethod
    def context_free(self):
        pass

    @property
    @abc.abstractmethod
    def alloc_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def leaf_layouts(self):  # or all layouts?
        pass

    @property
    @abc.abstractmethod
    def axis_trees(self) -> tuple[AbstractAxisTree, ...]:
        pass

    # }}}

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    @property
    def dtype(self) -> np.dtype:
        return self.buffer.dtype

    def assign(self, other, /, *, eager=False):
        return self._assign(other, "write", eager=eager)

    def iassign(self, other, /, *, eager=False):
        return self._assign(other, "inc", eager=eager)

    def _assign(self, other, mode, /, *, eager=False):
        # TODO: If eager should try and convert to some sort of maxpy operation
        # instead of doing a full code generation pass. Would have to make sure
        # that nothing is indexed. This could also catch the case of x.assign(x).
        # This will need to include expanding things like a(x + y) into ax + ay
        # (distributivity).
        expr = ArrayAssignment(self, other, mode)
        return expr() if eager else expr

    # TODO:
    # Note that this will only really work for vectors
    def maxpy(self, alpha: Iterable[numbers.Number], x: Iterable):
        raise NotImplementedError

    def duplicate(self, *, copy: bool = False) -> Tensor:
        name = f"{self.name}_copy"
        buffer = self.buffer.duplicate(copy=copy)
        return self.__record_init__(_name=name, _buffer=buffer)

    def copy(self) -> Tensor:
        return self.duplicate(copy=True)

    # NOTE: This is quite nasty
    @cached_property
    def loop_axes(self) -> tuple[Axis]:
        import pyop3.extras.debug
        pyop3.extras.debug.warn_todo("Nasty code, do it better")
        assert all(
            loop.iterset.is_linear
            for axes in self.axis_trees
            for loop in axes.outer_loops
        )
        return tuple(
            axis
            for axes in self.axis_trees
            for loop in axes.outer_loops
            for axis in loop.iterset.nodes
        )
