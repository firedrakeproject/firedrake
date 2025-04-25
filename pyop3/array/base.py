from __future__ import annotations

import abc
import dataclasses
from typing import Any, ClassVar

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from pyop3 import utils
from pyop3.axtree import ContextAware
from pyop3.axtree.tree import Expression
from pyop3.exceptions import InvalidIndexCountException
from pyop3.lang import FunctionArgument, BufferAssignment


# TODO: rename 'DataCarrier'? Or TENSOR!!!!!!! Array is overloaded
@dataclasses.dataclass(init=False, frozen=True)
class Array(ContextAware, FunctionArgument, Expression, utils.RecordMixin, abc.ABC):

    # {{{ Instance attrs

    name: str
    parent: Array | None

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "array"

    # }}}

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
    def dim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
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

    # }}}

    def assign(self, other, /, *, eager=False):
        expr = BufferAssignment(self, other, "write")
        return expr() if eager else expr


# TODO: make this a dataclass and accept the buffer there
@dataclasses.dataclass(init=False, frozen=True)
class DistributedArray(Array, abc.ABC):

    # {{{ abstract methods

    # NOTE: Why is this not an attr of the parent class?
    @property
    @abc.abstractmethod
    def buffer(self) -> Any:
        pass

    @property
    @abc.abstractmethod
    def comm(self) -> MPI.Comm:
        pass

    # }}}

    @property
    def dtype(self) -> np.dtype:
        return self.buffer.dtype
