import abc
from typing import Any

import numpy as np
from mpi4py import MPI

from pyop3.axtree import ContextAware
from pyop3.axtree.tree import Expression
from pyop3.exceptions import InvalidIndexCountException
from pyop3.lang import FunctionArgument, BufferAssignment
from pyop3.utils import UniqueNameGenerator


class Array(ContextAware, FunctionArgument, Expression, abc.ABC):
    _prefix = "array"
    _name_generator = UniqueNameGenerator()

    def __init__(self, name=None, *, prefix=None, parent=None) -> None:
        if name and prefix:
            raise ValueError("Can only specify one of name and prefix")
        self.name = name or self._name_generator(prefix or self._prefix)

        self.parent = parent

    def __getitem__(self, indices):
        # Handle the fact that 'obj[123]' sets 'indices' to '123' (i.e. not a tuple)
        # but 'obj[123, 456]' sets it to '(123, 456)' (i.e. a tuple).
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


class DistributedArray(Array, abc.ABC):

    # {{{ abstract methods

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
