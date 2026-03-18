from __future__ import annotations

import abc
import numbers
import typing
from functools import cached_property
from typing import Any, ClassVar, Callable, Hashable, Literal

import numpy as np
from immutabledict import immutabledict as idict
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.cache
from pyop3.cache import cached_method
from pyop3.expr.base import ExpressionT
import pyop3.record
from pyop3 import utils
from pyop3.sf import DistributedObject
from pyop3.tree.axis_tree import ContextAware
from pyop3.tree.axis_tree.tree import AbstractAxisTree
from pyop3.expr import TerminalExpression
from pyop3.exceptions import InvalidIndexCountException

if typing.TYPE_CHECKING:
    import pyop3.insn
    import pyop3.insn.exec


class Tensor(ContextAware, TerminalExpression, DistributedObject, abc.ABC):

    DEFAULT_PREFIX: ClassVar[str] = "array"

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

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

    @abc.abstractmethod
    def instruction_executor_cache_key(self, buffer_counter: Mapping[AbstractBuffer, int]) -> Hashable:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def transform(self):
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

    def assemble(self) -> None:
        """Ensure that values are up-to-date."""
        self.buffer.assemble()

    # TODO: remove these
    @abc.abstractmethod
    def with_context(self):
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

    def __iadd__(self, other: ExpressionT, /) -> Self:
        if other != 0:
            self.iassign(other, eager=True)
        return self

    def __isub__(self, other: ExpressionT, /) -> Self:
        if other != 0:
            self.iassign(-other, eager=True)
        return self

    def __imul__(self, other: ExpressionT, /) -> Self:
        if other != 1:
            self.assign(self*other, eager=True)
        return self

    def __itruediv__(self, other: ExpressionT, /) -> Self:
        if other != 1:
            self.assign(self//other, eager=True)
        return self

    @property
    def dtype(self) -> np.dtype:
        return self.buffer.dtype

    @PETSc.Log.EventDecorator()
    def assign(
        self,
        other: ExpressionT,
        /,
        *,
        eager: bool = False,
        eager_strategy: Literal["array", "compile"] | None = None,
        compiler_parameters: pyop3.insn.exec.CompilerParametersT | None = None,
    ) -> pyop3.insn.ArrayAssignment | None:
        return self._assign(other, "write", eager=eager, eager_strategy=eager_strategy, compiler_parameters=compiler_parameters)

    @PETSc.Log.EventDecorator()
    def iassign(
        self,
        other: ExpressionT,
        /,
        *,
        eager: bool = False,
        eager_strategy: Literal["array", "compile"] | None = None,
        compiler_parameters: pyop3.insn.exec.CompilerParametersT | None = None,
    ) -> pyop3.insn.ArrayAssignment | None:
        return self._assign(other, "inc", eager=eager, eager_strategy=eager_strategy, compiler_parameters=compiler_parameters)

    def _assign(
        self,
        other: ExpressionT,
        /,
        mode: Literal["write", "inc"],
        *,
        eager: bool,
        eager_strategy: Literal["array", "compile"] | None,
        compiler_parameters: pyop3.insn.exec.CompilerParametersT | None,
    ) -> pyop3.insn.ArrayAssignment | None:
        if compiler_parameters is not None and not eager:
            raise ValueError("Compiler parameters can only be passed to eager operations")

        if eager:
            # Have we already compiled code for this assignment? If so then reuse it
            # regardless of 'eager_strategy' (it will be faster).
            cache = pyop3.cache.get_method_cache(self)[self._symbolic_assign.__qualname__]
            cache_key = self._symbolic_assign.cache_key(self, other, mode)
            try:
                assign_insn = cache[cache_key]
            except KeyError:
                pass
            else:
                assign_insn(compiler_parameters=compiler_parameters)
                return

            if eager_strategy is None:
                eager_strategy = "array"

            if eager_strategy == "array":
                try:
                    self._array_assign(other, mode)
                    return
                except BaseException as e:
                    raise e
                    # log a warning and throw a good exception...
                    breakpoint()
                    self._symbolic_assign(other, mode)(compiler_parameters=compiler_parameters)
                    return
            else:
                assert eager_strategy == "compile"
                self._symbolic_assign(other, mode)(compiler_parameters=compiler_parameters)
                return
        else:
            if eager_strategy is not None:
                raise ValueError(
                    "'eager_strategy' is only a valid option for eagerly evaluated assignments"
                )

            return self._symbolic_assign(other, mode)

    @cached_method()
    def _symbolic_assign(self, other, /, mode: Literal["write", "inc"]) -> pyop3.insn.ArrayAssignment:
        from pyop3.insn import ArrayAssignment

        return ArrayAssignment(self, other, mode)

    @abc.abstractmethod
    def _array_assign(self, other: ExpressionT, /, mode: Literal["write", "inc"]) -> None:
        pass

    @PETSc.Log.EventDecorator()
    def zero(self, **kwargs) -> pyop3.insn.ArrayAssignment | None:
        return self.assign(0, **kwargs)

    def duplicate(self, *, copy: bool = False) -> Tensor:
        name = f"{self.name}_copy"
        buffer = self.buffer.duplicate(copy=copy)
        return self.__record_init__(_name=name, _buffer=buffer)

    def copy(self) -> Tensor:
        return self.duplicate(copy=True)

    @abc.abstractmethod
    def concretize(self):
        """Convert to an expression, can no longer be indexed properly"""


# NOTE: No idea if this is where this should live, quite possibly this is wrong
class TensorTransform(abc.ABC):

    @abc.abstractmethod
    def instruction_executor_cache_key(self, buffer_counter: Mapping[AbstractBuffer, int]) -> Hashable:
        pass

    @property
    @abc.abstractmethod
    def prev(self) -> TensorTransform | None:
        pass


class CallableTensorTransform(TensorTransform):
    pass


@pyop3.record.frozenrecord()
class OutOfPlaceCallableTensorTransform(CallableTensorTransform):

    # {{{ instance attrs

    transform_in: Callable[[Tensor, Tensor], None]
    transform_out: Callable[[Tensor, Tensor], None]
    _prev: TensorTransform | None = None

    # }}}

    # {{{ interface impls

    def instruction_executor_cache_key(self, buffer_counter: Mapping[AbstractBuffer, int]) -> Hashable:
        prev_key = self.prev.instruction_executor_cache_key(buffer_counter) if self.prev else None
        return (type(self), self.transform_in, self.transform_out, prev_key)

    prev = pyop3.record.attr("_prev")

    # }}}


class IdentityTensorTransform(TensorTransform):
    pass


@pyop3.record.frozenrecord()
class ReshapeTensorTransform(IdentityTensorTransform):

    # {{{ instance attrs

    axis_trees: tuple[AxisTree, ...]
    _prev: TensorTransform | None = None

    # }}}

    # {{{ interface impls

    def instruction_executor_cache_key(self, buffer_counter: Mapping[AbstractBuffer, int]) -> Hashable:
        prev_key = self.prev.instruction_executor_cache_key(buffer_counter) if self.prev else None
        return (type(self), self.axis_trees, prev_key)

    prev = pyop3.record.attr("_prev")

    # }}}
