from __future__ import annotations

import abc
import itertools
import typing
from collections.abc import Callable, Hashable
from functools import cached_property
from typing import ClassVar, Literal

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.cache
import pyop3.record
from pyop3.axis_tree.tree import AbstractNonUnitAxisTree
from pyop3.cache import cached_method
from pyop3.exceptions import InvalidIndexCountException
from pyop3.expr import TerminalExpression
from pyop3.expr.base import ExpressionT

if typing.TYPE_CHECKING:
    import pyop3.insn
    import pyop3.insn.exec


class Tensor(TerminalExpression, abc.ABC):

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

    __abstract_record_attrs = ("name", "buffer")

    @property
    @abc.abstractmethod
    def transform(self):
        pass

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        pass

    @abc.abstractmethod
    def getitem(self, *indices, strict=False):
        pass

    @abc.abstractmethod
    def with_axis_trees(self, trees):
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
    def axis_trees(self) -> tuple[AbstractNonUnitAxisTree, ...]:
        pass

    # }}}

    # {{{ arithmetic

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

    # }}}

    @property
    def dtype(self) -> np.dtype:
        return self.buffer.dtype

    @PETSc.Log.EventDecorator()
    def assign(
        self,
        other: ExpressionT,
        /,
        mode: Literal["write", "inc", "max", "min"] = "write",
        *,
        eager: bool = False,
        eager_strategy: Literal["array", "compile"] | None = None,
        compiler_parameters: pyop3.insn.exec.CompilerParametersT | None = None,
    ) -> pyop3.insn.Assignment | None:
        return self._assign(other, mode, eager=eager, eager_strategy=eager_strategy, compiler_parameters=compiler_parameters)

    def iassign(
        self,
        other: ExpressionT,
        /,
        *,
        eager: bool = False,
        eager_strategy: Literal["array", "compile"] | None = None,
        compiler_parameters: pyop3.insn.exec.CompilerParametersT | None = None,
    ) -> pyop3.insn.Assignment | None:
        return self._assign(other, "inc", eager=eager, eager_strategy=eager_strategy, compiler_parameters=compiler_parameters)

    def _assign(
        self,
        other: ExpressionT,
        /,
        mode: Literal["write", "inc", "max", "min"],
        *,
        eager: bool,
        eager_strategy: Literal["array", "compile"] | None,
        compiler_parameters: pyop3.insn.exec.CompilerParametersT | None,
    ) -> pyop3.insn.Assignment | None:
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
                try:
                    self._array_assign(other, mode)
                except BaseException as e:
                    raise e
                    # TODO: log a warning, or do something else sensible
                    self._symbolic_assign(other, mode)(compiler_parameters=compiler_parameters)
            elif eager_strategy == "array":
                self._array_assign(other, mode)
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
    def _symbolic_assign(self, other, /, mode: Literal["write", "inc"]) -> pyop3.insn.Assignment:
        from pyop3.insn import Assignment

        return Assignment(self, other, mode)

    @abc.abstractmethod
    def _array_assign(self, other: ExpressionT, /, mode: Literal["write", "inc"]) -> None:
        pass

    @PETSc.Log.EventDecorator()
    def zero(self, **kwargs) -> pyop3.insn.Assignment | None:
        return self.assign(0, **kwargs)

    def duplicate(self, *, copy: bool = False, constant: bool | None = None) -> Tensor:
        """Return a duplicate of the tensor.

        Parameters
        ----------
        copy
            Whether to copy values to the new object.
        constant
            Is the duplicate mutable or not? If `None` then default to the const-ness
            of the original object.

        """
        name = f"{self.name}_copy"
        buffer = self.buffer.duplicate(copy=copy, constant=constant)
        return self.record_new(_name=name, _buffer=buffer)

    def copy(self, *, constant: bool | None = None) -> Tensor:
        """Return a copy of the tensor.

        Parameters
        ----------
        constant
            Is the copy mutable or not? If `None` then default to the const-ness
            of the original object.

        """
        return self.duplicate(copy=True, constant=constant)

    def concretize(self):
        """Convert to an expression, can no longer be indexed properly"""
        raise NotImplementedError

    def with_context(self, context) -> Self:
        new_axis_trees = []
        for axis_tree in self.axis_trees:
            if isinstance(axis_tree, pyop3.index_tree.LoopContextSensitive):
                cf_axis_tree = axis_tree.with_context(context)
            else:
                cf_axis_tree = axis_tree
            new_axis_trees.append(cf_axis_tree)
        return self.with_axis_trees(new_axis_trees)


# NOTE: No idea if this is where this should live, quite possibly this is wrong
class TensorTransform(pyop3.obj.Object, abc.ABC):

    __abstract_record_attrs = ("prev",)

    # @property
    # @abc.abstractmethod
    # def nest_indices(self) -> tuple[tuple[int, int], ...]:
    #     pass


class CallableTensorTransform(TensorTransform):
    pass


@pyop3.record.frozenrecord()
class OutOfPlaceCallableTensorTransform(CallableTensorTransform):

    # {{{ instance attrs

    transform_in: Callable[[Tensor, Tensor], None]
    transform_out: Callable[[Tensor, Tensor], None]
    prev: TensorTransform | None = None

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            self.transform_in,
            self.transform_out,
            visitor(self.prev),
        )


    # }}}

    # @property
    # def nest_indices(self) -> tuple[tuple[int, int], ...]:
    #     raise NotImplementedError


class IdentityTensorTransform(TensorTransform):
    pass


@pyop3.record.frozenrecord()
class ReshapeTensorTransform(IdentityTensorTransform):

    # {{{ instance attrs

    axis_trees: tuple[AxisTree, ...]
    prev: TensorTransform | None = None

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            tuple(map(visitor, self.axis_trees)),
            visitor(self.prev),
        )


    # }}}

    # @cached_property
    # def nest_indices(self) -> tuple[tuple[int, int], ...]:
    #     return tuple(
    #         itertools.zip_longest(
    #             *(axes.nest_indices for axes in self.axis_trees)
    #         )
    #     )
