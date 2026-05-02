from __future__ import annotations

import collections
import functools
import itertools
import numbers
import types
import typing
from collections.abc import Hashable
from typing import Any

from immutabledict import immutabledict as idict

import pyop3.node
import pyop3.obj
from pyop3 import utils
from pyop3.collections import OrderedFrozenSet

if typing.TYPE_CHECKING:
    import pyop3.types


class BufferCollector(pyop3.node.NodeCollector):

    # TODO
    # @classmethod
    # @memory_cache(heavy=True)
    # def maybe_singleton(cls, comm) -> Self:
    #     return cls(comm)

    @functools.singledispatchmethod
    def process(self, obj: Any, /) -> OrderedFrozenSet:
        return super().process(obj)

    @process.register(types.NoneType)
    @process.register(numbers.Number)
    def _(self, obj: Any, /) -> OrderedFrozenSet:
        return OrderedFrozenSet()

    @process.register
    def _(self, obj: pyop3.obj.Pyop3Object, /) -> OrderedFrozenSet:
        return obj.collect_buffers(self)

    # @process.register(pyop3.insn.NullInstruction)
    # def _(self, insn: pyop3.insn.NullInstruction, /) -> OrderedFrozenSet:
    #     return OrderedFrozenSet()
    #
    #
    # @process.register(pyop3.insn.StandaloneCalledFunction)
    # def _(self, func: pyop3.insn.StandaloneCalledFunction, /) -> OrderedFrozenSet:
    #     return OrderedFrozenSet().union(
    #         *(self._expr_collector(arg) for arg in func.arguments)
    #     )
    #
    # @process.register(pyop3.insn.Exscan)
    # def _(self, exscan: pyop3.insn.Exscan, /) -> OrderedFrozenSet:
    #     return OrderedFrozenSet().union(
    #         self._expr_collector(exscan.assignee),
    #         self._expr_collector(exscan.expression),
    #         self._expr_collector(exscan.extent),
    #     )
    #
    # @process.register(pyop3.insn.ConcretizedNonEmptyArrayAssignment)
    # def _(self, assignment: pyop3.insn.ConcretizedNonEmptyArrayAssignment, /) -> Hashable:
    #     return (
    #         self._expr_collector(assignment.assignee)
    #         | self._expr_collector(assignment.expression)
    #         | utils.reduce("|", map(self._tree_collector, assignment.axis_trees))
    #     )


# def collect_buffers(insn: pyop3.insn.Instruction) -> OrderedFrozenSet:
#     return BufferCollector.maybe_singleton(insn.comm)(insn)
def collect_buffers(obj) -> OrderedFrozenSet:
    return BufferCollector()(obj)


class DiskCacheKeyGetter(pyop3.node.NodeVisitor):

    def __init__(self) -> None:
        self.renamer = utils.Renamer2()
        super().__init__()

    @functools.singledispatchmethod
    def process(self, obj: Any) -> Hashable:
        return super().process(obj)

    @process.register(types.NoneType)
    @process.register(numbers.Number)
    def _(self, obj: Any, /) -> Hashable:
        return obj

    @process.register
    def _(self, obj: pyop3.obj.Pyop3Object, /) -> Hashable:
        return obj.get_disk_cache_key(self)

    def relabel_path(self, path: pyop3.types.ConcretePathT) -> pyop3.types.ConcretePathT:
        return idict({
            self.renamer.add(axis, "Axis"): component
            for axis, component in path.items()
        })


# TODO: This cache key is slightly too restrictive. For instance an axis tree and
# indexed axis tree can be used identically in places (the output code is unchanged
# and you'd get the same result) but currently these hash differently.
def get_disk_cache_key(obj: pyop3.obj.Pyop3Object) -> Hashable:
    return DiskCacheKeyGetter()(obj)


class InstructionExecutorCacheKeyGetter(pyop3.node.NodeVisitor):

    def __init__(self) -> None:
        self.buffer_counter = utils.Renamer()
        super().__init__()

    @functools.singledispatchmethod
    def process(self, obj: Any) -> Hashable:
        return super().process(obj)

    @process.register(types.NoneType)
    @process.register(numbers.Number)
    def _(self, obj: Any, /) -> Hashable:
        return obj

    @process.register
    def _(self, obj: pyop3.obj.Pyop3Object) -> Hashable:
        return obj.get_instruction_executor_cache_key(self)
        # self._renamer.add(loop.index)  # TODO: needed?
    #     return (
    #         type(loop),
    #         loop.index.iterset,
    #         *(self(stmt) for stmt in loop.statements),
    #     )
    #
    # @process.register(pyop3.insn.CalledFunction)
    # def _(self, func: pyop3.insn.CalledFunction, /) -> Hashable:
    #     # TODO: don't really need loopy here
    #     loopy_key = LoopyKeyBuilder()(func.function)
    #     return (
    #         type(func),
    #         loopy_key,
    #         *(map(self._get_argument_key, func.arguments)),
    #     )
    #
    # @process.register(pyop3.insn.Exscan)
    # def _(self, exscan: pyop3.insn.Exscan) -> Hashable:
    #     return (
    #         type(exscan),
    #         self._get_argument_key(exscan.assignee),
    #         self._get_argument_key(exscan.expression),
    #         exscan.scan_type,
    #     )
    #
    # @functools.singledispatchmethod
    # def _get_argument_key(self, argument: Any, /) -> Hashable:
    #     utils.raise_visitor_type_error(argument)
    #
    # @_get_argument_key.register(numbers.Number)
    # @_get_argument_key.register(pyop3.expr.AxisVar)
    # @_get_argument_key.register(pyop3.expr.LoopIndexVar)
    # def _(self, var: Hashable, /) -> Hashable:
    #     return var
    #
    # @_get_argument_key.register(pyop3.expr.OpaqueTerminal)
    # @_get_argument_key.register(Tensor)
    # def _(self, tensor: Tensor, /) -> Hashable:
    #     return tensor.instruction_executor_cache_key(self._buffer_arg_counter)
    #
    # @_get_argument_key.register(pyop3.expr.AggregateDat)
    # def _(self, agg_dat: pyop3.expr.AggregateDat, /) -> Hashable:
    #     return (type(agg_dat), tuple(self._get_argument_key(subdat) for subdat in agg_dat.subdats))
    #
    # @_get_argument_key.register(ScalarBufferExpression)
    # def _(self, buffer_expr: BufferExpression, /) -> Hashable:
    #     return (type(buffer_expr), self._get_buffer_key(buffer_expr.buffer))
    #
    # @_get_argument_key.register(LinearDatBufferExpression)
    # def _(self, buffer_expr: BufferExpression, /) -> Hashable:
    #     return (type(buffer_expr), self._get_buffer_key(buffer_expr.buffer), buffer_expr.layout)
    #
    # @_get_argument_key.register(pyop3.expr.Operator)
    # def _(self, op: pyop3.expr.Operator, /) -> Hashable:
    #     return (type(op), tuple(self._get_argument_key(operand) for operand in op.operands))
    #
    # def _get_buffer_key(self, buffer):
    #     return (type(buffer), buffer.dtype, self._buffer_arg_counter[buffer], type(buffer.handle))


def get_instruction_executor_cache_key(obj: pyop3.obj.Pyop3Object) -> Hashable:
    """
    This cache key is different to, say, a disk cache key because it happens at the start of a calculation
    Also we only care about the top-level input buffers - buffers from things like indirection maps
    aren't considered replaceable because the idea is that we pass the input expression in here and
    get something back that we can reuse if we only change the top level buffers.

    e.g. dat1.assign(dat2) is the same as dat3.assign(dat4) if dat1/dat2 have the same axis trees
    as dat3/dat4. We can reuse the indirection maps and preprocessing optimisations etc and just change
    the buffers at the top-level.
    """
    return InstructionExecutorCacheKeyGetter()(obj)
