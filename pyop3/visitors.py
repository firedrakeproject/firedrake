import functools
import numbers
import types
from collections.abc import Hashable
from typing import Any

import pyop3.node
import pyop3.obj
from pyop3 import utils
from pyop3.collections import OrderedFrozenSet


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

    # @process.register(pyop3.insn.StandaloneCalledFunction)
    # def _(self, func: pyop3.insn.StandaloneCalledFunction) -> Hashable:
    #     from pyop3.expr.visitors import get_disk_cache_key as get_expr_disk_cache_key
    #
    #     loopy_key = LoopyKeyBuilder()(func.function)
    #     return (
    #         type(func),
    #         loopy_key,
    #         *(get_expr_disk_cache_key(arg, self._renamer) for arg in func.arguments),
    #     )
    #
    # @process.register(pyop3.insn.Exscan)
    # def _(self, exscan: pyop3.insn.Exscan) -> Hashable:
    #     from pyop3.expr.visitors import get_disk_cache_key as get_expr_disk_cache_key
    #
    #     return (
    #         type(exscan),
    #         get_expr_disk_cache_key(exscan.assignee, self._renamer),
    #         get_expr_disk_cache_key(exscan.expression, self._renamer),
    #         exscan.scan_type,
    #     )


def get_disk_cache_key(obj: pyop3.obj.Pyop3Object) -> Hashable:
    return DiskCacheKeyGetter()(obj)
