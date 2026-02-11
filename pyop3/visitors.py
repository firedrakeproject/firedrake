import functools
import numbers
import types
from collections.abc import Hashable
from typing import Any

from immutabledict import immutabledict as idict
from mpi4py import MPI

from pyop3 import utils
from pyop3.insn.visitors import Renamer
import pyop3.buffer
import pyop3.expr as op3_expr
import pyop3.tree.axis_tree as op3_axis_tree
from pyop3.expr.visitors import get_disk_cache_key as get_expr_parallel_safe_hashkey
from pyop3.tree.axis_tree.visitors import get_disk_cache_key as get_tree_parallel_safe_hashkey


def get_parallel_safe_hashkey(obj: Any) -> Hashable:
    renamer = Renamer()
    return _get_parallel_safe_hashkey(obj, renamer)


@functools.singledispatch
def _get_parallel_safe_hashkey(obj: Any, /, *args, **kwargs) -> Hashable:
    raise TypeError(
        f"Don't know how to generate a parallel safe cache key for '{utils.pretty_type(obj)}'"
    )


@_get_parallel_safe_hashkey.register(str)
@_get_parallel_safe_hashkey.register(type)
@_get_parallel_safe_hashkey.register(numbers.Number)
@_get_parallel_safe_hashkey.register(types.NoneType)
def _(obj, /, *args, **kwargs) -> Hashable: 
    return obj


@_get_parallel_safe_hashkey.register(dict)
def _(dict_, /, *args, **kwargs) -> idict[Hashable, Hashable]: 
    return idict({
        key: _get_parallel_safe_hashkey(value, *args, **kwargs)
        for key, value in dict_.items()
    })


@_get_parallel_safe_hashkey.register(list)
@_get_parallel_safe_hashkey.register(tuple)
def _(obj, /, *args, **kwargs) -> tuple[Hashable, ...]: 
    return tuple(_get_parallel_safe_hashkey(item, *args, **kwargs) for item in obj)


@_get_parallel_safe_hashkey.register(MPI.Comm)
def _(comm: MPI.Comm, /, *args, **kwargs) -> str: 
    # Communicators are not part of the parallel hashkey
    return "MPI.Comm"


@_get_parallel_safe_hashkey.register(op3_axis_tree.AbstractAxisTree)
@_get_parallel_safe_hashkey.register(op3_axis_tree.AxisComponent)
@_get_parallel_safe_hashkey.register(op3_axis_tree.AxisComponentRegion)
def _(axis_obj, /, renamer):
    return get_tree_parallel_safe_hashkey(axis_obj, renamer)


@_get_parallel_safe_hashkey.register(op3_expr.Expression)
def _(axis_obj, /, renamer):
    return get_expr_parallel_safe_hashkey(axis_obj, renamer)


@_get_parallel_safe_hashkey.register(pyop3.buffer.BufferRef)
def _(buffer, /, renamer):
    # being lazy
    return type(buffer)
