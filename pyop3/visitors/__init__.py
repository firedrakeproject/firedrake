from __future__ import annotations

import collections
import contextlib
import functools
import itertools
import numbers
import types
import typing
from collections.abc import Hashable
from typing import Any

import loopy as lp
from immutabledict import immutabledict as idict
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.constants
import pyop3.node
import pyop3.obj
import pyop3.sf
from pyop3 import utils
from pyop3.collections import OrderedFrozenSet
from pyop3.labeled_tree import MultiComponentLabelledNode

import pyop3.visitors.base

from pyop3.visitors.canonicalize_labels import canonicalize_labels  # noqa: F401
from pyop3.visitors.compress_indirections import materialize_indirections  # noqa: F401

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
    def _(self, obj: pyop3.obj.Object, /) -> OrderedFrozenSet:
        return obj.collect_buffers(self)


# def collect_buffers(insn: pyop3.insn.Instruction) -> OrderedFrozenSet:
#     return BufferCollector.maybe_singleton(insn.comm)(insn)
def collect_buffers(obj) -> OrderedFrozenSet:
    return BufferCollector()(obj)


class CacheKeyGetter(pyop3.node.NodeVisitor):

    def __init__(self) -> None:
        self.renamer = pyop3.visitors.base.Renamer()
        super().__init__()

    # NOTE: copied from pyop3/visitors/relabel.py
    def relabel_path(self, path):
        return idict({
            self._node_label_relabel_map.get(node): component
            for node, component in path.items()
        })

    # not a cached property because this changes as we traverse things
    @property
    def _node_label_relabel_map(self) -> dict:
        relabel_map = {}
        for key, new_label in self.renamer.store.items():
            if isinstance(key, tuple):
                obj_type, orig_label = key
                relabel_map[orig_label] = new_label
        return relabel_map


class DiskCacheKeyGetter(CacheKeyGetter):

    @functools.singledispatchmethod
    def process(self, obj: Any) -> Hashable:
        return super().process(obj)

    @process.register(types.NoneType)
    @process.register(numbers.Number)
    def _(self, obj: Any, /) -> Hashable:
        return obj

    @process.register
    def _(self, obj: pyop3.obj.Object, /) -> Hashable:
        return obj.get_disk_cache_key(self)



# TODO: This cache key is slightly too restrictive. For instance an axis tree and
# indexed axis tree can be used identically in places (the output code is unchanged
# and you'd get the same result) but currently these hash differently.
def get_disk_cache_key(obj: pyop3.obj.Object) -> Hashable:
    return DiskCacheKeyGetter()(obj)


class InstructionExecutorCacheKeyGetter(CacheKeyGetter):

    def __init__(self):
        # Flag that tells us what to do about buffers, do we consider
        # them replaceable or not?
        # TODO: awful name
        self.outer = True
        super().__init__()

    def __call__(self, obj, *, inside=None):
        if inside is not None:
            assert inside == True
            with self.inside():
                return super().__call__(obj)
        else:
            return super().__call__(obj)

    @contextlib.contextmanager
    def inside(self):
        prev_outer = self.outer
        self.outer = False
        yield
        self.outer = prev_outer

    @functools.singledispatchmethod
    def process(self, obj: Any) -> Hashable:
        return super().process(obj)

    @process.register(types.NoneType)
    @process.register(numbers.Number)
    def _(self, obj: Any, /) -> Hashable:
        return obj

    @process.register
    def _(self, obj: pyop3.obj.Object) -> Hashable:
        return obj.get_instruction_executor_cache_key(self)


def get_instruction_executor_cache_key(obj: pyop3.obj.Object) -> Hashable:
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


@functools.singledispatch
def get_comm(obj: Any, /) -> MPI.Comm:
    """Return the communicator associated with an object.

    If no communicator is available (e.g. trying to get the comm of an integer)
    then ``COMM_SELF`` is used.

    """
    utils.raise_missing_dispatch_handler(obj)


@get_comm.register
def _(comm: MPI.Comm, /) -> MPI.Comm:
    return comm


# this is the internal comm
# @get_comm.register
# def _(obj: PETSc.Object, /) -> MPI.Comm:
#     return obj.comm.tompi4py()


@get_comm.register
def _(sf: pyop3.sf.AbstractStarForest, /) -> MPI.Comm:
    return sf.comm


@get_comm.register
def _(obj: pyop3.obj.Object, /) -> MPI.Comm:
    return obj.comm


@get_comm.register(str)
@get_comm.register(numbers.Number)
@get_comm.register(types.NoneType)
@get_comm.register(lp.TranslationUnit)
@get_comm.register(pyop3.constants.Intent)
@get_comm.register(pyop3.constants._Decide)  # pyop3.DECIDE
def _(_, /) -> MPI.Comm:
    return MPI.COMM_SELF


@get_comm.register
def _(iterable: tuple | list, /) -> MPI.Comm:
    return common_comm(iterable, default=MPI.COMM_SELF)


@get_comm.register
def _(mapping: collections.abc.Mapping, /) -> MPI.Comm:
    return common_comm(mapping.values(), default=MPI.COMM_SELF)


@get_comm.register
def _(set_: collections.abc.Set, /) -> MPI.Comm:
    assert all(get_comm(item) == MPI.COMM_SELF for item in set_), \
        "Cannot have parallelism inside a set (unordered)"
    return MPI.COMM_SELF


def common_comm(objects: Iterable[Any], **kwargs) -> MPI.Comm:
    """Return a communicator valid for all objects.

    The valid communicator is defined as the one with the largest size.

    Parameters
    ----------
    objects
        Communicator-carrying objects to inspect. All object must define
        a ``comm`` attribute.

    Returns
    -------
    MPI.Comm
        A communicator that the provided objects are safely collective over.

    """
    return pyop3.mpi.common_comm(map(get_comm, objects), **kwargs)


def single_comm(*objects: Iterable[Any]) -> MPI.Comm:
    """Return the single comm shared by all objects."""
    return utils.single_valued(map(get_comm, objects))
