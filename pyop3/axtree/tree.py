from __future__ import annotations

import abc
import bisect
import collections
import copy
import dataclasses
import enum
import functools
import itertools
import numbers
import operator
import sys
import threading
import typing
from collections.abc import Iterable, Sized, Sequence
from functools import cached_property
from itertools import chain
from typing import Any, FrozenSet, Hashable, Mapping, Optional, Self, Tuple, Union, ClassVar

import cachetools
import numpy as np
import pyrsistent
from cachetools import cachedmethod
from mpi4py import MPI
from immutabledict import immutabledict
from petsc4py import PETSc

from pyop2.caching import cached_on, CacheMixin
from pyop3.exceptions import Pyop3Exception
from pyop3.dtypes import IntType
from pyop3.sf import StarForest, local_sf, single_star_sf
from pyop2.mpi import COMM_SELF, collective
from pyop3 import utils
from pyop3.tree import (
    ComponentLabelT,
    ComponentRegionLabelT,
    ComponentT,
    ConcretePathT,
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    MutableLabelledTreeMixin,
    NodeLabelT,
    accumulate_path,
    as_component_label,
    as_path,
    parent_path,
    postvisit,
    previsit,
)
from pyop3.utils import (
    has_unique_entries,
    unique_comm,
    debug_assert,
    deprecated,
    invert,
    just_one,
    merge_dicts,
    pairwise,
    single_valued,
    steps as steps_func,
    strict_int,
    strictly_all,
)

import pyop3.extras.debug


OWNED_REGION_LABEL = "owned"
GHOST_REGION_LABEL = "ghost"


class ExpectedLinearAxisTreeException(Pyop3Exception):
    ...


class ContextMismatchException(Pyop3Exception):
    pass


class ContextAware(abc.ABC):
    @abc.abstractmethod
    def with_context(self, context):
        pass


class ContextSensitive(ContextAware, abc.ABC):
    #     """Container of `IndexTree`s distinguished by outer loop information.
    #
    #     This class is required because multi-component outer loops can lead to
    #     ambiguity in the shape of the resulting `IndexTree`. Consider the loop:
    #
    #     .. code:: python
    #
    #         loop(p := mesh.points, kernel(dat0[closure(p)]))
    #
    #     In this case, assuming ``mesh`` to be at least 1-dimensional, ``p`` will
    #     loop over multiple components (cells, edges, vertices, etc) and each
    #     component will have a differently sized temporary. This is because
    #     vertices map to themselves whereas, for example, edges map to themselves
    #     *and* the incident vertices.
    #
    #     A `SplitIndexTree` is therefore useful as it allows the description of
    #     an `IndexTree` *per possible configuration of relevant loop indices*.
    #
    #     """
    #
    def __init__(self, context_map) -> None:
        self.context_map = immutabledict(context_map)

    @cached_property
    def keys(self):
        # loop is used just for unpacking
        for context in self.context_map.keys():
            indices = set()
            for loop_index in context.keys():
                indices.add(loop_index)
            return frozenset(indices)

    def with_context(self, context, *, strict=False):
        if not strict:
            context = self.filter_context(context)

        try:
            return self.context_map[context]
        except KeyError:
            raise ContextMismatchException

    def filter_context(self, context):
        key = {}
        for loop_index, path in context.items():
            if loop_index in self.keys:
                key.update({loop_index: tuple(path)})
        return immutabledict(key)

    def _shared_attr(self, attr: str):
        return single_valued(getattr(a, attr) for a in self.context_map.values())

# this is basically just syntactic sugar, might not be needed
# avoids the need for
# if isinstance(obj, ContextSensitive):
#     obj = obj.with_context(...)
class ContextFree(ContextAware, abc.ABC):
    def with_context(self, context):
        return self

    def filter_context(self, context):
        return immutabledict()

    @property
    def context_map(self):
        return immutabledict({immutabledict(): self})


class LoopIterable(abc.ABC):
    """Class representing something that can be looped over.

    In order for an object to be loop-able over it needs to have shape
    (``axes``) and an index expression per leaf of the shape. The simplest
    case is `AxisTree` since the index expression is just identity. This
    contrasts with something like an `IndexedLoopIterable` or `CalledMap`.
    For the former the index expression for ``axes[::2]`` would be ``2*i``
    and for the latter ``map(p)`` would be something like ``map[i, j]``.

    """

    @abc.abstractmethod
    def __getitem__(self, indices) -> Union[LoopIterable, ContextSensitiveLoopIterable]:
        raise NotImplementedError

    # not iterable in the Python sense
    __iter__ = None

    # should be .iter() (and support eager=True)
    # @abc.abstractmethod
    # def index(self) -> LoopIndex:
    #     pass


class ContextFreeLoopIterable(LoopIterable, ContextFree, abc.ABC):
    pass


class ContextSensitiveLoopIterable(LoopIterable, ContextSensitive, abc.ABC):
    @property
    def alloc_size(self):
        return max(ax.alloc_size for ax in self.context_map.values())


class UnrecognisedAxisException(ValueError):
    pass


# TODO: This is going to need some (trivial) tree manipulation routines
class _UnitAxisTree(CacheMixin):
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __str__(self) -> str:
        return "<UNIT>"

    size = 1
    max_size = 1
    alloc_size = 1
    depth = 1
    is_linear = True
    is_empty = False
    sf = single_star_sf(MPI.COMM_SELF) # no idea if this is right
    leaf_paths = (immutabledict(),)
    nodes = ()

    unindexed = property(lambda self: self)

    nest_indices = ()

    def prune(self) -> Self:
        return self

    def add_subtree(self, path: PathT, subtree):
        assert len(path) == 0
        return subtree

    def add_axis(self, path, axis):
        return AxisTree(axis)

    def with_context(self, *args, **kwargs):
        return self

    def materialize(self):
        return self

    def linearize(self, path):
        assert not path
        return self

    @property
    def leaf_subst_layouts(self):
        return immutabledict({immutabledict(): 0})

    @property
    def targets_acc(self):
        return ()

    @property
    def outer_loops(self):
        return ()

    def path_with_nodes(self, node) -> immutabledict:
        assert node is None
        return immutabledict()

    @property
    def _source_path_and_exprs(self):
        return immutabledict()

    def index(self) -> LoopIndex:
        from pyop3 import LoopIndex

        return LoopIndex(self)

    @property
    def comm(self):
        import pyop3.extras.debug
        pyop3.extras.debug.warn_todo("This comm choice is unsafe")
        return MPI.COMM_WORLD



UNIT_AXIS_TREE = _UnitAxisTree()
"""Placeholder value for an axis tree that is guaranteed to have a single entry.

It is useful when handling scalar indices that 'consume' axes because we need a way
to express a tree containing a single entry that does not need to be addressed using
labels.

"""


# class AxisComponentSize(abc.ABC):
#     @property
#     @abc.abstractmethod
#     def size(self):
#         pass
#
#
# @utils.frozenrecord()
# class FixedAxisComponentSize(AxisComponentSize):
#     _size: int
#
#     size: ClassVar = utils.attr("_size")
#
#
# class RaggedAxisComponentSize(AxisComponentSize):
#     _size: DatBufferExpression
#
#     size: ClassVar = utils.attr("_size")
#
#
# class MultiRegionAxisComponentSize(AxisComponentSize, metaclass=abc.ABCMeta):
#
#     @property
#     @abc.abstractmethod
#     def region_labels(self) -> tuple[ComponentRegionLabelT, ...]:
#         pass
#
#
# @utils.frozenrecord()
# class FixedMultiRegionAxisComponentSize(MultiRegionAxisComponentSize):
#     # this is a tuple of sizes and labels
#     regions: Any  # something
#
#     @property
#     def size(self) -> int:
#         return sum(s for _, s in self.regions)
#
#
# @utils.frozenrecord()
# class RaggedMultiRegionAxisComponentSize(MultiRegionAxisComponentSize):
#     _size: Any  # some sort of array
#     regions: Any  # something
#
#     @property
#     def size(self):
#         raise NotImplementedError


@utils.frozenrecord()
class AxisComponentRegion:
    size: numbers.Integral | LinearDatBufferExpression
    label: str | None = None

    def __init__(self, size, label=None):
        from pyop3 import as_linear_buffer_expression, Dat

        # this is a little clumsy
        if isinstance(size, Dat):
            size = as_linear_buffer_expression(size)

        object.__setattr__(self, "size", size)
        object.__setattr__(self, "label", label)

    def __str__(self) -> str:
        if self.label is None:
            return str(self.size)
        else:
            return f"{{{self.label}: {self.size}}}"


@functools.singledispatch
def _parse_regions(obj: Any) -> AxisComponentSize:
    from pyop3 import Dat
    from pyop3.tensor.dat import as_linear_buffer_expression

    if isinstance(obj, Dat):
        # Dats used as extents for axis component regions have a stricter
        # set of requirements than generic Dats so we eagerly cast them to
        # the constrained type here.
        orig_dat = obj
        dat = as_linear_buffer_expression(orig_dat)

        return (AxisComponentRegion(dat),)
    else:
        raise TypeError(f"No handler provided for {type(obj).__name__}")


# @_parse_regions.register(AxisComponentSize)
# def _(size_spec: AxisComponentSize) -> AxisComponentSize:
#     return size_spec


@_parse_regions.register(Sequence)
def _(regions: Sequence[AxisComponentRegion]) -> AxisComponentSize:
    regions = tuple(regions)

    if len(regions) > 1:
        if not has_unique_entries(r.label for r in regions):
            raise ValueError("Regions have duplicate labels")
        if any(r.label is None for r in regions):
            raise ValueError("Only regions for single-region components can be labelled None")

    return regions

@_parse_regions.register(numbers.Integral)
def _(num: numbers.Integral) -> FixedAxisComponentSize:
    return (AxisComponentRegion(num),)


@functools.singledispatch
def _parse_sf(obj: Any, size) -> StarForest | None:
    if obj is None:
        return None
    else:
        raise TypeError(f"No handler provided for {type(obj).__name__}")


@_parse_sf.register(StarForest)
def _(sf: StarForest, size) -> StarForest:
    if size != sf.size:
        raise ValueError("Size mismatch between regions and SF")
    return sf


@_parse_sf.register(PETSc.SF)
def _(sf: PETSc.SF, size) -> StarForest:
    return StarForest(sf, size)


def _partition_regions(regions: Sequence[AxisComponentRegion], sf: StarForest) -> tuple[AxisComponentRegion, ...]:
    """
    examples:

    (a, 5) and sf: {2 owned and 3 ghost -> (a_owned, 2), (a_ghost, 3)

    (a, 5), (b, 3) and sf: {2 owned and 6 ghost -> (a_owned, 2), (b_owned, 0), (a_ghost, 3), (b_ghost, 3)

    (a, 5), (b, 3) and sf: {6 owned and 2 ghost -> (a_owned, 5), (b_owned, 1), (a_ghost, 0), (b_ghost, 2)
    """
    region_sizes = {}
    ptr = 0
    for point_type in ["owned", "ghost"]:
        for region in regions:
            if point_type == "owned":
                size = min((region.size, sf.num_owned-ptr))
            else:
                size = region.size - region_sizes[_as_region_label(region.label, "owned")]
            region_sizes[_as_region_label(region.label, point_type)] = size
            ptr += size
    assert ptr == sf.size
    return tuple(AxisComponentRegion(size, label) for label, size in region_sizes.items())


def _as_region_label(initial_region_label: str | None, owned_or_ghost: str):
    if initial_region_label is None:
        return owned_or_ghost
    else:
        return (initial_region_label, owned_or_ghost)


class AxisComponent(LabelledNodeComponent):
    fields = LabelledNodeComponent.fields | {"regions", "sf"}

    def __init__(
        self,
        regions,
        label=utils.PYOP3_DECIDE,
        *,
        sf=None,
    ):
        regions = _parse_regions(regions)
        size = sum(r.size for r in regions)
        sf = _parse_sf(sf, size)

        super().__init__(label=label)
        self.regions = regions
        self.sf = sf

    def __str__(self) -> str:
        if self.has_non_trivial_regions:
            region_str = f"[{', '.join(map(str, self.regions))}]"
        else:
            region_str = str(utils.just_one(self.regions))

        if self.label is not None:
            return f"{{{self.label}: {region_str}}}"
        else:
            return region_str

    @cached_property
    def regionless(self) -> AxisComponent:
        return self.copy(regions=(AxisComponentRegion(self.local_size),), sf=None)

    @property
    def rank_equal(self) -> bool:
        """Return whether or not this axis component has constant size between ranks."""
        raise NotImplementedError

    @property
    @deprecated("size")
    def count(self) -> Any:
        return self.size

    @cached_property
    def size(self) -> Any:
        from pyop3 import Scalar

        if self.sf is not None:
            if not isinstance(self.local_size, numbers.Integral):
                raise NotImplementedError(
                    "Unsure what to do with non-integral sizes in parallel"
                )
            return Scalar(self.local_size)
        else:
            # can be an integer or a Dat
            return self.local_size

    @cached_property
    def local_size(self) -> Any:
        return sum(r.size for r in self.regions)

    @cached_property
    @collective
    def max_size(self):
        if not isinstance(self.local_size, numbers.Integral):
            raise NotImplementedError("Not sure what to do here yet")
        if self.sf is not None:
            return self.comm.reduce(self.local_size, MPI.MAX)
        else:
            return self.local_size

    @cached_property
    def _all_regions(self) -> tuple[AxisComponentRegion]:
        """Return axis component regions having expanded star forests into owned and ghost."""
        return _partition_regions(self.regions, self.sf) if self.sf else self.regions

    @property
    def has_non_trivial_regions(self) ->  bool:
        return utils.strictly_all(r.label is not None for r in self.regions)

    @property
    def comm(self) -> MPI.Comm | None:
        return self.sf.comm if self.sf else None

    @property
    def _region_labels(self) -> tuple[ComponentRegionLabelT]:
        return tuple(r.label for r in self.regions)

    @cached_property
    def _all_region_labels(self) -> tuple[str]:
        return tuple(r.label for r in self._all_regions)

    def localize(self) -> AxisComponent:
        return self._localized

    @cached_property
    def _localized(self) -> AxisComponent:
        return self.copy(sf=None)


class Axis(LoopIterable, MultiComponentLabelledNode, CacheMixin):
    fields = MultiComponentLabelledNode.fields | {"components"}

    def __init__(
        self,
        components,
        # label=utils.PYOP3_DECIDE,  # TODO
        label=None,
    ):
        components = self._parse_components(components)

        # relabel components if needed
        if utils.strictly_all(c.label is utils.PYOP3_DECIDE for c in components):
            if len(components) > 1:
                components = tuple(c.copy(label=i) for i, c in enumerate(components))
            else:
                components = (utils.just_one(components).copy(label=None),)

        self.components = components
        super().__init__(label=label)
        CacheMixin.__init__(self)

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.components == other.components
            and self.label == other.label
        )

    def __hash__(self):
        return hash(
            (type(self), self.components, self.label)
        )

    def __getitem__(self, indices):
        # NOTE: This *must* return an axis tree because that is where we attach
        # index expression information. Just returning as_axis_tree(self).root
        # here will break things.
        # Actually this is not the case for "identity" slices since index_exprs
        # and labels are unchanged (AxisTree vs IndexedAxisTree)
        # TODO: return a flat axis in these cases
        # TODO: Introduce IndexedAxis as an object to get around things here. It is really clunky to have to extract .root occasionally.
        return self._tree[indices]

    def __call__(self, *args):
        return as_axis_tree(self)(*args)

    def __str__(self) -> str:
        if len(self.components) == 1:
            component_str = str(utils.just_one(self.components))
        else:
            component_str = f"[{', '.join(map(str, self.components))}]}}"

        if self.label is None:
            raise NotImplementedError
        else:
            return f"{{{self.label}: {component_str}}}"

    def linearize(self, component_label):
        return self.copy(components=tuple(c for c in self.components if c.label == component_label))

    @cached_property
    def regionless(self) -> Axis:
        return self.copy(components=tuple(c.regionless for c in self.components))

    @property
    def component_labels(self):
        return tuple(c.label for c in self.components)

    @property
    def component(self):
        return just_one(self.components)

    def component_index(self, component) -> int:
        clabel = as_component_label(component)
        return self.component_labels.index(clabel)

    @property
    def comm(self) -> MPI.Comm | None:
        return unique_comm(self.components)

    @property
    def size(self):
        return self._tree.size

    @property
    def count(self):
        """Return the total number of entries in the axis across all axis parts.
        Will fail if axis parts do not have integer counts.
        """
        # hacky but right (no inner shape)
        return self.size

    @cached_property
    def count_per_component(self):
        return immutabledict({c.label: c.count for c in self.components})

    @cached_property
    def owned(self):
        return self._tree.owned.root

    def index(self):
        return self._tree.index()

    def iter(self, *args, **kwargs):
        return self._tree.iter(*args, **kwargs)

    @property
    def target_path_per_component(self):
        return self._tree.target_path_per_component

    @property
    def index_exprs_per_component(self):
        return self._tree.index_exprs_per_component

    @property
    def layout_exprs_per_component(self):
        return self._tree.layout_exprs_per_component

    @deprecated("as_tree")
    @property
    def axes(self):
        return self.as_tree()

    @property
    def index_exprs(self):
        return self._tree.index_exprs

    def as_tree(self) -> AxisTree:
        """Convert the axis to a tree that contains it.

        Returns
        -------
        Axis Tree
            TODO

        Notes
        -----
        The result of this function is cached because `AxisTree`s are immutable
        and we want to cache expensive computations on them.

        """
        return self._tree

    def localize(self):
        return self._localized

    @cached_property
    def _localized(self):
        return self.copy(components=[c.localize() for c in self.components])

    # TODO: Most of these should be deleted
    # Ideally I want to cythonize a lot of these methods
    def component_numbering(self, component):
        cidx = self.component_index(component)
        return self._default_to_applied_numbering[cidx]

    def component_permutation(self, component):
        cidx = self.component_index(component)
        return self._default_to_applied_permutation[cidx]

    def default_to_applied_component_number(self, component, number):
        cidx = self.component_index(component)
        return self._default_to_applied_numbering[cidx][number]

    def applied_to_default_component_number(self, component, number):
        cidx = self.component_index(component)
        return self._applied_to_default_numbering[cidx][number]

    def axis_to_component_number(self, number):
        # return axis_to_component_number(self, number)
        cidx = self._axis_number_to_component_index(number)
        return self.components[cidx], number - self._component_offsets[cidx]

    def component_offset(self, component):
        cidx = self.component_index(component)
        return self._component_offsets[cidx]

    def component_to_axis_number(self, component, number):
        cidx = self.component_index(component)
        return self._component_offsets[cidx] + number

    def renumber_point(self, component, point):
        renumbering = self.component_numbering(component)
        return renumbering[point]

    @cached_property
    def _tree(self):
        return AxisTree(self)

    @cached_property
    def _component_offsets(self):
        return (0,) + tuple(np.cumsum([c.count for c in self.components], dtype=int))

    @cached_property
    def _default_to_applied_numbering(self):
        return tuple(np.arange(c.count, dtype=IntType) for c in self.components)

        # renumbering = [np.empty(c.count, dtype=IntType) for c in self.components]
        # counters = [itertools.count() for _ in range(self.degree)]
        # for pt in self.numbering.data_ro:
        #     cidx = self._axis_number_to_component_index(pt)
        #     old_cpt_pt = pt - self._component_offsets[cidx]
        #     renumbering[cidx][old_cpt_pt] = next(counters[cidx])
        # assert all(next(counters[i]) == c.count for i, c in enumerate(self.components))
        # return tuple(renumbering)

    @cached_property
    def _default_to_applied_permutation(self):
        # is this right?
        return self._applied_to_default_numbering

    # same as the permutation...
    @cached_property
    def _applied_to_default_numbering(self):
        return self._default_to_applied_numbering
        # return tuple(invert(num) for num in self._default_to_applied_numbering)

    def _axis_number_to_component_index(self, number):
        off = self._component_offsets
        for i, (min_, max_) in enumerate(zip(off, off[1:])):
            if min_ <= number < max_:
                return i
        raise ValueError(f"{number} not found")

    @staticmethod
    def _parse_components(components):
        if isinstance(components, Mapping):
            return tuple(
                AxisComponent(count, clabel) for clabel, count in components.items()
            )
        elif isinstance(components, Iterable):
            return tuple(as_axis_component(c) for c in components)
        else:
            return (as_axis_component(components),)


# NOTE: does this sort of expression stuff live in here? Or expr.py perhaps? Definitely
# TODO: define __str__ as an abc?
class Expression(abc.ABC):

    MAX_NUM_CHARS = 120

    # {{{ abstract methods

    # TODO: I reckon that this isn't strictly necessary - we can always detect the shape
    # via a traversal. Similarly for loop axes and such.
    @property
    @abc.abstractmethod
    def shape(self) -> AxisTree:
        pass

    @property
    @abc.abstractmethod
    def loop_axes(self) -> tuple[Axis]:
        pass

    @property
    @abc.abstractmethod
    def _full_str(self) -> str:
        pass

    # }}}

    def __str__(self) -> str:
        full_str = self._full_str
        if len(full_str) > self.MAX_NUM_CHARS:
            pos = self.MAX_NUM_CHARS // 2 - 1
            return f"{full_str[:pos]}..{full_str[-pos:]}"
        else:
            return full_str

    def __add__(self, other):
        if other == 0:
            return self
        else:
            return Add(self, other)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return Add(other, self)

    def __sub__(self, other) -> Sub | Self:
        if other == 0:
            return self
        else:
            return Sub(self, other)

    def __rsub__(self, other) -> Sub | Self:
        if other == 0:
            return self
        else:
            return Sub(other, self)

    def __mul__(self, other) -> Mul | Self:
        if other == 1:
            return self
        else:
            return Mul(self, other)

    def __rmul__(self, other) -> Mul | Self:
        if other == 1:
            return self
        else:
            return Mul(other, self)

    def __floordiv__(self, other) -> FloorDiv | Self:
        if not isinstance(other, numbers.Integral):
            return NotImplemented

        if other == 1:
            return self
        else:
            return FloorDiv(self, other)

    def __mod__(self, other) -> Modulo | Self:
        # TODO: raise nice exception
        assert isinstance(other, numbers.Number)

        if other == 1:
            return self
        else:
            return Modulo(self, other)

    def __neg__(self) -> Neg:
        if isinstance(self, Neg):
            # Neg(Neg(obj)) == obj
            return self.operand
        else:
            return Neg(self)

    def __lt__(self, other):
        return LessThan(self, other)

    def __gt__(self, other):
        return GreaterThan(self, other)

    def __le__(self, other):
        return LessThanOrEqual(self, other)

    def __ge__(self, other):
        return GreaterThanOrEqual(self, other)

    def __or__(self, other) -> Or | bool:
        return self._maybe_eager_or(self, other)

    def __ror__(self, other) -> Or | bool:
        return self._maybe_eager_or(other, self)

    @classmethod
    def _maybe_eager_or(cls, a, b) -> Or | Expression | bool:
        from pyop3 import evaluate
        from pyop3.expr_visitors import RuntimeVariableException  # put in main namespace?

        try:
            a_result = evaluate(a)
        except RuntimeVariableException:
            a_result = None

        try:
            b_result = evaluate(b)
        except RuntimeVariableException:
            b_result = None

        if a_result or b_result:
            return True
        elif a_result is False:
            if b_result is False:
                return False
            else:
                assert b_result is None
                return b
        else:
            assert a_result is None
            if b_result is False:
                return a
            else:
                assert b_result is None
                return Or(a, b)


class Operator(Expression, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def operands(self):
        pass



class UnaryOperator(Operator, metaclass=abc.ABCMeta):

    # {{{ interface impls

    @property
    def _full_str(self) -> str:
        return f"{self.symbol}{as_str(self.a)}"

    # }}}

    @property
    def operands(self):
        return (self.a,)

    @property
    def operand(self):
        return utils.just_one(self.operands)

    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        pass

    def __init__(self, a, /) -> None:
        self.a = a

    @property
    def shape(self):
        return self.a.shape

    @property
    def loop_axes(self):
        return self.a.loop_axes


class Neg(UnaryOperator):
    @property
    def symbol(self) -> str:
        return "-"


class BinaryOperator(Operator, metaclass=abc.ABCMeta):
    @property
    def operands(self):
        return (self.a, self.b)

    def __init__(self, a, b, /):
        self.a = a
        self.b = b

    @cached_property
    def shape(self) -> tuple[AxisTree]:
        from pyop3.expr_visitors import get_shape

        return (
            merge_trees2(
                utils.just_one(get_shape(self.a)),
                utils.just_one(get_shape(self.b)),
            ),
        )

    @cached_property
    def loop_axes(self):
        from pyop3.expr_visitors import get_loop_axes

        a_loop_axes = get_loop_axes(self.a)
        b_loop_axes = get_loop_axes(self.b)
        axes = collections.defaultdict(tuple, a_loop_axes)
        for loop_index, loop_axes in b_loop_axes.items():
            axes[loop_index] = utils.unique((*axes[loop_index], *loop_axes))
        return immutabledict(axes)

    def __hash__(self) -> int:
        return hash((type(self), self.a, self.b))

    def __eq__(self, other, /) -> bool:
        return type(self) == type(other) and other.a == self.a  and other.b == self.b

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.a!r}, {self.b!r})"

    @property
    def _full_str(self) -> str:
        # Always use brackets to avoid having to deal with operator precedence rules
        return f"({as_str(self.a)} {self._symbol} {as_str(self.b)})"

    @property
    @abc.abstractmethod
    def _symbol(self) -> str:
        pass

    # def as_str(self, *, full=True) -> str:
    #     # Always use brackets to avoid having to deal with operator precedence rules
    #     return f"({self.as_str(a, full=full)} {self._symbol} {self.as_str(b, full=full)})"


class Add(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "+"

    # def __init__(self, *args):
    #     super().__init__(*args)
    #
    #     if "array_22" in str(self):
    #         breakpoint()


class Sub(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "-"


class Mul(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "*"


class FloorDiv(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "//"


class Modulo(BinaryOperator):
    @property
    def _symbol(self) -> str:
        return "%"


class Condition(Operator, metaclass=abc.ABCMeta):
    pass


class BinaryCondition(BinaryOperator, Condition, metaclass=abc.ABCMeta):
    pass


class LessThan(BinaryCondition):
    @property
    def _symbol(self) -> str:
        return "<"


class GreaterThan(BinaryCondition):
    @property
    def _symbol(self) -> str:
        return ">"


class LessThanOrEqual(BinaryCondition):
    @property
    def _symbol(self) -> str:
        return "<="


class GreaterThanOrEqual(BinaryCondition):
    @property
    def _symbol(self) -> str:
        return ">="


class Or(BinaryCondition):

    @property
    def _symbol(self) -> str:
        return "|"


class TernaryOperator(Operator, metaclass=abc.ABCMeta):
    @property
    def operands(self):
        return (self.a, self.b, self.c)

    def __init__(self, a, b, c, /) -> None:
        self.a = a
        self.b = b
        self.c = c


class Conditional(TernaryOperator):

    # {{{ interface impls

    # TODO: should have a way of truncating here to keep a bit of all of them, maybe we also need a "short str" method?
    def __str__(self) -> str:
        return f"{self.predicate} ? {self.if_true} : {self.if_false}"

    @property
    def _full_str(self) -> str:
        return f"{as_str(self.predicate)} ? {as_str(self.if_true)} : {as_str(self.if_false)}"

    # }}}

    def __init__(self, predicate, if_true, if_false) -> None:
        # if if_true is self:
        #     breakpoint()
        super().__init__(predicate, if_true, if_false)

    @property
    def shape(self):
        from pyop3.expr_visitors import get_shape

        if not isinstance(self.if_true, numbers.Number):
            true_shape = get_shape(self.if_true)
            if not isinstance(self.if_false, numbers.Number):
                false_shape = self.if_false.shape
                return utils.single_valued((true_shape, false_shape))
            else:
                return true_shape
        else:
            return get_shape(self.if_false)

    @cached_property
    def loop_axes(self) -> tuple[Axis]:
        from pyop3.expr_visitors import get_loop_axes

        a_loop_axes = get_loop_axes(self.a)
        b_loop_axes = get_loop_axes(self.b)
        c_loop_axes = get_loop_axes(self.b)
        axes = collections.defaultdict(tuple, a_loop_axes)
        for loop_index, loop_axes in b_loop_axes.items():
            axes[loop_index] = utils.unique(axes[loop_index], loop_axes)
        for loop_index, loop_axes in c_loop_axes.items():
            axes[loop_index] = utils.unique(axes[loop_index], loop_axes)
        return immutabledict(axes)

    @property
    def predicate(self):
        return self.a

    @property
    def if_true(self):
        return self.b

    @property
    def if_false(self):
        return self.c


def conditional(predicate, if_true, if_false):
    from pyop3 import evaluate
    from pyop3.expr_visitors import RuntimeVariableException  # put in main namespace?

    # Try to simplify by eagerly evaluating the operands

    # If both branches are the same then just return one of them.
    if if_true == if_false:
        return if_true

    # Attempt to eagerly evaluate 'predicate' to avoid creating
    # unnecessary objects.
    try:
        return if_true if evaluate(predicate) else if_false
    except RuntimeVariableException:
        return Conditional(predicate, if_true, if_false)


class Terminal(Expression, abc.ABC):

    def __hash__(self) -> int:
        return hash((type(self), self.terminal_key))

    def __eq__(self, other, /) -> bool:
        return type(self) == type(other) and other.terminal_key == self.terminal_key

    @property
    @abc.abstractmethod
    def terminal_key(self):
        # used in `replace_terminals()`
        pass


class AxisVar(Terminal):

    # {{{ interface impls

    @cached_property
    def shape(self) -> tuple[AxisTree]:
        return (self.axis.as_tree(),)

    loop_axes = immutabledict()

    @property
    def _full_str(self) -> str:
        return f"i_{{{self.axis_label}}}"

    # }}}

    def __init__(self, axis: Axis) -> None:
        assert not isinstance(axis, str), "debugging"
        self.axis = axis

    # TODO: deprecate?
    @property
    def axis_label(self):
        return self.axis.label

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.axis_label!r})"

    @property
    def terminal_key(self) -> str:
        return self.axis_label


class NaN(Terminal):
    # {{{ interface impls

    shape = UNIT_AXIS_TREE
    loop_axes = ()

    _full_str = "NaN"

    # }}}

    def __repr__(self) -> str:
        return "NaN()"

    @property
    def terminal_key(self) -> str:
        return str(self)


NAN = NaN()


# TODO: Refactor so loop ID passed in not the actual index
class LoopIndexVar(Terminal):

    # {{{ interface impls

    @property
    def shape(self):
        return (UNIT_AXIS_TREE,)

    @property
    def loop_axes(self):
        return immutabledict({self.loop_index: (self.axis,)})

    @property
    def _full_str(self) -> str:
        return f"L_{{{self.loop_index.id}, {self.axis_label}}}"

    # }}}

    def __init__(self, loop_index, axis) -> None:
        from pyop3 import LoopIndex

        assert not isinstance(axis, str), "changed"
        assert isinstance(loop_index, LoopIndex)
        self.loop_index = loop_index
        self.axis = axis

        # we must be linear at this point
        assert len(self.axis.components) == 1

    # TODO: deprecate me
    @property
    def axis_label(self):
        return self.axis.label

    @property
    def loop_id(self):
        return self.loop_index.id

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.loop_index!r}, {self.axis_label!r})"

    @property
    def terminal_key(self) -> tuple:
        return (self.loop_index.id, self.axis_label)


# typing
# ExpressionT = Expression | numbers.Number  (in 3.10)
ExpressionT = Union[Expression, numbers.Number]


class AbstractAxisTree(ContextFreeLoopIterable, LabelledTree, CacheMixin):

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def unindexed(self) -> AxisTree:
        pass

    @property
    @abc.abstractmethod
    def nest_indices(self) -> tuple[int, ...]:
        pass

    @abc.abstractmethod
    def restrict_nest(self, nest_index: int) -> AbstractAxisTree:
        """
        The idea here is to trim ``orig_axes`` with index such that we can pretend
        that the axes always looked truncated in that form.
        """

    @abc.abstractmethod
    def blocked(self, block_shape: Sequence[int, ...]) -> AbstractAxisTree:
        pass

    # }}}

    # {{{ interface impls

    @functools.singledispatchmethod
    @classmethod
    def as_node(cls, obj: Any) -> Axis:
        raise TypeError(f"No handler defined for {type(obj).__name__}")

    @as_node.register(Axis)
    @classmethod
    def _(cls, axis: Axis) -> Axis:
        return axis

    @as_node.register(numbers.Integral)
    @classmethod
    def _(cls, num: numbers.Integral) -> Axis:
        return Axis(AxisComponent(num))

    # }}}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CacheMixin.__init__(self)

    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    # TODO: Cache this function.
    def getitem(self, indices, *, strict=False):
        from pyop3.itree.parse import as_index_forests
        from pyop3.itree import index_axes

        if indices is Ellipsis:
            return self

        axis_trees = {}
        for context, index_forest in as_index_forests(indices, axes=self).items():
            axis_trees[context] = []
            for index_tree in index_forest:
                axis_trees[context].append(index_axes(index_tree, context, self))

        if len(axis_trees) == 1:
            indexed_axis_trees = just_one(axis_trees.values())
            if len(indexed_axis_trees) > 1:
                raise NotImplementedError("Need axis forests")
            else:
                return just_one(indexed_axis_trees)
        else:
            raise NotImplementedError
            return ContextSensitiveAxisTree(axis_trees)

    @property
    def axes(self):
        return self.nodes

    @cached_property
    def pruned(self) -> AxisTree:
        return prune_zero_sized_branches(self)

    def prune(self) -> AxisTree:
        return self.pruned

    @property
    @abc.abstractmethod
    def unindexed(self):
        pass

    # @property
    # @abc.abstractmethod
    # def paths(self):
    #     pass

    @cached_property
    def source_path(self):
        # return self.paths[-1]
        return self._match_path_and_exprs(self)[0]

    @cached_property
    def target_path(self):
        # return self.paths[0]
        return self._match_path_and_exprs(self.unindexed)[0]

    @property
    @abc.abstractmethod
    def index_exprs(self):
        pass

    @property
    def source_exprs(self):
        # return self.index_exprs[-1]
        return self._match_path_and_exprs(self)[1]

    @property
    def target_exprs(self):
        # return self.index_exprs[0]
        return self._match_path_and_exprs(self.unindexed)[1]

    @property
    @abc.abstractmethod
    def layout_exprs(self):
        pass

    @property
    @abc.abstractmethod
    def layouts(self):
        pass

    @property
    @abc.abstractmethod
    def outer_loops(self):
        pass

    def subst_layouts(self):
        return self._subst_layouts_default

    # NOTE: Do we ever want non-leaf subst_layouts?
    @property
    def leaf_subst_layouts(self) -> immutabledict:
        return immutabledict({leaf_path: self.subst_layouts()[leaf_path] for leaf_path in self.leaf_paths})

    # TODO: rename to iter
    def index(self) -> LoopIndex:
        from pyop3.itree.tree import LoopIndex

        return LoopIndex(self)

    def iter(self, eagerouter_loops=(), loop_index=None, include=False, **kwargs):
        from pyop3.itree.tree import iter_axis_tree

        return iter_axis_tree(
            # hack because sometimes we know the right loop index to use
            loop_index or self.index(),
            self,
            self.target_path,
            self.target_exprs,
            outer_loops,
            include,
            **kwargs,
        )

    def as_tree(self):
        return self

    def materialize(self):
        """Return a new "unindexed" axis tree with the same shape."""
        return self._materialized

    @property
    @abc.abstractmethod
    def _materialized(self):
        pass

    @property
    @abc.abstractmethod
    def regionless(self) -> AbstractAxisTree:
        pass

    @cached_property
    def global_numbering(self) -> np.ndarray[IntType]:
        # NOTE: Identical code is used elsewhere
        if not self.comm:  # maybe goes away if we disallow comm=None
            numbering = np.arange(self.size, dtype=IntType)
        else:
            start = self.sf.comm.exscan(self.owned.size) or 0
            numbering = np.arange(start, start + self.size, dtype=IntType)

            # set ghost entries to -1 to make sure they are overwritten
            # TODO: if config.debug:
            numbering[self.owned.size:] = -1
            self.sf.broadcast(numbering, MPI.REPLACE)
            debug_assert(lambda: (numbering >= 0).all())
        return numbering

    def offset(self, indices, path=None, *, loop_exprs=immutabledict()):
        from pyop3.axtree.layout import eval_offset
        return eval_offset(
            self,
            self.subst_layouts(),
            indices,
            path,
            loop_exprs=loop_exprs,
        )

    @cached_property
    def leaf_target_paths(self):
        return tuple(
            merge_dicts(
                self.target_paths.get((ax.id, clabel), {})
                for ax, clabel in self.path_with_nodes(*leaf, ordered=True)
            )
            for leaf in self.leaves
        )

    @property
    def leaf_axis(self):
        return self.node_map[parent_path(self.leaf_path)]

    @property
    def leaf_component(self):
        return self.leaf_axis.component

    @cached_property
    def size(self):
        from pyop3.axtree.layout import _axis_tree_size

        return _axis_tree_size(self)

    @cached_property
    def max_size(self):
        from pyop3.expr_visitors import max_

        return max_(self.size)

    @cached_property
    @collective
    def global_size(self):
        return self.comm.allreduce(self.owned.size)

    @cached_property
    def alloc_size(self):
        return self._alloc_size()

    @cached_property
    def sf(self) -> StarForest:
        from pyop3.axtree.parallel import collect_star_forests, concatenate_star_forests

        has_sfs = bool(list(filter(None, (component.sf for axis in self.axes for component in axis.components))))
        if has_sfs:
            sfs = collect_star_forests(self)
            return concatenate_star_forests(sfs)
        else:
            return None
            return local_sf(self.size, self.comm)

    def section(self, path: PathT, component: ComponentT) -> PETSc.Section:
        from pyop3 import Dat
        from pyop3.axtree.layout import _axis_tree_size_rec
        # from pyop3.expr_visitors import extract_axes

        # TODO: cast component to AxisComponent

        path = as_path(path)
        axis = self.node_map[path]

        subpath = path | {axis.label: component.label}
        size_expr = _axis_tree_size_rec(self, subpath)

        # NOTE: This is bizarre, what was I doing?
        if isinstance(size_expr, numbers.Integral):
            size_axes = Axis(component.local_size)
        else:
            # size_axes, _ = extract_axes(size_expr, self, (), {})
            size_axes = utils.just_one(size_expr.shape).linearize(subpath, partial=True)

        size_dat = Dat.empty(size_axes, dtype=IntType)
        size_dat.assign(size_expr, eager=True)

        sizes = size_dat.buffer.data_ro_with_halos

        import pyop3.extras.debug
        pyop3.extras.debug.warn_todo("Cythonize")

        section = PETSc.Section().create(comm=self.comm)
        section.setChart(0, component.local_size)
        for point in range(component.local_size):
            section.setDof(point, sizes[point])
        section.setUp()
        return section

    @property
    def comm(self) -> MPI.Comm:
        return unique_comm(self.nodes) or MPI.COMM_SELF

    @cached_property
    def owned(self):
        """Return the owned portion of the axis tree."""
        if self.comm.size == 1:
            return self
        else:
            return self.with_region_label(OWNED_REGION_LABEL)

    def with_region_label(self, region_label: str) -> IndexedAxisTree:
        """TODO"""
        return self.with_region_labels([region_label])

    def with_region_labels(self, region_labels: Sequence[ComponentRegionLabelT]) -> IndexedAxisTree:
        """TODO"""
        if not region_labels:
            return self
        region_labels = set(region_labels)
        if missing_labels := region_labels - set(self._all_region_labels):
            raise ValueError(f"{missing_labels} not found")

        region_slice = self._region_slice(region_labels)
        return self[region_slice]

    # NOTE: Unsure if this should be a method
    def _region_slice(self, region_labels: set, *, path: PathT = immutabledict()) -> "IndexTree":
        from pyop3.itree import AffineSliceComponent, RegionSliceComponent, IndexTree, Slice

        path = as_path(path)
        axis = self.node_map[path]

        region_label_matches_all_components = True
        region_label_matches_no_components = True
        matching_label = None
        slice_components = []
        for component in axis.components:
            if matching_labels := region_labels & set(component._all_region_labels):
                matching_label = utils.just_one(matching_labels)
                region_label_matches_no_components = False
                slice_component = RegionSliceComponent(component.label, matching_label, label=f"{component.label}_{matching_label}")
            else:
                region_label_matches_all_components = False
                slice_component = AffineSliceComponent(component.label, label=component.label)
            slice_components.append(slice_component)

        # do not change axis label if nothing changes
        if region_label_matches_all_components:
            assert matching_label is not None
            axis_label = f"{axis.label}_{matching_label}"
        elif region_label_matches_no_components:
            axis_label = axis.label
        else:
            # match some, generate something
            axis_label = None

        # NOTE: Ultimately I don't think that this step will be necessary. When axes are reused more we can
        # start to think about keying certain things on the axis itself, rather than its label.
        # slice_ = Slice(axis.label, slice_components, label=axis.label)
        slice_ = Slice(axis.label, slice_components, label=axis_label)

        index_tree = IndexTree(slice_)
        for component, slice_component in zip(axis.components, slice_.components, strict=True):
            path_ = path | {axis.label: component.label}
            if self.node_map[path_]:
                subtree = self._region_slice(region_labels, path=path_)
                index_tree = index_tree.add_subtree({slice_.label: slice_component.label}, subtree)
        return index_tree

    def _accumulate_targets(self, targets_per_axis, *, path: ConcretePathT=immutabledict(), target_path_acc=None, target_exprs_acc=None):
        """Traverse the tree and accumulate per-node targets."""
        axis = self.node_map[path]
        targets = {}

        if path == immutabledict():
            target_path_acc, target_exprs_acc = targets_per_axis.get(path, (immutabledict(), immutabledict()))
            targets[path] = (target_path_acc, target_exprs_acc)

        for component in axis.components:
            path_ = path | {axis.label: component.label}
            axis_target_path, axis_target_exprs = targets_per_axis.get(path_, (immutabledict(), immutabledict()))
            target_path_acc_ = target_path_acc | axis_target_path
            target_exprs_acc_ = target_exprs_acc | axis_target_exprs
            targets[path_] = (target_path_acc_, target_exprs_acc_)

            if self.node_map[path_]:
                targets_ = self._accumulate_targets(
                    targets_per_axis,
                    path=path_,
                    target_path_acc=target_path_acc_,
                    target_exprs_acc=target_exprs_acc_,
                )
                targets.update(targets_)
        return immutabledict(targets)

    # TODO: refactor/move
    def _match_path_and_exprs(self, tree):
        """
        Find the set of paths and expressions that match the given tree. This is
        needed because we have multiple such expressions for intermediate indexing.

        If we retained an order then this might be easier to index with 0 and -1.
        """
        map_path = None
        map_exprs = None
        for paths_and_exprs in self.paths_and_exprs:
            matching = True
            for key, (mypath, myexprs) in paths_and_exprs.items():
                # check if mypath is consistent with the labels of tree
                # NOTE: should probably also check component labels
                if not (mypath.keys() <= tree.node_labels):
                    matching = False
                    break

            if not matching:
                continue

            assert map_path is None and map_exprs is None
            # do an accumulation
            map_path = {}
            map_exprs = {}
            for key, (mypath, myexprs) in paths_and_exprs.items():
                map_path[key] = mypath
                map_exprs[key] = myexprs
        assert map_path is not None and map_exprs is not None
        return map_path, map_exprs

    @property
    @abc.abstractmethod
    def _matching_target(self):
        pass

    @cached_property
    def _subst_layouts_default(self):
        return subst_layouts(self, self._matching_target, self.layouts)

    @cached_property
    def _source_paths(self) -> tuple[PathT]:
        if self.is_empty:
            return immutabledict({immutabledict(): immutabledict()})
        else:
            return self._collect_source_path(path=immutabledict())

    def _collect_source_path(self, *, path: PathT):
        assert not self.is_empty

        source_path = {}
        if path == immutabledict():
            source_path |= {immutabledict(): immutabledict()}

        axis = self.node_map[path]
        for component in axis.components:
            path_ = path | {axis.label: component.label}
            source_path |= {path_: immutabledict({axis.label: component.label})}
            if self.node_map[path_]:
                source_path |= self._collect_source_path(path=path_)
        return immutabledict(source_path)

    @cached_property
    def _source_exprs(self):
        assert not self.is_empty, "handle outside?"
        if self.is_empty:
            return immutabledict({immutabledict(): immutabledict()})
        else:
            return self._collect_source_exprs(path=immutabledict())

    def _collect_source_exprs(self, *, path: ConcretePathT) -> immutabledict:
        source_exprs = {}

        if path == immutabledict():
            source_exprs |= {immutabledict(): immutabledict()}

        axis = self.node_map[path]
        for component in axis.components:
            path_ = path | {axis.label: component.label}
            source_exprs |= {path_: immutabledict({axis.label: AxisVar(axis)})}
            if self.node_map[path_]:
                source_exprs |= self._collect_source_exprs(path=path_)
        return immutabledict(source_exprs)

    def _check_labels(self):
        def check(node, prev_labels):
            if node == self.root:
                return prev_labels
            if node.label in prev_labels:
                raise ValueError("shouldn't have the same label as above")
            return prev_labels | {node.label}

        previsit(self, check, self.root, frozenset())

    @property
    @abc.abstractmethod
    def _buffer_slice(self) -> slice | np.ndarray[IntType]:
        pass

    def _alloc_size(self, axis=None):
        if self.is_empty:
            pyop3.extras.debug.warn_todo("think about zero-sized things, should this be allowed?")
            return 1
        axis = axis or self.root
        return sum(cpt.alloc_size(self, axis) for cpt in axis.components)

    @cached_property
    def _all_region_labels(self) -> tuple[ComponentRegionLabelT]:
        region_labels = utils.OrderedSet()
        for axis in self.axes:
            for component in axis.components:
                for region in component._all_regions:
                    if region.label is not None:
                        region_labels.add(region.label)
        return tuple(region_labels)

    def _block_indices(self, block_shape: Sequence[int, ...]) -> tuple[ScalarIndex, ...]:
        from pyop3 import ScalarIndex

        indices = []
        # Pop entries off the bottom of the tree in reverse order. These must
        # match for all leaves.
        blocked_tree = self.materialize()
        for block_size in reversed(block_shape):
            block_axis = utils.single_valued(blocked_tree.leaves)
            assert block_axis.component.size == block_size

            index = ScalarIndex(block_axis.label, block_axis.component.label, 0)
            indices.append(index)

            # now trim the leaves
            node_map = dict(blocked_tree.node_map)
            for leaf_path in blocked_tree.leaf_paths:
                del node_map[leaf_path]
                node_map[parent_path(leaf_path)] = None
            blocked_tree = AxisTree(node_map)
        return tuple(indices)

    # {{{ parallel

    # TODO: cached method
    def lgmap(self, *, block_shape: tuple[int, ...] = ()) -> PETSc.LGMap:
        blocked_axes = self.blocked(block_shape)
        indices = blocked_axes.global_numbering
        bsize = np.prod(block_shape, dtype=int)
        return PETSc.LGMap().create(indices, bsize=bsize, comm=self.comm)

    # }}}


class AxisTree(MutableLabelledTreeMixin, AbstractAxisTree):

    # {{{ interface impls

    @property
    def unindexed(self) -> AxisTree:
        return self

    @property
    def _materialized(self):
        return self

    @cached_property
    def regionless(self) -> AxisTree:
        node_map = {
            path: axis.regionless if axis else None
            for path, axis in self.node_map.items()
        }
        return type(self)(node_map)

    @property
    def nest_indices(self) -> tuple[()]:
        return ()

    def restrict_nest(self, nest_index: int) -> AxisTree:
        return self[nest_index].materialize()

    def blocked(self, block_shape: Sequence[int, ...]) -> AxisTree:
        if len(block_shape) == 0:
            return self
        else:
            return self[self._block_indices(block_shape)].materialize()

    # }}}

    def localize(self) -> AxisTree:
        return self._localized

    @cached_property
    def _localized(self) -> AxisTree:
        node_map = {
            path: axis.localize() if axis else None
            for path, axis in self.node_map.items()
        }
        return type(self)(node_map)

    # bit of a hack, think about the design
    @cached_property
    def targets_acc(self):
        return frozenset({self._accumulate_targets(self._source_path_and_exprs)})

    @cached_property
    def _source_path_and_exprs(self) -> immutabledict:
        return immutabledict({
            path: (self._source_paths[path], self._source_exprs[path])
            for path in self.paths
        })
        # TODO: merge source path and source expr collection here
        return immutabledict({key: (self._source_path[key], self._source_exprs[key]) for key in self._source_path})

    @cached_property
    def paths_and_exprs(self):
        """
        Return the possible paths represented by this tree.
        """
        return frozenset({self._source_path_and_exprs})

    @cached_property
    def index_exprs(self):
        return (self._source_exprs,)

    @property
    def layout_exprs(self):
        return self.index_exprs

    @property
    def outer_loops(self):
        return ()

    def linearize(self, path: PathT, *, partial: bool = False) -> AxisTree:
        """Return the axis tree dropping all components not specified in the path.

        partial :
            If `True` then only linearise using a partial path.

        """
        path = as_path(path)

        if not partial and path not in self.leaf_paths:
            raise ValueError("Provided path must go all the way from the root to a leaf")

        assert path in self.node_map

        linear_axes = []
        for axis, component_label in self.visited_nodes(path):
            component = utils.just_one(c for c in axis.components if c.label == component_label)
            linear_axis = Axis([component], axis.label)
            linear_axes.append(linear_axis)
        axis_tree = AxisTree.from_iterable(linear_axes)

        if partial and path not in self.leaf_paths:
            breakpoint()

        return axis_tree

    # NOTE: should default to appending (assuming linear)
    def add_axis(self, path: PathT, axis: Axis) -> AxisTree:
        return super().add_node(path, axis)

    def append_axis(self, axis: Axis) -> AxisTree:
        if len(self.leaf_paths) > 1:
            raise ExpectedLinearAxisTreeException(
                "Can only append axes to trees with one leaf."
            )
        return self.add_axis(self.leaf_path, axis)

    @property
    def layout_axes(self):
        return self

    @cached_property
    def layouts(self):
        """Initialise the multi-axis by computing the layout functions."""
        from pyop3.axtree.layout import make_layouts

        loop_vars = self.outer_loop_bits[1] if self.outer_loops else {}

        with PETSc.Log.Event("pyop3: tabulate layouts"):
            layouts = make_layouts(self, loop_vars)

        if self.outer_loops:
            _, loop_vars = self.outer_loop_bits

            layouts_ = {}
            for k, layout in layouts.items():
                layouts_[k] = replace(layout, loop_vars)
            layouts = immutabledict(layouts_)

        # for now
        return immutabledict(layouts)

        # Have not considered how to do sparse things with external loops
        if self.layout_axes.depth > self.depth:
            return layouts

        layouts_ = {immutabledict(): 0}
        for axis in self.nodes:
            for component in axis.components:
                orig_path = self.path(axis, component)
                new_path = {}
                replace_map = {}
                for ax, cpt in self.path_with_nodes(axis, component).items():
                    new_path.update(self.target_paths.get((ax.id, cpt), {}))
                    replace_map.update(self.layout_exprs.get((ax.id, cpt), {}))
                new_path = immutabledict(new_path)

                orig_layout = layouts[orig_path]
                new_layout = IndexExpressionReplacer(replace_map, loop_exprs)(
                    orig_layout
                )
                layouts_[new_path] = new_layout
        return immutabledict(layouts_)

    @property
    def _matching_target(self):
        return self._source_path_and_exprs

    @cached_property
    def _buffer_slice(self) -> slice:
        return slice(self.size)


class IndexedAxisTree(AbstractAxisTree):

    # {{{ interface impls

    @cached_property
    def _materialized(self) -> AxisTree:
        if self.is_empty:
            return AxisTree()
        else:
            return AxisTree(self.node_map)

    @cached_property
    def regionless(self) -> IndexedAxisTree:
        return type(self)(
            self.materialize().regionless,
            targets=self.targets,
            unindexed=self.unindexed.regionless,
            outer_loops=self.outer_loops,
        )

    @property
    def unindexed(self):
        return self._unindexed

    # TODO: Should have nest indices and nest labels as separate concepts.
    # The former is useful for buffers and the latter for trees
    @cached_property
    def nest_indices(self) -> tuple[int, ...]:
        # Compare the 'fully indexed' bits of the matching target and try to
        # match to the unindexed tree.
        if immutabledict() not in self._matching_target:
            return ()
        consumed_axes = dict(self._matching_target[immutabledict()][0])

        nest_indices_ = []
        path = immutabledict()
        while consumed_axes:
            axis = self.unindexed.node_map[path]
            component_label = consumed_axes.pop(axis.label)
            component_index = axis.component_labels.index(component_label)

            if axis.components[component_index].size != 1:
                # indexed bit is not a scalar axis anymore, nest indices
                # don't make sense here
                break

            path = path | {axis.label: component_label}
            nest_indices_.append(component_index)
        return tuple(nest_indices_)

    def restrict_nest(self, nest_label: ComponentLabelT) -> IndexedAxisTree:
        """Given an already indexed thing, discard the prescribed nest shape."""

        subtree_unindexed = self.unindexed[nest_label].materialize()

        # remove the nest label from the targets
        subtree_targets = tuple(
            {
                axis_path: (
                    {
                        axis_label: path
                        for axis_label, path in target_spec[0].items()
                        if axis_label != self.unindexed.root.label
                    },
                    {
                        axis_label: expr
                        for axis_label, expr in target_spec[1].items()
                        if axis_label != self.unindexed.root.label
                    },
                )
                for axis_path, target_spec in target.items()
            }
            for target in self.targets
        )

        return IndexedAxisTree(
            self.node_map,
            unindexed=subtree_unindexed,
            targets=subtree_targets,
            outer_loops=self.outer_loops,
        )

    def blocked(self, block_shape: Sequence[int, ...]) -> IndexedAxisTree:
        """
        Note: this function assumes that the block shape still exists in the tree.
        """
        if len(block_shape) == 0:
            return self

        block_indices = self._block_indices(block_shape)

        self_blocked = self[block_indices]
        unindexed_blocked = self.unindexed.blocked(block_shape)

        # remove the block axes from the targets
        block_axis_labels = frozenset(index.axis for index in block_indices)
        targets_blocked = tuple(
            {
                axis_path: (
                    {
                        axis_label: path
                        for axis_label, path in target_spec[0].items()
                        if axis_label not in block_axis_labels
                    },
                    {
                        axis_label: expr
                        for axis_label, expr in target_spec[1].items()
                        if axis_label not in block_axis_labels
                    },
                )
                for axis_path, target_spec in target.items()
            }
            for target in self_blocked.targets
        )

        return IndexedAxisTree(
            self_blocked.node_map,
            unindexed=unindexed_blocked,
            targets=targets_blocked,
            outer_loops=self.outer_loops,
        )

    # }}}


    # NOTE: It is OK for unindexed to be None, then we just have a map-like thing
    def __init__(
        self,
        node_map,
        unindexed,  # allowed to be None
        *,
        targets,
        layout_exprs=None,  # not used
        outer_loops=(),
    ):
        if isinstance(node_map, AxisTree):
            node_map = node_map.node_map

        # drop duplicate entries as they are necessarily equivalent
        targets = utils.unique(targets)
        targets = utils.freeze(targets)

        if layout_exprs is None:
            layout_exprs = immutabledict()
        if outer_loops is None:
            outer_loops = ()

        self.targets = targets
        self._unindexed = unindexed
        self._outer_loops = tuple(outer_loops)
        super().__init__(node_map)


    # old alias
    @property
    def _targets(self):
        return self.targets

    @property
    def comm(self):
        return self.unindexed.comm

    # ideally this is ordered
    # TODO: should include source I think to be consistent with AxisTree
    @cached_property
    def targets_acc(self):
        return frozenset(self._accumulate_targets(t) for t in self.targets)

    # compat for now while I tinker
    @cached_property
    def _target_paths(self):
        return frozenset({
            immutabledict({key: path for key, (path, _) in target.items()})
            for target in self._targets
        })

    @cached_property
    def _target_exprs(self):
        return frozenset({
            immutabledict({key: exprs for key, (_, exprs) in target.items()})
            for target in self._targets
        })

    @cached_property
    def paths(self):
        """
        Return a `tuple` of the possible paths represented by this tree.
        """
        # return self._target_paths | {self._source_path}
        return self._target_paths

    # @cached_property
    # def _source_path_and_exprs(self):
    #     # TODO: merge source path and source expr collection here
    #     return freeze({key: (self._source_path[key], self._source_exprs[key]) for key in self._source_path})

    @cached_property
    def paths_and_exprs(self):
        """
        Return a `tuple` of the possible paths represented by this tree.
        """
        # return self._targets | {self._source_path_and_exprs}
        return self._targets

    # def _collect_paths(self, *, axis=None):
    #     """
    #     Traverse the tree and add the trivial path to the possible paths
    #     represented by each node.
    #     """
    #     paths = {}
    #
    #     if axis is None:
    #         axis = self.root
    #         paths[None] = self._target_paths.get(None, {})
    #
    #     for component in axis.components:
    #         axis_key = (axis.id, component.label)
    #         source_path = pmap({axis.label: component.label})
    #         target_paths = self._target_paths.get(axis_key, ())
    #         paths[axis_key] = target_paths + (source_path,)
    #
    #         if subaxis := self.child(axis, component):
    #             paths_ = self._collect_paths(axis=subaxis)
    #             paths.update(paths_)
    #
    #     return freeze(paths)


    @property
    def index_exprs(self):
        # return self._target_exprs | {self._source_exprs}
        return self._target_exprs

    @property
    def layout_exprs(self):
        return self._layout_exprs

    @property
    def layouts(self):
        return self.unindexed.layouts

    @property
    def outer_loops(self):
        return self._outer_loops

    def linearize(self, path: PathT) -> IndexedAxisTree:
        """Return the axis tree dropping all components not specified in the path."""
        path = as_path(path)

        linearized_axis_tree = self.materialize().linearize(path)

        # linearize the targets
        linearized_targets = []
        path_set = frozenset(path.items())
        for orig_target in self.targets:
            linearized_target = {}
            for axis_path, target_spec in orig_target.items():
                axis_path_set = frozenset(axis_path.items())
                if axis_path_set <= path_set:
                    linearized_target[axis_path] = target_spec
            linearized_target = immutabledict(linearized_target)
            linearized_targets.append(linearized_target)

        return IndexedAxisTree(
            linearized_axis_tree, self.unindexed, targets=linearized_targets,
        )

    @cached_property
    def layout_axes(self) -> AxisTree:
        if not self.outer_loops:
            return self
        raise NotImplementedError
        loop_axes, _ = self.outer_loop_bits
        return loop_axes.add_subtree(self, *loop_axes.leaf)

    # This could easily be two functions
    @cached_property
    def outer_loop_bits(self):
        assert False, "dont think I use this"
        # from pyop3.itree.tree import LocalLoopIndexVariable

        if len(self.outer_loops) > 1:
            # We do not yet support something like dat[p, q] if p and q
            # are independent (i.e. q != f(p) ).
            raise NotImplementedError(
                "Multiple independent outer loops are not supported."
            )
        loop = just_one(self.outer_loops)

        # TODO: Don't think this is needed
        # Since loop itersets must be linear, we can unpack target_paths
        # and index_exprs from
        #
        #     {(axis_id, component_label): {axis_label: expr}}
        #
        # to simply
        #
        #     {axis_label: expr}
        flat_target_paths = {}
        flat_index_exprs = {}
        for axis in loop.iterset.nodes:
            key = (axis.id, axis.component.label)
            flat_target_paths.update(loop.iterset.target_paths.get(key, {}))
            flat_index_exprs.update(loop.iterset.index_exprs.get(key, {}))

        # Make sure that the layout axes are uniquely labelled.
        suffix = f"_{loop.id}"
        loop_axes = relabel_axes(loop.iterset, suffix)

        # Nasty hack: loop_axes need to be a PartialAxisTree so we can add to it.
        loop_axes = AxisTree(loop_axes.node_map)

        # When we tabulate the layout, the layout expressions will contain
        # axis variables that we actually want to be loop index variables. Here
        # we construct the right replacement map.
        loop_vars = {
            axis.label + suffix: LocalLoopIndexVariable(loop, axis.label)
            for axis in loop.iterset.nodes
        }

        # Recursively fetch other outer loops and make them the root of
        # the current axes.
        if loop.iterset.outer_loops:
            ax_rec, lv_rec = loop.iterset.outer_loop_bits
            loop_axes = ax_rec.add_subtree(loop_axes, *ax_rec.leaf)
            loop_vars.update(lv_rec)

        return loop_axes, immutabledict(loop_vars)

    def materialize(self):
        """Return a new "unindexed" axis tree with the same shape."""
        # "unindexed" axis tree
        # strip parallel semantics (in a bad way)
        # parent_to_children = collections.defaultdict(list)
        # for p, cs in self.node_map.items():
        #     for c in cs:
        #         if c is not None and c.sf is not None:
        #             c = c.copy(sf=None)
        #         parent_to_children[p].append(c)
        #
        # return AxisTree(parent_to_children)
        return AxisTree(self.node_map)


    @cached_property
    def _matching_target(self) -> immutabledict:
        return find_matching_target(self)

    @cached_property
    def _buffer_slice(self) -> np.ndarray[IntType]:
        from pyop3 import Dat, do_loop

        if self.size == 0:
            return slice(0, 0)

        # NOTE: The below method might be better...
        # mask_dat = Dat.zeros(self.unindexed.localize(), dtype=bool, prefix="mask")
        # do_loop(p := self.index(), mask_dat[p].assign(1))
        # indices = just_one(np.nonzero(mask_dat.buffer.data_ro))

        indices_dat = Dat.empty(self.materialize().localize(), dtype=IntType, prefix="indices")
        for leaf_path in self.leaf_paths:
            iterset = self.linearize(leaf_path)
            p = iterset.index()
            offset_expr = just_one(self[p].leaf_subst_layouts.values())
            do_loop(p, indices_dat[p].assign(offset_expr))
        indices = indices_dat.buffer.data_ro_with_halos

        indices = np.unique(np.sort(indices))

        debug_assert(lambda: min(indices) >= 0 and max(indices) <= self.unindexed.size)

        # then convert to a slice if possible, do in Cython!!!
        pyop3.extras.debug.warn_todo("Convert to cython")
        slice_ = None
        n = len(indices)

        assert n > 0
        if n == 1:
            start = indices[0]
            return slice(start, start+1)
        else:
            step = indices[1] - indices[0]

            for i in range(1, n-1):
                new_step = indices[i+1] - indices[i]
                # non-const step, abort and use indices
                if new_step != step:
                    return indices

            return slice(indices[0], indices[-1]+1, step)

    # {{{ parallel

    @cached_property
    def lgmap_dat(self) -> Dat:
        return self.unindexed.lgmap_dat.__record_init__(axes=self)

    # }}}



# TODO: Choose a suitable base class
class UnitIndexedAxisTree:
    """An indexed axis tree representing something indexed down to a scalar."""
    def __init__(
        self,
        unindexed,  # allowed to be None
        *,
        targets,
        layout_exprs=None,  # not used
        outer_loops=(),  # not used?
    ):
        # drop duplicate entries as they are necessarily equivalent
        targets = utils.unique(targets)

        self.unindexed = unindexed
        self.targets = targets
        self.outer_loops = outer_loops

    def __str__(self) -> str:
        return "<UNIT>"

    def materialize(self):
        return UNIT_AXIS_TREE

    size = 1
    is_linear = True
    is_empty = False

    @property
    def regionless(self):
        return self

    @cached_property
    def _subst_layouts_default(self):
        return subst_layouts(self, self._matching_target, self.unindexed.layouts)

    @property
    def leaf_subst_layouts(self) -> immutabledict:
        return immutabledict({leaf_path: self._subst_layouts_default[leaf_path] for leaf_path in self.leaf_paths})

    @cached_property
    def _matching_target(self):
        return find_matching_target(self)

    @property
    def leaf_paths(self):
        return (immutabledict(),)

    @property
    def leaves(self):
        return (None,)

    def path_with_nodes(self, leaf):
        assert leaf is None
        return immutabledict()

    @utils.deprecated()
    @property
    def layouts(self):
        return self.unindexed.layouts

    def with_context(self, context):
        return self

    # TODO: shared with other index tree
    @cached_property
    def nest_indices(self):
        if immutabledict() not in self._matching_target:
            return ()
        consumed_axes = dict(self._matching_target[immutabledict()][0])

        nest_indices_ = []
        path = immutabledict()
        while consumed_axes:
            axis = self.unindexed.node_map[path]
            component_label = consumed_axes.pop(axis.label)
            component_index = axis.component_labels.index(component_label)

            if axis.components[component_index].size != 1:
                # indexed bit is not a scalar axis anymore, nest indices
                # don't make sense here
                break

            path = path | {axis.label: component_label}
            nest_indices_.append(component_index)
        return tuple(nest_indices_)

    # TODO: shared with other index tree
    @cached_property
    def _matching_target(self):
        return find_matching_target(self)


def find_matching_target(self):
    matching_targets = []
    for target in self.targets:
        all_leaves_match = True
        for leaf_path in self.leaf_paths:
            target_path = {}
            for leaf_path_acc in accumulate_path(leaf_path):
                if leaf_path_acc in target:
                    target_path_, _ = target[leaf_path_acc]
                    target_path.update(target_path_)
            target_path = immutabledict(target_path)

            # NOTE: We assume that if we get an empty target path then something has
            # gone wrong. This is needed because of .get() calls which are needed
            # because sometimes targets are incomplete.
            if not target_path or not target_path in self.unindexed.node_map:
                all_leaves_match = False
                break

        if all_leaves_match:
            # drop empty mappings as they lead to conflicts
            target = immutabledict({
                key: value
                for key, value in target.items()
                if value != (immutabledict(), immutabledict())
            })
            matching_targets.append(target)

    return utils.single_valued(matching_targets)


class AxisForest:
    """A collection of equivalent axis trees.

    Axis forests are useful to describe circumstances where there are multiple
    viable axis trees for describing a layout. For instance, one can view
    the data layout for a function space as a set of DoFs per mesh strata, or
    as a flat set of nodes. These layouts cannot be transformed between each
    other and so must coexist.

    """
    def __init__(self, trees: Sequence[AbstractAxisTree]) -> None:
        # TODO: Should check the trees for compatibility (e.g. do they have the same SF?)
        self.trees = tuple(trees)

    def __repr__(self) -> str:
        return f"AxisTree([{', '.join(repr(tree) for tree in self.trees)}])"

    def __getitem__(self, indices) -> AxisForest | AxisTree:
        return self.getitem(indices, strict=False)

    def getitem(self, indices, *, strict=False):
        breakpoint()


class ContextSensitiveAxisTree(ContextSensitiveLoopIterable):
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.context_map!r})"

    def __str__(self) -> str:
        return "\n".join(
            f"{context}\n{tree}" for context, tree in self.context_map.items()
        )

    def __getitem__(self, indices) -> ContextSensitiveAxisTree:
        raise NotImplementedError
        # TODO think harder about composing context maps
        # answer is something like:
        # new_context_map = {}
        # for context, axes in self.context_map.items():
        #     for context_, axes_ in index_axes(axes, indices).items():
        #         new_context_map[context | context_] = axes_
        # return ContextSensitiveAxisTree(new_context_map)

    def index(self) -> LoopIndex:
        from pyop3.itree import LoopIndex

        # TODO
        # return LoopIndex(self.owned)
        return LoopIndex(self)

    @cached_property
    def datamap(self):
        return merge_dicts(axes.datamap for axes in self.context_map.values())

    # seems a bit dodgy
    @cached_property
    def sf(self):
        return single_valued([ax.sf for ax in self.context_map.values()])

    @cached_property
    def unindexed(self):
        return single_valued([ax.unindexed for ax in self.context_map.values()])

    @cached_property
    def context_free(self):
        return just_one(self.context_map.values())


@functools.singledispatch
def as_axis_tree(arg: Any) -> AxisTree:
    axis = as_axis(arg)
    return as_axis_tree(axis)


@as_axis_tree.register
def _(axes: ContextSensitiveAxisTree) -> ContextSensitiveAxisTree:
    return axes


@as_axis_tree.register
def _(axes_per_context: collections.abc.Mapping) -> ContextSensitiveAxisTree:
    return ContextSensitiveAxisTree(axes_per_context)


@as_axis_tree.register(AxisTree)
@as_axis_tree.register(_UnitAxisTree)
def _(axes: AxisTree) -> AxisTree:
    return axes


@as_axis_tree.register
def _(axes: IndexedAxisTree) -> IndexedAxisTree:
    return axes


@as_axis_tree.register
def _(axis: Axis) -> AxisTree:
    return AxisTree(axis)


@functools.singledispatch
def as_axis(arg: Any) -> Axis:
    component = as_axis_component(arg)
    return as_axis(component)


@as_axis.register
def _(axis: Axis) -> Axis:
    return axis


@as_axis.register
def _(component: AxisComponent) -> Axis:
    return Axis(component)


@functools.singledispatch
def as_axis_component(arg: Any) -> AxisComponent:
    from pyop3 import Dat  # cyclic import

    if isinstance(arg, Dat):
        return AxisComponent(arg)
    else:
        raise TypeError(f"No handler defined for {type(arg).__name__}")


@as_axis_component.register
def _(component: AxisComponent) -> AxisComponent:
    return component


@as_axis_component.register
def _(arg: numbers.Integral) -> AxisComponent:
    return AxisComponent(arg)


# NOTE: Is this used any more?
@cached_on(lambda trees: trees[0], key=lambda trees: tuple(trees[1:]))
def merge_axis_trees(axis_trees):
    """

    Notes
    -----
    The result of this function is cached on the first axis tree.

    """
    nonempty_axis_trees = tuple(axis_tree for axis_tree in axis_trees if not axis_tree.is_empty)
    if nonempty_axis_trees:
        root, subnode_map = _merge_node_maps(nonempty_axis_trees)
        node_map = {None: (root,)}
        node_map.update(subnode_map)
    else:
        node_map = None

    if all(isinstance(axis_tree, AxisTree) for axis_tree in axis_trees):
        return AxisTree(node_map)
    else:
        targets = tuple(
            _merge_targets(axis_trees, targetss)
            for targetss in itertools.product(*[axis_tree.targets for axis_tree in axis_trees])
        )

        unindexed = merge_axis_trees([axis_tree.unindexed for axis_tree in axis_trees])
        return IndexedAxisTree(node_map, unindexed, targets=targets)


def _merge_node_maps(axis_trees, *, axis_tree_index=0, axis=None, suffix="") -> tuple[Axis, immutabledict]:
    assert axis_tree_index < len(axis_trees)

    axis_tree = axis_trees[axis_tree_index]
    assert not axis_tree.is_empty

    if axis is None:
        axis = axis_tree.root

    relabelled_axis = axis.copy(
        label=f"{axis.label}_{axis_tree_index}",
        id=f"{axis.id}{suffix}",
    )

    relabelled_children = []
    relabelled_subnode_map = {}
    for i, component in enumerate(axis.components):
        if subaxis := axis_tree.child(axis, component):
            relabelled_subaxis, relabelled_subnode_map_ = _merge_node_maps(
                axis_trees, axis_tree_index=axis_tree_index, axis=subaxis, suffix=f"{suffix}_{i}"
            )
            relabelled_children.append(relabelled_subaxis)
            relabelled_subnode_map.update(relabelled_subnode_map_)
        elif axis_tree_index + 1 < len(axis_trees):
            relabelled_subaxis, relabelled_subnode_map_ = _merge_node_maps(
                axis_trees, axis_tree_index=axis_tree_index+1, axis=None, suffix=f"{suffix}_{i}"
            )
            relabelled_children.append(relabelled_subaxis)
            relabelled_subnode_map.update(relabelled_subnode_map_)
        else:
            relabelled_children.append(None)

    relabelled_subnode_map[relabelled_axis.id] = tuple(relabelled_children)
    return (relabelled_axis, immutabledict(relabelled_subnode_map))


def _merge_targets(axis_trees, targetss, *, axis_tree_index=0, axis=None, suffix="") -> immutabledict:
    from pyop3.expr_visitors import relabel as relabel_expression

    assert axis_tree_index < len(axis_trees)

    axis_tree = axis_trees[axis_tree_index]

    orig_targets = targetss[axis_tree_index]
    relabelled_targets = {}

    if axis is None:
        axis = axis_tree.root

    for i, component in enumerate(axis.components):
        orig_axis_key = (axis.id, component.label)
        orig_target = orig_targets[orig_axis_key]
        orig_path, orig_exprs = orig_target

        relabelled_path = {
            f"{axis_label}_{axis_tree_index}": component_label
            for axis_label, component_label in orig_path.items()
        }
        relabelled_exprs = {
            f"{axis_label}_{axis_tree_index}": relabel_expression(expr, f"_{axis_tree_index}")
            for axis_label, expr in orig_exprs.items()
        }
        relabelled_target = (relabelled_path, relabelled_exprs)
        relabelled_axis_key = (f"{axis.id}{suffix}", component.label)
        relabelled_targets[relabelled_axis_key] = relabelled_target

        if subaxis := axis_tree.child(axis, component):
            relabelled_targets_ = _merge_targets(
                axis_trees, targetss, axis_tree_index=axis_tree_index, axis=subaxis, suffix=f"{suffix}_{i}"
            )
            relabelled_targets.update(relabelled_targets_)
        elif axis_tree_index + 1 < len(axis_trees):
            relabelled_targets_ = _merge_targets(
                axis_trees, targetss, axis_tree_index=axis_tree_index+1, axis=None, suffix=f"{suffix}_{i}"
            )
            relabelled_targets.update(relabelled_targets_)

    return immutabledict(relabelled_targets)


def merge_axis_trees2(trees: Iterable[AxisTree]) -> AxisTree:
    if not trees:
        raise ValueError

    current_tree, *remaining_trees = trees
    while remaining_trees:
        next_tree, *remaining_trees = remaining_trees
        current_tree = merge_trees2(current_tree, next_tree)
    return current_tree


# blast, this doesn't work...
# @cached_on(lambda t1, t2: t1, key=lambda t1, t2: t2)
def merge_trees2(tree1: AxisTree, tree2: AxisTree) -> AxisTree:
    """Merge two axis trees together.

    If the second tree has no common axes (share a lable) with the first then it is
    appended to every leaf of the first tree. Any common axes are skipped.

    Case 1:

        TODO: show example where 
        axis_a = Axis({"x": 2, "y": 2}, "a")
        axis_b = Axis({"x": 2}, "b")
        axis_c = Axis({"x": 2}, "c")
        AxisTree.from_nest({axis_a: [axis_b, axis_c]})

        is added to axis_a: things should split up.

    """
    if tree1 and not isinstance(tree1, _UnitAxisTree):
        if tree2 and not isinstance(tree2, _UnitAxisTree):
            # This is all quite magic. What this does is traverse the first tree
            # and collate all the visited axes. Then, at each leaf of tree 1, we
            # traverse tree 2 and build a per-leaf subtree as appropriate. These
            # are then all stuck together in the final step.

            subtrees = _merge_trees(tree1, tree2)

            merged = AxisTree(tree1.node_map)
            for leaf, subtree in subtrees:
                merged = merged.add_subtree(leaf, subtree)
        else:
            merged = tree1
    else:
        merged = tree2

    return merged


def _merge_trees(tree1, tree2, *, path1=immutabledict(), parents=immutabledict()):
    axis1 = tree1.node_map[path1]
    subtrees = []
    for component1 in axis1.components:
        path1_ = path1 | {axis1.label: component1.label}
        parents_ = parents | {axis1: component1}
        if tree1.node_map[path1_]:
            subtrees_ = _merge_trees(tree1, tree2, path1=path1_, parents=parents_)
            subtrees.extend(subtrees_)
        else:
            # at the bottom, now visit tree2 and try to add bits
            subtree = _build_distinct_subtree(tree2, parents_)
            subtrees.append((path1_, subtree))
    return tuple(subtrees)


def _build_distinct_subtree(axes, parents, *, path=immutabledict()):
    axis = axes.node_map[path]

    if axis in parents:
        # Axis is already visited, do not include in the new tree and make sure
        # to only use the right component
        component = parents[axis]
        path_ = path | {axis.label: component.label}
        if axes.node_map[path_]:
            return _build_distinct_subtree(axes, parents, path=path_)
        else:
            return AxisTree()
    else:
        # Axis has not yet been visited, include in the new tree
        # and traverse all subaxes
        subtree = AxisTree(axis)
        for component in axis.components:
            path_ = path | {axis.label: component.label}
            if axes.node_map[path_]:
                subtree_ = _build_distinct_subtree(axes, parents, path=path_)
                subtree = subtree.add_subtree(path_, subtree_)
        return subtree


# TODO: Move this function into another file.
def subst_layouts(
    axes,
    targets,
    layouts,
    *,
    path=immutabledict(),
    target_paths_and_exprs_acc=None,
):
    from pyop3.expr_visitors import replace_terminals

    layouts_subst = {}
    # if strictly_all(x is None for x in [axis, path, target_path_acc, index_exprs_acc]):
    if path == immutabledict():

        # NOTE: I think I can get rid of this if I prescribe an empty axis tree to expression
        # arrays
        target_paths_and_exprs_acc = {immutabledict(): targets.get(path, (immutabledict(), immutabledict()))}

        accumulated_path = merge_dicts(p for p, _ in target_paths_and_exprs_acc.values())

        # layouts_subst[path] = replace(layouts[accumulated_path], linear_axes_acc, target_paths_and_exprs_acc)
        replace_map = merge_dicts(t for _, t in target_paths_and_exprs_acc.values())

        # If we have indexed using a different order to the initial axis tree then sometimes
        # the accumulated path is not valid. In this case do not emit a layout function.
        if accumulated_path in layouts:
            layouts_subst[path] = replace_terminals(layouts[accumulated_path], replace_map)

        if axes.is_empty or axes is UNIT_AXIS_TREE or isinstance(axes, UnitIndexedAxisTree):
            return layouts_subst

    axis = axes.node_map[path]
    for component in axis.components:
        path_ = path | {axis.label: component.label}

        target_paths_and_exprs_acc_ = target_paths_and_exprs_acc | {
            path_: targets.get(path_, (immutabledict(), immutabledict()))
        }

        accumulated_path = merge_dicts(p for p, _ in target_paths_and_exprs_acc_.values())
        replace_map = merge_dicts(t for _, t in target_paths_and_exprs_acc_.values())

        # If we have indexed using a different order to the initial axis tree then sometimes
        # the accumulated path is not valid. In this case do not emit a layout function.
        if accumulated_path in layouts:
            layouts_subst[path_] = replace_terminals(layouts[accumulated_path], replace_map)

        if axes.node_map[path_]:
            layouts_subst.update(
                subst_layouts(
                    axes,
                    targets,
                    layouts,
                    path=path_,
                    target_paths_and_exprs_acc=target_paths_and_exprs_acc_,
                )
            )
    return immutabledict(layouts_subst)


def prune_zero_sized_branches(axis_tree: AbstractAxisTree, *, path=immutabledict()) -> AxisTree:
    # needed now we have unit trees?
    # if axis_tree.is_empty:
    #     return AxisTree()
    if axis_tree is UNIT_AXIS_TREE or isinstance(axis_tree, UnitIndexedAxisTree):
        return UNIT_AXIS_TREE

    _axis = axis_tree.node_map[path]

    new_components = []
    subtrees = []
    for component in _axis.components:
        path_ = path | {_axis.label: component.label}

        if component.size == 0:
            continue

        if axis_tree.node_map[path_]:
            subtree = prune_zero_sized_branches(axis_tree, path=path_)
            if subtree.size == 0:
                continue
        else:
            subtree = None

        new_components.append(component)
        subtrees.append(subtree)

    if not new_components:
        return AxisTree()

    new_axis = Axis(new_components, _axis.label)
    new_axis_tree = AxisTree(new_axis)
    for new_component, subtree in zip(new_components, subtrees, strict=True):
        if subtree is not None:
            new_axis_tree = new_axis_tree.add_subtree({_axis.label: new_component.label}, subtree)
    return new_axis_tree


def relabel_path(path, suffix:str):
    return {f"{axis_label}_{suffix}": component_label for axis_label, component_label in path.items()}


@functools.singledispatch
def as_str(expr):
    return expr._full_str


@as_str.register(Expression)
def _(expr):
    return expr._full_str


@as_str.register(numbers.Number)
@as_str.register(bool)
@as_str.register(np.bool)
def _(expr):
    return str(expr)


def get_loop_tree(expr) -> tuple[AxisTree, Mapping[LoopIndexVar, AxisVar]]:
    from pyop3.expr_visitors import get_loop_axes

    axes = []
    loop_var_replace_map = {}
    for loop_index, loop_axes in get_loop_axes(expr).items():
        for loop_axis in loop_axes:
            axis_label = f"{loop_axis.label}_{loop_index.id}"

            axis = loop_axis.copy(label=axis_label)
            axes.append(axis)

            loop_var = LoopIndexVar(loop_index, loop_axis)
            axis_var = AxisVar(axis)
            loop_var_replace_map[loop_var] = axis_var

    return (AxisTree.from_iterable(axes), loop_var_replace_map)


def loopified_shape(expr: Expression) -> tuple[AxisTree, Mapping[LoopIndexVar, AxisVar]]:
    from pyop3.expr_visitors import replace, get_shape

    loop_tree, loop_var_replace_map = get_loop_tree(expr)

    # assume single tree for now
    shape = utils.just_one(get_shape(expr))

    if shape is UNIT_AXIS_TREE:
        axis_tree = loop_tree
    else:
        # Replace any references to the loop indices
        new_node_map = {}
        for path, axis in shape.node_map.items():
            if axis is None:
                new_node_map[path] = None
                continue

            new_components = []
            for component in axis.components:
                new_regions = []
                for region in component.regions:
                    new_size = replace(region.size, loop_var_replace_map)
                    new_regions.append(region.__record_init__(size=new_size))
                new_components.append(component.copy(regions=new_regions))
            new_node_map[path] = axis.copy(components=new_components)
        subtree = AxisTree(new_node_map)
        axis_tree = loop_tree.add_subtree(loop_tree.leaf_path, subtree)

    return axis_tree, loop_var_replace_map


def full_shape(axes):
    """Augment axes with extra axes from the size expressions."""
    from pyop3.expr_visitors import get_shape

    # only deal in axis trees
    axes = axes.materialize()

    shapes = []
    for axis in axes.nodes:
        for component in axis.components:
            for region in component.regions:
                region_shape = utils.just_one(get_shape(region.size))
                if region_shape.size != 1:
                    shapes.append(region_shape)

    existing = frozenset({axis.label for axis in axes.nodes})
    shape_axes = utils.unique(
        axis for shape in shapes for axis in shape.nodes
        if axis.label not in existing
    )
    if shapes:
        fulltree = AxisTree.from_iterable(shape_axes)
        fulltree = fulltree.add_subtree(fulltree.leaf_path, axes)
        return fulltree
    else:
        return axes
