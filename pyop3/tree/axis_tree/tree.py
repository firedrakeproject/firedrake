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
from types import GeneratorType
import typing
from collections.abc import Iterable, Sized, Sequence
from functools import cached_property
from itertools import chain
from typing import Any, FrozenSet, Hashable, Mapping, Optional, Self, Tuple, Union, ClassVar

import cachetools
import numpy as np
from cachetools import cachedmethod
from mpi4py import MPI
from immutabledict import immutabledict as idict
from petsc4py import PETSc

from pyop3.cache import active_scoped_cache, cached_on, CacheMixin
from pyop3.exceptions import InvalidIndexTargetException, Pyop3Exception
from pyop3.dtypes import IntType
from pyop3.sf import DistributedObject, AbstractStarForest, NullStarForest, ParallelAwareObject, StarForest, local_sf, single_star_sf
from pyop3.mpi import collective
from pyop3 import utils
from pyop3.tree.labelled_tree import (
    as_node_map,
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    MutableLabelledTreeMixin,
    accumulate_path,
    as_component_label,
    as_path,
    parent_path,
    postvisit,
    previsit,
)
from pyop3.utils import (
    has_unique_entries,
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

if typing.TYPE_CHECKING:
    from pyop3.expr import LinearDatBufferExpression
    from pyop3.types import *



OWNED_REGION_LABEL = "owned"
GHOST_REGION_LABEL = "ghost"




class ExpectedLinearAxisTreeException(Pyop3Exception):
    ...


class ContextMismatchException(Pyop3Exception):
    pass


class MissingVariableException(Pyop3Exception):
    """Exception raised when information about an axis variable is missing."""


class InvalidExpressionException(Pyop3Exception):
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
        self.context_map = idict(context_map)

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
        return idict(key)

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
        return idict()

    @property
    def context_map(self):
        return idict({idict(): self})


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
    local_size = 1
    depth = 1
    is_linear = True
    is_empty = False
    sf = single_star_sf(MPI.COMM_SELF) # no idea if this is right
    leaf_paths = (idict(),)
    leaf_path = idict()
    nodes = ()
    _all_region_labels = ()

    targets = (idict({idict(): ()}),)  # not sure

    unindexed = property(lambda self: self)
    regionless = property(lambda self: self)

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
        return idict({idict(): 0})

    # @property
    # def targets_acc(self):
    #     return ()

    @property
    def outer_loops(self):
        return ()

    def path_with_nodes(self, node) -> idict:
        assert node is None
        return idict()

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


@utils.frozenrecord()
class AxisComponentRegion:
    size: AxisComponentRegionSizeT
    label: str | None = None

    def __init__(self, size, label=None):
        from pyop3 import as_linear_buffer_expression, Tensor

        # this is a little clumsy
        if isinstance(size, Tensor):
            size = size.concretize()

        object.__setattr__(self, "size", size)
        object.__setattr__(self, "label", label)

        self.__post_init__()

    def __post_init__(self) -> None:
        if isinstance(self.size, numbers.Integral):
            assert self.size >= 0

    def __str__(self) -> str:
        if self.label is None:
            return str(self.size)
        else:
            return f"{{{self.label}: {self.size}}}"


@functools.singledispatch
def _parse_regions(obj: Any) -> AxisComponentSize:
    from pyop3 import Dat
    from pyop3.expr.buffer import LinearDatBufferExpression

    if isinstance(obj, (Dat, LinearDatBufferExpression)):
        return (AxisComponentRegion(obj),)
    else:
        raise TypeError(f"No handler provided for {type(obj).__name__}")


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
def _parse_sf(obj: Any, size) -> AbstractStarForest | None:
    if obj is None:
        return None
    else:
        raise TypeError(f"No handler provided for {type(obj).__name__}")


@_parse_sf.register(AbstractStarForest)
def _(sf: AbstractStarForest, size) -> AbstractStarForest:
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

        if sf is not None:
            assert self.local_size == self.sf.size

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

        # TODO: check the communicator instead?
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
        return utils.strictly_all(r.label is not None for r in self._all_regions)

    @property
    def user_comm(self) -> MPI.Comm | None:
        return self.sf.user_comm if self.sf else None

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


class Axis(LoopIterable, MultiComponentLabelledNode, CacheMixin, ParallelAwareObject):
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
        from .parse import as_axis_tree

        return as_axis_tree(self)(*args)

    def __str__(self) -> str:
        if len(self.components) == 1:
            component_str = str(utils.just_one(self.components))
        else:
            component_str = f"[{', '.join(map(str, self.components))}]"

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

    def matching_component(self, component_label: ComponentLabelT) -> AxisComponent:
        return self.components[self.component_index(component_label)]

    @property
    def user_comm(self) -> MPI.Comm | None:
        return utils.single_comm(self.components, "user_comm", allow_undefined=True)

    @property
    def user_comm(self) -> MPI.Comm | None:
        return utils.single_comm(self.components, "user_comm", allow_undefined=True)

    @property
    def size(self):
        return self._tree.size

    @property
    def local_size(self):
        return self._tree.local_size

    @property
    def count(self):
        """Return the total number of entries in the axis across all axis parts.
        Will fail if axis parts do not have integer counts.
        """
        # hacky but right (no inner shape)
        return self.size

    @cached_property
    def count_per_component(self):
        return idict({c.label: c.count for c in self.components})

    @cached_property
    def owned(self):
        return self._tree.owned.root

    def index(self):
        return self._tree.index()

    def iter(self, **kwargs) -> LoopIndex | GeneratorType[IteratorIndexT]:
        return self._tree.iter(**kwargs)

    @deprecated("as_tree")
    @property
    def axes(self):
        return self.as_tree()

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

    def component_offset(self, component):
        cidx = self.component_index(component)
        return self._component_offsets[cidx]

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
        from .parse import as_axis_component

        if isinstance(components, Mapping):
            return tuple(
                AxisComponent(count, clabel) for clabel, count in components.items()
            )
        elif isinstance(components, Iterable):
            return tuple(as_axis_component(c) for c in components)
        else:
            return (as_axis_component(components),)


@utils.frozenrecord()
class AxisTarget:
    """TODO.

    (this is hard to explain)

    """
    axis: AxisLabelT
    component: AxisComponentLabelT
    expr: ExpressionT

    @property
    def path(self) -> ConcretePathT:
        return idict({self.axis: self.component})

    @property
    def replace_map(self) -> idict[AxisLabelT, ExpressionT]:
        return idict({self.axis: self.expr})


# TODO: implement this so we don't have lists of lists everywhere
class EquivalentAxisTargetSet(tuple):
    pass


class AbstractAxisTree(ContextFreeLoopIterable, LabelledTree, DistributedObject):

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

    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    # TODO: Cache this function.
    def getitem(self, indices, *, strict=False) -> AbstractAxisTree | AxisForest | ContextSensitiveAxisTree:
        from pyop3.tree.index_tree.parse import as_index_forests
        from pyop3.tree.index_tree import index_axes

        if utils.is_ellipsis_type(indices):
            return self


        # key = (indices, strict)
        # if key in self._cache:
        #     return self._cache[key]

        index_forests = as_index_forests(indices, axes=self, strict=strict)

        if len(index_forests) == 1:
            # There is no outer loop context to consider. Needn't return a
            # context sensitive object.
            index_forest = just_one(index_forests.values())

            # Loop over "restricted" index trees. This is necessary because maps
            # can yield multiple equivalent indexed axis trees. For example,
            # closure(cell) can map any of:
            #
            #   "points"  ->  {"points"}
            #   "points"  ->  {"cells", "edges", "vertices"}
            #   "cells"   ->  {"points"}
            #   "cells"   ->  {"cells", "edges", "vertices"}
            #
            # In each case the required arrays are different from each other and the
            # resulting axis tree is also different. Hence in order for things to work
            # we need to consider each of these separately and produce an axis *forest*.
            indexed_axess = []
            for restricted_index_tree in index_forest:
                indexed_axes = index_axes(restricted_index_tree, idict(), self)
                indexed_axess.append(indexed_axes)

            if len(indexed_axess) > 1:
                return AxisForest(indexed_axess)
            else:
                return just_one(indexed_axess)
        else:
            # TODO: This is identical to what happens above, refactor
            axis_tree_context_map = {}
            for loop_context, index_forest in index_forests.items():
                indexed_axess = []
                for index_tree in index_forest:
                    indexed_axes = index_axes(index_tree, idict(), self)
                    indexed_axess.append(indexed_axes)

                if len(indexed_axess) > 1:
                    raise NotImplementedError("Need axis forests")
                else:
                    indexed_axes = just_one(indexed_axess)
                    axis_tree_context_map[loop_context] = indexed_axes
            return ContextSensitiveAxisTree(axis_tree_context_map)

    def as_axis(self) -> Axis:
        return utils.just_one(self.axes)

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

    @cached_property
    def sf(self) -> StarForest:
        from pyop3.tree.axis_tree.parallel import collect_star_forests, concatenate_star_forests

        has_sfs = bool(list(filter(None, (component.sf for axis in self.axes for component in axis.components))))
        if has_sfs:
            sfs = collect_star_forests(self)
            return concatenate_star_forests(sfs)
        else:
            return NullStarForest(self.size)


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
    def source_exprs(self):
        # return self.index_exprs[-1]
        return self._match_path_and_exprs(self)[1]

    @property
    def target_exprs(self):
        # return self.index_exprs[0]
        return self._match_path_and_exprs(self.unindexed)[1]

    @property
    @abc.abstractmethod
    def layouts(self):
        pass

    @property
    @abc.abstractmethod
    def outer_loops(self):
        pass

    @cached_property
    def _matching_target(self):
        return match_target(self, self.unindexed, self.targets)

    def subst_layouts(self):
        return self._subst_layouts_default

    # NOTE: Do we ever want non-leaf subst_layouts?
    @property
    def leaf_subst_layouts(self) -> idict:
        return idict({leaf_path: self.subst_layouts()[leaf_path] for leaf_path in self.leaf_paths})

    @deprecated("iter")
    def index(self) -> LoopIndex:
        return self.iter()

    def iter(self, *, eager=False) -> LoopIndex | GeneratorType[IteratorIndexT]:
        from pyop3 import LoopIndex

        if eager:
            return _iter_axis_tree(self)
        else:
            return LoopIndex(self)

    def as_tree(self) -> Self:
        return self

    def component_size(self, path: PathT, component_label: ComponentLabelT) -> ExpressionT:
        from .visitors import compute_axis_tree_component_size

        return compute_axis_tree_component_size(self, path, component_label)

    def materialize(self):
        """Return a new "unindexed" axis tree with the same shape."""
        return self._materialized

    @property
    @abc.abstractmethod
    def _materialized(self):
        pass

    @property
    @abc.abstractmethod
    def global_numbering(self) -> op3.Dat:
        pass

    @property
    @abc.abstractmethod
    def regionless(self) -> AbstractAxisTree:
        pass

    @property
    def leaf_axis(self):
        return self.node_map[parent_path(self.leaf_path)]

    @property
    def leaf_component(self):
        return self.leaf_axis.component

    @cached_property
    def size(self):
        from pyop3 import Scalar

        if  self.is_empty:
            return 0
        elif self.internal_comm.size > 1:
            return Scalar(self.local_size)
        else:
            return self.local_size

    @cached_property
    def local_size(self):
        from .visitors import compute_axis_tree_size

        return compute_axis_tree_size(self)

    @cached_property
    def max_size(self):
        from pyop3.expr.visitors import max_value

        return max_value(self.local_size)

    @cached_property
    @collective
    def global_size(self):
        return self.comm.allreduce(self.owned.local_size)

    # old
    @cached_property
    def alloc_size(self):
        return self._alloc_size()

    def section(self, path: PathT, component: ComponentT) -> PETSc.Section:
        from pyop3 import Dat

        path = as_path(path)
        axis = self.node_map[path]

        subpath = path | {axis.label: component.label}
        if subpath in self.leaf_paths:
            size_expr = 1
        else:
            size_expr = self.materialize().subtree(subpath).local_size

        size_dat = Dat.empty(axis.linearize(component.label).regionless, dtype=IntType)
        size_dat.assign(size_expr, eager=True)

        sizes = size_dat.buffer.data_ro

        import pyop3.extras.debug
        pyop3.extras.debug.warn_todo("Cythonize")

        section = PETSc.Section().create(comm=self.comm)
        section.setChart(0, component.local_size)
        for point in range(component.local_size):
            section.setDof(point, sizes[point])
        section.setUp()
        return section

    @property
    @utils.deprecated("internal_comm")
    def comm(self) -> MPI.Comm:
        return self.internal_comm

    @cached_property
    def owned(self):
        """Return the owned portion of the axis tree."""
        if self.comm.size == 1:
            return self
        else:
            return self.with_region_label(OWNED_REGION_LABEL)

    def with_region_label(self, region_label: str) -> IndexedAxisTree:
        """TODO"""
        return self.with_region_labels({region_label})

    def with_region_labels(self, region_labels: Sequence[ComponentRegionLabelT], *, allow_missing: bool = False) -> IndexedAxisTree:
        """TODO"""
        if not region_labels:
            return self

        if not allow_missing and set(region_labels) - set(self._all_region_labels):
            raise ValueError

        return self[self._region_slice(region_labels)]

    # NOTE: Unsure if this should be a method
    def _region_slice(self, region_labels: set, *, path: PathT = idict()) -> "IndexTree":
        from pyop3.tree.index_tree import AffineSliceComponent, RegionSliceComponent, IndexTree, Slice

        region_labels = set(region_labels)

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
    def targets(self) -> tuple[idict[ConcretePathT, tuple[AxisTarget, ...]], ...]:
        pass

    @cached_property
    def _subst_layouts_default(self):
        return subst_layouts(self, self._matching_target, self.layouts)

    # NOTE: This is only really used in one place, probably don't need to make a property then
    # TODO: This is really ick, are the targets or leaves outermost?
    @cached_property
    def leaf_target_paths(self) -> ConcretePathT:
        leaf_target_paths_ = []
        for leaf_path in self.leaf_paths:
            AAA = [idict()]
            # as we go down we want to make this bigger and bigger
            for partial_path in accumulate_path(leaf_path):
                newAAA = []
                for ts in self.targets[partial_path]:
                    partial_path = utils.merge_dicts(t.path for t in ts)
                    for a in AAA:
                        newAAA.append(a | partial_path)
                AAA = newAAA

            leaf_target_paths_.append(AAA)

            # also used in compose_targets
            # merged = []
            # for debug in itertools.product(AAA):
            #     for debug2 in itertools.product(*debug):
            #         merged.append(list(chain(*debug2)))
            # breakpoint()
            # leaf_target_paths_.append(merged)
        return utils.freeze(leaf_target_paths_)

        # for target in self.targets:
        #     leaf_target_paths_per_target = utils.StrictlyUniqueDict()
        #     for leaf_path in self.leaf_paths:
        #         leaf_target_path = utils.StrictlyUniqueDict()
        #         for partial_path in accumulate_path(leaf_path):
        #             for t in target[partial_path]:
        #                 leaf_target_path[t.axis] = t.component
        #         leaf_target_paths_per_target[leaf_path] = leaf_target_path
        #     leaf_target_paths_.append(leaf_target_paths_per_target)
        # return utils.freeze(leaf_target_paths_)

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
            try:
                block_axis = utils.single_valued(blocked_tree.leaves)
            except:
                breakpoint()
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
    def lgmap(self, block_shape: tuple[int, ...] = ()) -> PETSc.LGMap:
        assert False, "old code I think"
        blocked_axes = self.blocked(block_shape)
        indices = blocked_axes.global_numbering
        bsize = np.prod(block_shape, dtype=int)
        return PETSc.LGMap().create(indices, bsize=bsize, comm=self.comm)

    # }}}


@utils.frozenrecord()
class AxisTree(MutableLabelledTreeMixin, AbstractAxisTree):

    # {{{ instance attrs

    _node_map: idict

    def __init__(self, node_map: Mapping[PathT, Node] | None | None = None) -> None:
        object.__setattr__(self, "_node_map", as_node_map(node_map))

    # }}}

    # {{{ interface impls

    node_map = utils.attr("_node_map")

    @property
    def unindexed(self) -> AxisTree:
        return self

    @cached_property
    def targets(self) -> idict[ConcretePathT, tuple[tuple[AxisTarget, ...], ...]]:
        from pyop3 import AxisVar

        targets_ = utils.StrictlyUniqueDict({idict(): ((),)})
        for path, axis in self.node_map.items():
            if axis is None:
                continue

            for component in axis.components:
                path_ = path | {axis.label: component.label}
                expr = AxisVar(axis.linearize(component.label).regionless)
                target = AxisTarget(axis.label, component.label, expr)
                targets_[path_] = [[target]]
        return utils.freeze(targets_)

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

    @property
    def user_comm(self):
        return utils.single_comm(self.nodes, "user_comm", allow_undefined=True) or MPI.COMM_SELF

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

    @cached_property
    def paths_and_exprs(self):
        """
        Return the possible paths represented by this tree.
        """
        return frozenset({self._source_path_and_exprs})

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

        if partial:
            axis_tree = axis_tree.add_subtree(axis_tree.leaf_path, self.subtree(path))

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
    def layouts(self) -> idict:
        """Initialise the multi-axis by computing the layout functions."""
        from .visitors import compute_layouts

        with active_scoped_cache(self):
            return compute_layouts(self)

    @cached_property
    def _buffer_slice(self) -> slice:
        return slice(self.local_size)

    @cached_property
    def global_numbering(self) -> Dat[IntType]:
        from pyop3 import Dat

        # NOTE: Identical code is used elsewhere
        if not self.comm:  # maybe goes away if we disallow comm=None
            numbering = np.arange(self.size, dtype=IntType)
        else:
            start = self.sf.internal_comm.exscan(self.owned.local_size) or 0
            numbering = np.arange(start, start + self.local_size, dtype=IntType)

            # set ghost entries to -1 to make sure they are overwritten
            # TODO: if config.debug:
            numbering[self.owned.local_size:] = -1
            self.sf.broadcast(numbering, MPI.REPLACE)
            debug_assert(lambda: (numbering >= 0).all())
        return Dat(self, data=numbering)


@utils.frozenrecord()
class IndexedAxisTree(AbstractAxisTree):

    # {{{ instance attrs

    _node_map: idict[ConcretePathT, Axis]
    # NOTE: It is OK for unindexed to be None, then we just have a map-like thing
    _unindexed: AxisTree | None
    _targets: tuple[idict[ConcretePathT, tuple[AxisTarget, ...]], ...]

    # TODO: where to put *, and order?
    def __init__(
        self,
        node_map,
        unindexed,
        *,
        targets,
    ):
        if isinstance(node_map, AxisTree):
            node_map = node_map.node_map
        else:
            node_map = as_node_map(node_map)

        # for consistency we don't want missing bits, even if they are empty
        if idict() not in targets:
            targets = targets | {idict(): ((),)}
        # new_targets = []
        # for target in targets:
        #     if idict() not in target:
        #         target = {idict(): ()} | target
        #     new_targets.append(target)
        # # drop duplicate entries as they are necessarily equivalent
        # # TODO: remove this ideally...
        # targets = utils.unique(new_targets)
        # targets = utils.freeze(targets)
        # assert len(targets) > 0

        # debugging
        assert isinstance(targets, idict)
        for vs in targets.values():
            assert isinstance(vs, tuple)
            assert all(isinstance(v_, AxisTarget) for v in vs for v_ in v)

        object.__setattr__(self, "_node_map", node_map)
        object.__setattr__(self, "_unindexed", unindexed)
        object.__setattr__(self, "_targets", targets)

    # }}}

    # {{{ interface impls

    node_map = utils.attr("_node_map")
    unindexed = utils.attr("_unindexed")

    @cached_property
    def targets(self) -> tuple[idict[ConcretePathT, tuple[AxisTarget, ...]], ...]:
        targets_ = utils.StrictlyUniqueDict()
        for path, axis in self.node_map.items():
            targets_[path] = utils.unique(self._targets[path] + self._materialized.targets[path])
        return complete_axis_targets(targets_)

    # TODO: Should this return LoopIndexVars?
    # TODO: We should check the sizes of the axes for loop indices too (and for AxisTrees)
    @cached_property
    def outer_loops(self) -> tuple[LoopIndex, ...]:
        return gather_loop_indices_from_targets(self.targets)

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
        )

    # TODO: Should have nest indices and nest labels as separate concepts.
    # The former is useful for buffers and the latter for trees
    @cached_property
    def nest_indices(self) -> tuple[int, ...]:
        # Compare the 'fully indexed' bits of the matching target and try to
        # match to the unindexed tree.
        consumed_axes = dict(utils.merge_dicts(t.path for t in self._matching_target[idict()]))

        nest_indices_ = []
        path = idict()
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
        subtree_targets = trim_axis_targets(self.targets, {self.unindexed.root.label})

        return IndexedAxisTree(
            self.node_map,
            unindexed=subtree_unindexed,
            targets=subtree_targets,
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
        targets_blocked = trim_axis_targets(self_blocked.targets, block_axis_labels)

        return IndexedAxisTree(
            self_blocked.node_map,
            unindexed=unindexed_blocked,
            targets=targets_blocked,
        )

    # }}}


    @property
    def user_comm(self):
        return self.unindexed.user_comm

    @property
    def layouts(self):
        return self.unindexed.layouts

    def linearize(self, path: PathT, *, partial: bool = False) -> IndexedAxisTree:
        """Return the axis tree dropping all components not specified in the path."""
        path = as_path(path)

        linearized_axis_tree = self.materialize().linearize(path, partial=partial)

        # linearize the targets
        linearized_targets = []

        for orig_target in self.targets:
            linearized_target = {}
            for axis_path, target_spec in orig_target.items():
                if axis_path in linearized_axis_tree.node_map:
                    linearized_target[axis_path] = target_spec
            linearized_target = idict(linearized_target)
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


    # TODO: how do we know if buffer_slice will produce the same object across all ranks?
    # Need to make forming a slice or a subset an active decision!
    @cached_property
    def _buffer_slice(self) -> np.ndarray[IntType]:
        from pyop3 import Dat, do_loop

        # FIXME: parallel!!!
        if self.local_size == 0:
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
        indices = indices_dat.buffer.data_ro

        indices = np.unique(np.sort(indices))

        debug_assert(lambda: min(indices) >= 0 and max(indices) <= self.unindexed.local_size)

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
    def global_numbering(self) -> Dat[IntType]:
        from pyop3 import Dat

        return Dat(self, buffer=self.unindexed.global_numbering.buffer)

    # }}}



# TODO: Choose a suitable base class
class UnitIndexedAxisTree(DistributedObject):
    """An indexed axis tree representing something indexed down to a scalar."""
    def __init__(
        self,
        unindexed,  # allowed to be None
        *,
        targets,
    ):
        if idict() not in targets:
            targets = targets | {idict(): ((),)}
        # same as indexed axis tree
        # new_targets = []
        # for target in targets:
        #     # for consistency we don't want missing bits, even if they are empty
        #     if idict() not in target:
        #         target = {idict(): ()} | target
        #     new_targets.append(target)
        # drop duplicate entries as they are necessarily equivalent
        # TODO: remove this ideally...
        # targets = utils.unique(new_targets)
        # targets = utils.freeze(targets)
        # assert len(targets) > 0

        # debugging
        assert isinstance(targets, idict)
        for vs in targets.values():
            assert isinstance(vs, tuple)
            assert all(isinstance(v_, AxisTarget) for v in vs for v_ in v)

        self.unindexed = unindexed
        self._targets = targets

    @cached_property
    def targets(self) -> tuple[idict[ConcretePathT, tuple[AxisTarget, ...]], ...]:
        raise NotImplementedError("TODO")
        return self._targets + self.materialize().targets

    @property
    def user_comm(self) -> MPI.Comm:
        return self.unindexed.user_comm

    def __str__(self) -> str:
        return "<UNIT>"

    @cached_property
    def outer_loops(self):
        return gather_loop_indices_from_targets(self.targets)

    def materialize(self):
        return UNIT_AXIS_TREE

    size = 1
    is_linear = True
    is_empty = False

    def as_axis(self) -> Axis:
        return Axis(0)

    @property
    def regionless(self):
        return self

    @cached_property
    def _subst_layouts_default(self):
        return subst_layouts(self, self._matching_target, self.unindexed.layouts)

    @property
    def leaf_subst_layouts(self) -> idict:
        return idict({leaf_path: self._subst_layouts_default[leaf_path] for leaf_path in self.leaf_paths})

    subst_layouts = lambda self: self.leaf_subst_layouts

    @property
    def leaf_paths(self):
        return (idict(),)

    # same  as abstract tree case
    @cached_property
    def _matching_target(self):
        return match_target(self, self.unindexed, self.targets)

    @property
    def leaves(self):
        return (None,)

    def path_with_nodes(self, leaf):
        assert leaf is None
        return idict()

    @utils.deprecated()
    @property
    def layouts(self):
        return self.unindexed.layouts

    def with_context(self, context):
        return self

    # TODO: shared with other index tree
    @cached_property
    def nest_indices(self):
        if idict() not in self._matching_target:
            return ()
        consumed_axes = dict(self._matching_target[idict()][0])

        nest_indices_ = []
        path = idict()
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

    def restrict_nest(self, nest_label: ComponentLabelT) -> UnitIndexedAxisTree:
        subtree_unindexed = self.unindexed[nest_label].materialize()

        # remove the nest label from the targets
        subtree_targets = trim_axis_targets(self.targets, {self.unindexed.root.label})

        return UnitIndexedAxisTree(
            unindexed=subtree_unindexed,
            targets=subtree_targets,
        )

    # TODO: shared with other index tree
    @cached_property
    def _matching_target(self):
        return match_target(self, self.unindexed, self.targets)


def find_matching_target(self, target_set):
    assert False, "old code"
    # NOTE: I don't currently know why we still need this, but we apparently do as things otherwise fail
    matching_targets = []
    for target in target_set:
        all_leaves_match = True
        for leaf_path in self.leaf_paths:
            target_path = {}
            for leaf_path_acc in accumulate_path(leaf_path):
                if leaf_path_acc in target:
                    for target_ in target[leaf_path_acc]:
                        target_path[target_.axis] = target_.component
            target_path = idict(target_path)

            # NOTE: We assume that if we get an empty target path then something has
            # gone wrong. This is needed because of .get() calls which are needed
            # because sometimes targets are incomplete.
            # these both work for some cases but not others...
            # if not target_path or not target_path in self.unindexed.leaf_paths:
            if not target_path or not target_path in self.unindexed.node_map:
                all_leaves_match = False
                break

        if all_leaves_match:
            # drop empty mappings as they lead to conflicts
            target = idict({
                key: value
                for key, value in target.items()
                if value != (idict(), idict())
            })
            matching_targets.append(target)

    return utils.single_valued(matching_targets)


class IncompatibleAxisTargetException(Pyop3Exception):
    pass


def match_target(source_axes, target_axes, target_set):
    return _match_target_rec(source_axes, target_axes, target_set, source_path=None, target_path=idict())


def _match_target_rec(source_axes, target_axes, target_set, *, source_path, target_path):
    if source_path is None:
        source_paths = (idict(),)
    else:
        source_axis = source_axes.node_map[source_path]
        source_paths = tuple(
            source_path | {source_axis.label: source_component.label}
            for source_component in source_axis.components
        )

    matching_target = utils.StrictlyUniqueDict()
    for source_path_ in source_paths:
        match_found = False
        for candidate_targets in target_set[source_path_]:
            target_path_ = target_path | merge_dicts(t.path for t in candidate_targets)
            if source_axes.node_map.get(source_path_):
                if not any(target_path_.items() <= leaf_path.items() for leaf_path in target_axes.leaf_paths):
                    continue  # incompatible paths, skip
                try:
                    submatching_target = _match_target_rec(source_axes, target_axes, target_set, source_path=source_path_, target_path=target_path_)
                except IncompatibleAxisTargetException:
                    pass
                else:
                    assert not match_found
                    match_found = True
                    matching_target[source_path_] = candidate_targets
                    matching_target |= submatching_target
            else:  # at a leaf
                if target_path_ in target_axes.leaf_paths:
                    assert not match_found
                    match_found = True
                    matching_target[source_path_] = candidate_targets

        if not match_found:
            raise IncompatibleAxisTargetException
    return utils.freeze(matching_target)


class AxisForest(DistributedObject):
    """A collection of equivalent axis trees.

    Axis forests are useful to describe circumstances where there are multiple
    viable axis trees for describing a layout. For instance, one can view
    the data layout for a function space as a set of DoFs per mesh strata, or
    as a flat set of nodes. These layouts cannot be transformed between each
    other and so must coexist.

    """
    def __init__(self, trees: Sequence[AbstractAxisTree]) -> None:
        # TODO: Should check the trees for compatibility (e.g. do they have the same SF?)
        trees = tuple(trees)

        if not all(isinstance(tree, (AbstractAxisTree, UnitIndexedAxisTree, _UnitAxisTree)) for tree in trees):
            raise TypeError

        self.trees = trees

    def __eq__(self, /, other: Any) -> bool:
        return type(other) is type(self) and other.trees == self.trees

    def __hash__(self) -> int:
        return hash((type(self), self.trees))

    def __repr__(self) -> str:
        return f"AxisForest(({', '.join(repr(tree) for tree in self.trees)}))"

    def __getitem__(self, indices) -> AxisForest | AxisTree:
        return self.getitem(indices, strict=False)

    def getitem(self, indices, *, strict=False):
        if utils.is_ellipsis_type(indices):
            return self

        # FIXME: This will not always work, catch exceptions!
        indexed_trees = []
        for tree in self.trees:
            try:
                indexed_tree = tree.getitem(indices, strict=strict)
                indexed_trees.append(indexed_tree)
            except InvalidIndexTargetException:
                pass

        if not indexed_trees:
            raise RuntimeError("Cannot find any indexable candidates")

        if utils.strictly_all(
            isinstance(indexed_tree, ContextSensitiveAxisTree) for indexed_tree in indexed_trees
        ):
            cs_trees = indexed_trees
            # We currently assume that if things are context sensitive then
            # the loop contexts must be the same in all cases.
            loop_contexts = utils.single_valued(cs_tree.context_map.keys() for cs_tree in cs_trees)
            axis_forest_context_map = collections.defaultdict(list)
            for loop_context in loop_contexts:
                for cs_tree in cs_trees:
                    indexed_tree = cs_tree.context_map[loop_context]
                    if isinstance(indexed_tree, AxisForest):
                        axis_forest_context_map[loop_context].extend(indexed_tree.trees)
                    else:
                        axis_forest_context_map[loop_context].append(indexed_tree)

            # now turns lists into axis forests
            context_map2 = {}
            for loop_context, trees in axis_forest_context_map.items():
                if len(trees) == 1:
                    context_map2[loop_context] = utils.just_one(trees)
                else:
                    context_map2[loop_context] = AxisForest(trees)
            return ContextSensitiveAxisTree(context_map2)

        else:
            indexed_trees_ = []
            for indexed_tree in indexed_trees:
                if isinstance(indexed_tree, AxisForest):
                    indexed_trees_.extend(indexed_tree.trees)
                else:
                    indexed_trees_.append(indexed_tree)

            if len(indexed_trees_) == 1:
                return utils.just_one(indexed_trees_)
            else:
                return AxisForest(indexed_trees_)

    @property
    def user_comm(self) -> MPI.Comm:
        return utils.common_comm(self.trees, "user_comm")

    def materialize(self):
        return type(self)((tree.materialize() for tree in self.trees))

    @cached_property
    def regionless(self):
        return type(self)((tree.regionless for tree in self.trees))

    def prune(self) -> AxisForest:
        return type(self)((tree.prune() for tree in self.trees))

    def blocked(self, block_shape):
        return type(self)(map(operator.methodcaller("blocked", block_shape), self.trees))

    def restrict_nest(self, index):
        return type(self)((tree.restrict_nest(index) for tree in self.trees))

    # def prune(self):
    #     return type(self)((tree.prune() for tree in self.trees))
    #
    # @property
    # def regionless(self):
    #     return type(self)((tree.regionless for tree in self.trees))

    @property
    def size(self):
        return self.trees[0].size

    @property
    def local_size(self) -> int:
        return utils.single_valued((tree.local_size for tree in self.trees))

    @property
    def global_size(self) -> int:
        return utils.single_valued((tree.global_size for tree in self.trees))

    def with_context(self, context):
        return type(self)((tree.with_context(context) for tree in self.trees))

    @cached_property
    def unindexed(self):
        unindexeds = [tree.unindexed for tree in self.trees]
        if utils.is_single_valued(unindexeds):
            return utils.single_valued(unindexeds)
        else:
            return AxisForest(unindexeds)

    @cached_property
    def global_numbering(self) -> Dat:
        from pyop3 import Dat

        return Dat(self, buffer=self.trees[0].global_numbering.buffer)

    @property
    @utils.deprecated("internal_comm")
    def comm(self) -> MPI.Comm:
        return self.internal_comm

    @property
    def owned(self) -> AxisForest:
        return type(self)((tree.owned for tree in self.trees))

    @property
    def _buffer_slice(self):
        return self.trees[0]._buffer_slice


class ContextSensitiveAxisTree(ContextSensitiveLoopIterable, DistributedObject):

    @property
    def user_comm(self) -> MPI.Comm:
        return utils.single_comm(self.context_map.values(), "user_comm")

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
        from pyop3.tree.index_tree import LoopIndex

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


def merge_axis_trees(trees: Iterable[AxisTree]) -> AxisTree:
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


def _merge_trees(tree1, tree2, *, path1=idict(), parents=idict()):
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


def _build_distinct_subtree(axes, parents, *, path=idict()):
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
    path=idict(),
    target_paths_and_exprs_acc=None,
):
    from pyop3.expr.visitors import replace_terminals

    layouts_subst = {}
    # if strictly_all(x is None for x in [axis, path, target_path_acc, index_exprs_acc]):
    if path == idict():
        target_paths_and_exprs_acc = {idict(): targets[idict()]}

        accumulated_path = merge_dicts(t.path for ts in target_paths_and_exprs_acc.values() for t in ts)

        # layouts_subst[path] = replace(layouts[accumulated_path], linear_axes_acc, target_paths_and_exprs_acc)
        replace_map = merge_dicts(t.replace_map for ts in target_paths_and_exprs_acc.values() for t in ts)

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
            path_: targets.get(path_, (idict(), idict()))
        }

        accumulated_path = merge_dicts(t.path for ts in target_paths_and_exprs_acc_.values() for t in ts)
        replace_map = merge_dicts(t.replace_map for ts in target_paths_and_exprs_acc_.values() for t in ts)

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
    return idict(layouts_subst)


def prune_zero_sized_branches(axis_tree: AbstractAxisTree, *, path=idict()) -> AxisTree:
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


# FIXME: This isn't a sufficient check. The regions can be constant sized and the loop index can come from somewhere else... the targets...
def loopify_axis_tree(axis_tree: AbstractAxisTree) -> tuple[AxisTree, Mapping]:
    from pyop3.expr.base import get_loop_tree

    loop_axes = utils.OrderedSet()
    loop_var_replace_map = {}
    replaced_node_map = {}
    for path, axis in axis_tree.node_map.items():
        if axis is None:
            continue

        for component in axis.components:
            for region in component.regions:
                region_loop_tree, region_loop_var_replace_map = get_loop_tree(region.size)
                loop_axes |= region_loop_tree.nodes
                loop_var_replace_map |= region_loop_var_replace_map
        replaced_node_map[path] = replace_exprs(axis, loop_var_replace_map)

    loop_tree = AxisTree.from_iterable(loop_axes)
    loopified_axis_tree = loop_tree.add_subtree(loop_tree.leaf_path, AxisTree(replaced_node_map))

    axis_var_replace_map = utils.invert_mapping(loop_var_replace_map)

    return loopified_axis_tree, axis_var_replace_map


def full_shape(axes):
    """Augment axes with extra axes from the size expressions."""
    from pyop3.expr.visitors import loopified_shape

    # only deal in axis trees
    axes = axes.materialize()

    replace_map = {}

    shapes = []
    for axis in axes.nodes:
        for component in axis.components:
            for region in component.regions:
                region_shape, mymap = loopified_shape(region.size)
                replace_map |= mymap
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
        return fulltree, replace_map
    else:
        return axes, replace_map


def _iter_axis_tree(axis_tree: AbstractAxisTree) -> GeneratorType[IteratorIndexT]:
    if isinstance(axis_tree, IndexedAxisTree):
        raise NotImplementedError("Need to consider targets")

    return _iter_axis_tree_rec(axis_tree, idict(), idict())


def _iter_axis_tree_rec(axis_tree: AbstractAxisTree, path: ConcretePathT, indices: idict[AxisLabelT, int]) -> GeneratorType[IteratorIndexT]:
    from pyop3 import evaluate

    axis = axis_tree.node_map[path]
    for component in axis.components:
        path_ = path | {axis.label: component.label}

        component_size = evaluate(component.size, indices)
        for i in range(component_size):
            indices_ = indices | {axis.label: i}
            if axis_tree.node_map[path_]:
                yield from _iter_axis_tree_rec(axis_tree, path_, indices_)
            else:
                yield (path_, indices_)


@functools.singledispatch
def replace_exprs(treelike, replace_map):
    raise NotImplementedError


@replace_exprs.register(Axis)
def _(axis: Axis, /, replace_map):
    return axis.copy(components=tuple(replace_exprs(c, replace_map) for c in axis.components))


@replace_exprs.register(AxisComponent)
def _(component: AxisComponent, /, replace_map):
    return component.copy(regions=tuple(replace_exprs(r, replace_map) for r in component.regions))


@replace_exprs.register(AxisComponentRegion)
def _(region: AxisComponentRegion, /, replace_map):
    from pyop3.expr.visitors import replace

    return region.__record_init__(size=replace(region.size, replace_map))


def gather_loop_indices_from_targets(targets):
    # NOTE: think this isn't really needed, remove with 'outer_loops'
    from pyop3.expr.visitors import collect_loop_index_vars

    loop_indices = utils.OrderedSet()
    for target in targets:
        for axis_targets in target.values():
            for axis_target in axis_targets:
                for loop_var in collect_loop_index_vars(axis_target.expr):
                    loop_indices.add(loop_var.loop_index)
    return tuple(loop_indices)


def trim_axis_targets(targets, to_trim):
    return tuple(
        {
            path: tuple(
                axis_target
                for axis_target in axis_targets
                if axis_target.axis not in to_trim
            )
            for path, axis_targets in target.items()
        }
        for target in targets
    )


# ContextFreeSingleAxisTreeT = ???
# ContextFreeAxisTreeT = ContextFreeSingleAxisTreeT | AxisForest
# AxisTreeT = ContextFreeAxisTreeT | ContextSensitiveAxisTree


def matching_axis_tree(candidate: ContextFreeAxisTreeT, target: AxisTree | _UnitAxisTree) -> ContextFreeAxisTreeT:
    if isinstance(candidate, AxisForest):
        return next(
            candidate_
            for candidate_ in candidate.trees
            if axis_tree_is_valid_subset(candidate_, target)
        )
    else:
        assert axis_tree_is_valid_subset(candidate, target)
        return candidate


def axis_tree_is_valid_subset(candidate: ContextFreeSingleAxisTreeT, target: ContextFreeSingleAxisTreeT) -> bool:
    """Return if one axis tree may be 'overlaid' on top of another.

    The trees need not exactly match, but they must have the same number of branches.

    """
    target_leaf_paths = set(target.leaf_paths)
    for candidate_leaf_path in candidate.leaf_paths:
        for target_leaf_path in target_leaf_paths:
            if candidate_leaf_path.keys() <= target_leaf_path.keys():
                target_leaf_paths.remove(target_leaf_path)
                break
    return not target_leaf_paths


def complete_axis_targets(targets: idict[ConcretePathT, tuple[tuple]]) -> idict:
    new_targets = dict(targets)
    if idict() not in targets:
        new_targets[idict()] = ((),)
    # drop duplicates
    for path, candidate_axis_targets in targets.items():
        new_targets[path] = utils.unique(candidate_axis_targets)
    return utils.freeze(new_targets)
