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
from collections import defaultdict
from collections.abc import Iterable, Sized, Sequence
from functools import cached_property
from itertools import chain
from types import NoneType
from typing import Any, FrozenSet, Hashable, Mapping, Optional, Self, Tuple, Union, ClassVar

import cachetools
import numpy as np
from mpi4py import MPI
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.cache
import pyop3.record
from pyop3.cache import cached_on, memory_cache, cached_method
from pyop3.collections import StrictlyUniqueDict, OrderedSet, OrderedFrozenSet
from pyop3.constants import PYOP3_DECIDE
from pyop3.dtypes import IntType
from pyop3.exceptions import InvalidIndexTargetException, Pyop3Exception
from pyop3.sf import DistributedObject, AbstractStarForest, NullStarForest, ParallelAwareObject, StarForest, local_sf, single_star_sf
from pyop3.mpi import collective, temp_internal_comm
from pyop3 import utils
from pyop3.labeled_tree import (
    as_node_map,
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    MutableLabelledTreeMixin,
    accumulate_path,
    as_component_label,
    as_path,
    is_subpath,
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

from ._tree_cy import apply_constraints
from pyop3.device import on_host


if typing.TYPE_CHECKING:
    from pyop3.expr import LinearDatBufferExpression
    from pyop3.types import *


# debugging
mycount = 0
myreprs = set()
seen = set()



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
    def loop_indices(self):
        # all branches must have the same loop indices
        return utils.single_valued(c.keys() for c in self.context_map.keys())

    def with_context(self, context, *, strict=False):
        if not strict:
            context = self.filter_context(context)

        try:
            return self.context_map[context]
        except KeyError:
            raise ContextMismatchException

    def filter_context(self, context):
        return idict({
            loop_index: path
            for loop_index, path in context.items()
            if loop_index in self.loop_indices
        })

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


@pyop3.record.frozenrecord()
class AxisComponentRegion(pyop3.obj.Pyop3Object):

    # {{{ instance attrs

    size: AxisComponentRegionSizeT
    label: frozenset | None = None

    def collect_buffers(self, visitor):
        return visitor(self.size)

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), ("size", visitor(self.size)), ("label", self.label))

    get_instruction_executor_cache_key = get_disk_cache_key

    def __init__(self, size, label=None):
        from pyop3 import as_linear_buffer_expression, Tensor

        if isinstance(label, str):
            label = frozenset({label})

        # this is a little clumsy
        if isinstance(size, Tensor):
            size = size.concretize()

        object.__setattr__(self, "size", size)
        object.__setattr__(self, "label", label)

        self.__post_init__()

    def __post_init__(self) -> None:
        from pyop3 import Scalar
        from pyop3.expr import ScalarBufferExpression

        assert not isinstance(self.label, str), "old API"

        if isinstance(self.size, numbers.Integral):
            assert self.size >= 0
        elif isinstance(self.size, Scalar | ScalarBufferExpression):
            assert self.size.value >= 0

    # }}}

    @property
    def comm(self) -> MPI.Comm:
        if isinstance(self.size, numbers.Integral):
            return MPI.COMM_SELF
        else:
            return self.size.comm

    def __str__(self) -> str:
        if self.label is None:
            return str(self.size)
        else:
            return f"{{{self.label}: {self.size}}}"

    @property
    def local_size(self):
        from pyop3 import evaluate

        try:
            return evaluate(self.size)
        except MissingVariableException:
            return self.size


@functools.singledispatch
def _parse_regions(obj: Any) -> AxisComponentSize:
    from pyop3 import Dat, Scalar
    from pyop3.expr.buffer import LinearDatBufferExpression, ScalarBufferExpression

    if isinstance(obj, (Dat, LinearDatBufferExpression, Scalar, ScalarBufferExpression)):
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


def _partition_regions(regions: Sequence[AxisComponentRegion], sf: AbstractStarForest) -> tuple[AxisComponentRegion, ...]:
    """
    examples:

    (a, 5) and sf: {2 owned and 3 ghost -> (a_owned, 2), (a_ghost, 3)

    (a, 5), (b, 3) and sf: {2 owned and 6 ghost -> (a_owned, 2), (b_owned, 0), (a_ghost, 3), (b_ghost, 3)

    (a, 5), (b, 3) and sf: {6 owned and 2 ghost -> (a_owned, 5), (b_owned, 1), (a_ghost, 0), (b_ghost, 2)
    """
    from pyop3 import Scalar

    region_sizes = {}
    ptr = 0
    for point_type in ["owned", "ghost"]:
        for region in regions:
            if point_type == "owned":
                size = min((region.local_size, sf.num_owned-ptr))
            else:
                size = region.local_size - region_sizes[_as_region_label(region.label, "owned")]
            region_sizes[_as_region_label(region.label, point_type)] = size
            ptr += size
    assert ptr == sf.size
    return tuple(
        AxisComponentRegion(Scalar(size, constant=True), label)
        for label, size in region_sizes.items()
    )


def _as_region_label(initial_region_label: str | None, owned_or_ghost: str):
    if initial_region_label is None:
        return frozenset({owned_or_ghost})
    else:
        raise NotImplementedError("old code I think")
        # could be a frozenset?
        return (initial_region_label, owned_or_ghost)


def _region_label_matches(region, label) -> bool:
    return (
        region.label == label
        or not isinstance(region.label, str | NoneType) and label in region.label
    )


@pyop3.record.frozenrecord()
class AxisComponent(LabelledNodeComponent):
    """
    Parameters
    ----------
    size
        This is useful if we know a-priori that the region sizes sum to something.
        For example the number of unconstrained+constrained dofs will always add to 3.
    """

    # {{{ instance attrs

    regions: Any
    _size: Any
    _label: Any
    sf: Any

    def collect_buffers(self, visitor):
        return OrderedFrozenSet().union(
            *(map(visitor, self.regions)),
            visitor(self._size),
        )

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (
            type(self), ("regions", tuple(map(visitor, self.regions))), ("size", visitor(self._size)), ("label", self.label)
        )

    get_instruction_executor_cache_key = get_disk_cache_key

    def __init__(
        self,
        regions,
        label=utils.PYOP3_DECIDE,
        *,
        sf=None,
        size: Any = None,
    ) -> None:
        from pyop3 import Scalar, evaluate
        from pyop3.expr import ScalarBufferExpression

        regions = _parse_regions(regions)
        if sf is not None:
            if any(
                _region_label_matches(region, label_)
                for region in regions
                for label_ in {OWNED_REGION_LABEL, GHOST_REGION_LABEL}
            ):
                # owned/ghost labels present, regions must be consistent with the SF
                num_owned = 0
                num_ghost = 0
                for region in regions:
                    assert not isinstance(region.size, numbers.Integral)
                    if _region_label_matches(region, OWNED_REGION_LABEL):
                        num_owned += region.local_size
                    else:
                        assert _region_label_matches(region, GHOST_REGION_LABEL)
                        num_ghost += region.local_size
                assert evaluate(num_owned) == sf.num_owned and evaluate(num_ghost) == sf.num_ghost
            else:
                regions = _partition_regions(regions, sf)

        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "_size", size)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "sf", sf)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.sf is not None:
            assert self.local_size == self.sf.size

    # }}}

    # {{{ interface impls

    label = pyop3.record.attr("_label")

    # }}}

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
        assert False, "old code"
        return self.__record_init__(regions=(AxisComponentRegion(self.local_size),), sf=None)

    @property
    def rank_equal(self) -> bool:
        """Return whether or not this axis component has constant size between ranks."""
        raise NotImplementedError

    @property
    @deprecated("size")
    def count(self) -> Any:
        return self.size

    @cached_property
    def size(self) -> ExpressionT:
        if self._size is not None:
            return self._size
        else:
            return sum(r.size for r in self.regions)

    @cached_property
    def local_size(self) -> Any:
        from pyop3 import evaluate

        try:
            return evaluate(self.size)
        except MissingVariableException:
            return self.size

    @cached_property
    def local_max_size(self):
        from pyop3.expr.visitors import get_local_max

        return get_local_max(self.local_size)

    @cached_property
    def _all_regions(self) -> tuple[AxisComponentRegion]:
        assert False, "old code"
        """Return axis component regions having expanded star forests into owned and ghost."""
        return _partition_regions(self.regions, self.sf) if self.sf else self.regions

    @property
    def has_non_trivial_regions(self) ->  bool:
        return len(self.regions) > 1 or utils.just_one(self.regions).label is not None

    @property
    def comm(self) -> MPI.Comm | None:
        return self.sf.comm if self.sf else None

    @property
    def region_labels(self) -> tuple[ComponentRegionLabelT]:
        return tuple(r.label for r in self.regions)

    @property
    def flat_region_labels(self):
        flat = set()
        for l in self.region_labels:
            if l is None:
                continue
            elif isinstance(l, str):
                flat.add(l)
            else:
                flat |= l
        return flat

    # TODO: not used any more?
    @cached_method()
    def localize(self) -> AxisComponent:
        # Region labels are ("owned", "ghost)
        # Want to combine them into a single unlabelled region
        # TODO: implementation is simplified if region labels are always frozensets
        if self.region_labels == (OWNED_REGION_LABEL, GHOST_REGION_LABEL):
            new_region = AxisComponentRegion(sum(r.size for r in self.regions), label=None)
            return self.__record_init__(regions=(new_region,), sf=None)

        # Region labels are ({"owned", "X"}, {"owned", "Y"}, {"ghost", "X"}, {"ghost", "Y"})
        # Want to combine them into two regions ("X", "Y")
        elif utils.strictly_all(
            isinstance(label, frozenset)
            and (OWNED_REGION_LABEL in label or GHOST_REGION_LABEL in label)
            for label in self.region_labels
        ):
            split_regions = collections.defaultdict(list)
            for region in self.regions:
                new_label = region.label - {OWNED_REGION_LABEL, GHOST_REGION_LABEL}
                split_regions[new_label].append(region)

            new_regions = [] 
            for new_label, regions in split_regions.items():
                new_region = AxisComponentRegion(sum(r.size for r in regions), label=new_label)
                new_regions.append(new_region)
            new_regions = tuple(new_regions)
            return self.__record_init__(regions=new_regions, sf=None)

        else:
            assert self.sf is None
            return self

    @cached_method()
    def regionless(self) -> AxisComponent:
        if len(self.regions) > 1:
            merged_region = AxisComponentRegion(sum(r.size for r in self.regions), label=None)
            return self.__record_init__(regions=(merged_region,), sf=None)
        else:
            assert self.sf is None
            return self


@pyop3.record.frozenrecord()
class Axis(LoopIterable, MultiComponentLabelledNode, ParallelAwareObject):

    # {{{ instance attrs

    components: tuple[AxisComponent, ...]
    _label: Any

    def collect_buffers(self, visitor):
        return OrderedFrozenSet().union(*(map(visitor, self.components)))

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), tuple(map(visitor, self.components)), visitor.renamer.add(self._label, "Axis"))

    get_instruction_executor_cache_key = get_disk_cache_key

    def __init__(
        self,
        components,
        label=utils.PYOP3_DECIDE,
    ):
        components = self._parse_components(components)
        # relabel components if needed
        if utils.strictly_all(c.label is utils.PYOP3_DECIDE for c in components):
            if len(components) > 1:
                components = tuple(c.__record_init__(_label=i) for i, c in enumerate(components))
            else:
                components = (utils.just_one(components).__record_init__(_label=None),)

        label = label if label is not PYOP3_DECIDE else self.unique_label()

        object.__setattr__(self, "components", components)
        object.__setattr__(self, "_label", label)
        self.__post_init__()

    def __post_init__(self) -> None:
        assert isinstance(self.components, tuple)
        super().__post_init__()

    # }}}

    label = pyop3.record.attr("_label")

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
        assert component_label in self.component_labels
        if len(self.component_labels) == 1:
            return self
        else:
            return self.__record_init__(components=tuple(c for c in self.components if c.label == component_label))

    @cached_property
    def regionless(self) -> Axis:
        return self.__record_init__(components=tuple(c.regionless for c in self.components))

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
    def comm(self) -> MPI.Comm | None:
        return utils.single_comm(self.components, "comm", allow_undefined=True)

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

    @cached_method()
    def localize(self):
        return self.__record_init__(components=tuple(c.localize() for c in self.components))

    @cached_method()
    def regionless(self):
        return self.__record_init__(components=tuple(c.regionless() for c in self.components))

    @cached_property
    def _tree(self):
        return AxisTree(self)

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


@pyop3.record.frozenrecord()
class AxisTarget(pyop3.obj.Pyop3Object):
    """TODO.

    (this is hard to explain)

    """

    # {{{ instance attrs

    axis: AxisLabelT
    component: AxisComponentLabelT
    expr: ExpressionT

    def collect_buffers(self, visitor) -> OrderedFrozenSet:
        return visitor(self.expr)

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            visitor.renamer.add(self.axis, "Axis"),
            self.component,
            visitor(self.expr),
        )

    get_instruction_executor_cache_key = get_disk_cache_key

    # }}}

    @property
    def path(self) -> ConcretePathT:
        return idict({self.axis: self.component})

    @property
    def replace_map(self) -> idict[AxisLabelT, ExpressionT]:
        return idict({self.axis: self.expr})


# TODO: implement this so we don't have lists of lists everywhere
class EquivalentAxisTargetSet(tuple):
    pass


def _getitem_cache_key(indices, *, strict=False) -> Hashable:
    if isinstance(indices, list):
        indices = tuple(indices)
    return (indices, strict)


class AbstractAxisTreeLike(pyop3.obj.Pyop3Object):
    """Base class for things that look like axis trees or forests."""

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def trees(self) -> tuple[AbstractAxisTree, ...]:
        pass

    @property
    @abc.abstractmethod
    def unindexed(self) -> AbstractAxisTreeLike | None:
        pass

    @property
    @abc.abstractmethod
    def owned(self) -> Self:
        pass

    @property
    @abc.abstractmethod
    def unconstrained(self) -> Self:
        pass

    @abc.abstractmethod
    def with_region_labels(self, *args, **kwargs) -> Self:
        pass

    @property
    @abc.abstractmethod
    def region_sets(self) -> tuple[frozenset[str], ...]:
         pass

    @property
    @abc.abstractmethod
    def buffer_slice(self) -> slice | np.ndarray:
        """Indices of the buffer entries corresponding to this axis tree."""

    @property
    @abc.abstractmethod
    def buffer_size(self) -> int:
        """The number of entries that a buffer built on this axis tree would have.

        Since an axis tree may contain degenerate entries (entries that map to the
        same offsets), this size may be less than the size of the tree itself.

        """

    @property
    @abc.abstractmethod
    def block_shape(self) -> tuple[int, ...]:
        pass

    @property
    @abc.abstractmethod
    def sf(self) -> StarForest:
        pass

    # }}}

    @cached_property
    def free(self) -> Self:
        return self.with_region_labels({"owned", "unconstrained"}, allow_missing=True)

    @property
    def block_size(self) -> int:
        return np.prod(self.block_shape, dtype=int)


class AbstractAxisTree(AbstractAxisTreeLike):
    """Base class for non-forest axis tree types."""

    # {{{ interface impls

    @property
    def trees(self) -> tuple[AbstractAxisTree, ...]:
        return (self,)

    # }}}


class AbstractUnitAxisTree(AbstractAxisTree):
    """Base class for 'unit' (1-sized) axis trees."""

    # {{{ interface impls

    @property
    def owned(self):
         raise NotImplementedError("unsure what to do here, legal?")

    @property
    def unconstrained(self):
         raise NotImplementedError("unsure what to do here, legal?")

    def with_region_labels(self, *args, **kwargs):
         raise NotImplementedError("unsure what to do here, legal?")

    region_sets = ()

    buffer_size = 1
    block_shape = ()

    # }}}

    def __str__(self, /) -> str:
        return "<UNIT>"

    def __contains__(self, obj: Any, /) -> bool:
        return False

    size = 1
    is_linear = True
    is_empty = False

    node_map = idict({idict(): None})


class AbstractNonUnitAxisTree(AbstractAxisTree, ContextFreeLoopIterable, LabelledTree, DistributedObject):
    """Base class for non-unit axis trees."""

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def nest_indices(self) -> tuple[int, ...]:
        pass

    @abc.abstractmethod
    def restrict_nest(self, nest_index: int) -> AbstractNonUnitAxisTree:
        """
        The idea here is to trim ``orig_axes`` with index such that we can pretend
        that the axes always looked truncated in that form.
        """

    @abc.abstractmethod
    def blocked(self, block_shape: Sequence[int, ...]) -> AbstractNonUnitAxisTree:
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

    @cached_property
    def region_sets(self) -> tuple[frozenset[str], ...]:
        # First collect the sets of mutually exclusive region labels. For example this could be
        # '[[{"owned"}, {"ghost"}], [{"unconstrained"}, {"constrained"}]]'.
        mut_excl_region_label_sets = OrderedSet()
        for axis in self.axes:
            for component in axis.components:
                if utils.strictly_all(rl is None for rl in component.region_labels):
                    continue

                # TODO: remove ick casting to frozenset by always making
                # region labels frozensets
                mut_excl_region_label_set = [
                    frozenset({rl}) if isinstance(rl, str) else rl
                    for rl in  component.region_labels
                ]
                mut_excl_region_label_sets.add(mut_excl_region_label_set)

        # Eliminate label sets if they are a strict subset of another set
        # (e.g. {"owned"} vs {"owned", "constrained"})
        mut_excl_region_label_sets = [
            label_set
            for label_set in mut_excl_region_label_sets
            if not any(label_set < label_set_ for label_set_ in mut_excl_region_label_sets)
        ]

        # Now take the product of these mutually exclusive sets to return the actual regions
        merged_regions = []
        for merged_region in itertools.product(*mut_excl_region_label_sets):
            merged_regions.append(frozenset().union(*merged_region))
        return tuple(merged_regions)

    # }}}

    def __getitem__(self, indices):
        return self.getitem(indices, strict=False)

    @cached_method(key=_getitem_cache_key)
    def getitem(self, indices, *, strict=False) -> AbstractNonUnitAxisTree | AxisForest | ContextSensitiveAxisTree:
        from pyop3.index_tree.parse import as_index_forests
        from pyop3.index_tree import index_axes

        if utils.is_ellipsis_type(indices):
            return self

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

    @cached_property
    def block_shape(self) -> tuple[int, ...]:
        from .visitors import get_block_shape

        return get_block_shape(self)

    @property
    @abc.abstractmethod
    def layouts(self):
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
        from pyop3 import Scalar
        from pyop3.expr import ScalarBufferExpression
        from .visitors import compute_axis_tree_component_size

        size = compute_axis_tree_component_size(self, path, component_label)
        if isinstance(size, Scalar | ScalarBufferExpression):
            return size.value
        else:
            return size

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
    def regionless(self) -> AbstractNonUnitAxisTree:
        pass

    @property
    def leaf_axis(self):
        return self.node_map[parent_path(self.leaf_path)]

    @property
    def leaf_component(self):
        return self.leaf_axis.component

    @cached_property
    def size(self):
        from .visitors import compute_axis_tree_size

        return compute_axis_tree_size(self)

    @cached_property
    def local_size(self):
        from pyop3 import evaluate

        try:
            return evaluate(self.size)
        except MissingVariableException:
            return self.size

    @cached_property
    def local_max_size(self) -> numbers.Number:
        from pyop3.expr.visitors import get_local_max

        return get_local_max(self.local_size)

    @cached_property
    @collective
    def global_size(self):
        return self.comm.allreduce(self.owned.local_size)

    @abc.abstractmethod
    def section(self, path: PathT, component: ComponentT, indices=idict()) -> PETSc.Section:
        pass

    @cached_property
    def owned(self):
        """Return the owned portion of the axis tree."""
        # TODO: can i remove this check and apply universally?
        if self.comm.size == 1:
            return self
        else:
            return self.with_region_label(OWNED_REGION_LABEL)

    @cached_property
    def unconstrained(self):
        """Return the unconstrained portion of the axis tree."""
        return self.with_region_label("unconstrained", allow_missing=True)

    def with_region_label(self, region_label: str, *, allow_missing: bool = False) -> IndexedAxisTree:
        """TODO"""
        return self.with_region_labels({region_label}, allow_missing=allow_missing)

    def with_region_labels(self, region_labels: Sequence[ComponentRegionLabelT], *, allow_missing: bool = False) -> IndexedAxisTree:
        """TODO"""
        if not region_labels:
            return self

        # not sure about this
        if not allow_missing and set(region_labels) - set(self._all_region_labels):
            raise ValueError

        return self[self._region_slice(region_labels)]

    def _region_slice(self, region_labels: set, *, path: PathT = idict()) -> "IndexTree":
        from pyop3.index_tree import AffineSliceComponent, RegionSliceComponent, IndexTree, Slice

        region_labels = set(region_labels)

        path = as_path(path)
        axis = self.node_map[path]

        region_label_matches_all_components = True
        region_label_matches_no_components = True
        matching_labels = None
        slice_components = []
        for component in axis.components:
            if matching_labels_ := region_labels & set(component.flat_region_labels):
                matching_labels = matching_labels_
                new_label = f"{component.label}_{'_'.join(map(str, matching_labels))}"
                region_label_matches_no_components = False
                slice_component = RegionSliceComponent(component.label, matching_labels, label=new_label)
            else:
                region_label_matches_all_components = False
                slice_component = AffineSliceComponent(component.label, label=component.label)
            slice_components.append(slice_component)

        # do not change axis label if nothing changes
        if region_label_matches_all_components:
            assert matching_labels is not None
            axis_label = f"{axis.label}_{'_'.join(map(str, matching_labels))}"
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
    @memory_cache(heavy=True, get_comm=lambda self: self.comm)
    def _subst_layouts_default(self):
        return subst_layouts(self, self._matching_target, self.layouts)

    def _alloc_size(self, axis=None):
        if self.is_empty:
            pyop3.debug.warn_todo("think about zero-sized things, should this be allowed?")
            return 1
        axis = axis or self.root
        return sum(cpt.alloc_size(self, axis) for cpt in axis.components)

    # TODO: rename to just _region_labels or similar
    @cached_property
    def _all_region_labels(self) -> tuple[ComponentRegionLabelT]:
        region_labels = OrderedSet()
        for axis in self.axes:
            for component in axis.components:
                for region in component.regions:
                    if region.label is not None:
                        if isinstance(region.label, collections.abc.Set):
                            region_labels.update(region.label)
                        else:
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

    @cached_method()
    def template_vec(self, block_shape: tuple[int, ...]) -> PETSc.Vec:
        """Dummy PETSc Vec of the right size for this set of axes."""
        vec = PETSc.Vec().create(comm=self.comm)
        # As far as PETSc is concerned, the only DoFs that it knows about are those
        # held in the first region (which is 'owned' + 'unconstrained').
        size = self.free.buffer_size
        block_size = np.prod(block_shape, dtype=int)
        vec.setSizes((size, None), bsize=block_size)
        vec.setUp()
        return vec


class AbstractUnindexedAxisTree(AbstractAxisTreeLike):
    """Base class for axis trees that are not indexed."""

    @property
    def unindexed(self) -> Self:
        return self


class AbstractIndexedAxisTree(AbstractAxisTreeLike):
    """Base class for axis trees that are indexed."""

    @cached_property
    def sf(self) -> StarForest:
        petsc_sf = pyop3.sf.filter_petsc_sf(
            self.unindexed.sf.sf, self._buffer_indices, 0, self.local_size
        )
        return StarForest(petsc_sf, self.comm)


# TODO: This should take a comm!
class _UnitAxisTree(AbstractUnitAxisTree, AbstractUnindexedAxisTree):

    # {{{ instance attrs (there aren't any)

    def get_disk_cache_key(self, visitor) -> Hashable:
        return type(self)

    get_instruction_executor_cache_key = get_disk_cache_key

    def collect_buffers(self, visitor):
        return OrderedFrozenSet()

    # }}}

    # {{{ interface impls

    buffer_slice = slice(0, 1, 1)

    # }}}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    local_max_size = 1
    alloc_size = 1
    local_size = 1
    depth = 1
    sf = single_star_sf(MPI.COMM_SELF) # no idea if this is right, probably not since the comm is anything...
    leaf_paths = (idict(),)
    leaf_path = idict()
    nodes = ()
    node_labels = frozenset()
    _all_region_labels = ()
    node_map = idict({idict(): None})

    targets = idict({idict(): ((),)})

    regionless = property(lambda self: self)

    nest_indices = ()

    def _subtree_node_map(self, path: ConcretePathT) -> idict:
        assert not path
        return idict()

    def localize(self):
        return self

    def regionless(self):
        return self

    def prune(self) -> Self:
        return self

    def add_subtree(self, path: PathT, subtree):
        assert not path
        return subtree

    def add_axis(self, path, axis):
        assert not path
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

    def subst_layouts(self):
        return self.leaf_subst_layouts

    def path_with_nodes(self, node) -> idict:
        assert node is None
        return idict()

    def index(self) -> LoopIndex:
        from pyop3 import LoopIndex

        return LoopIndex(self)

    @property
    def comm(self):
        from pyop3.debug import warn_todo
        warn_todo("This comm choice is unsafe")
        return MPI.COMM_SELF



UNIT_AXIS_TREE = _UnitAxisTree()
"""Placeholder value for an axis tree that is guaranteed to have a single entry.

It is useful when handling scalar indices that 'consume' axes because we need a way
to express a tree containing a single entry that does not need to be addressed using
labels.

"""




@pyop3.record.frozenrecord()
class AxisTree(MutableLabelledTreeMixin, AbstractNonUnitAxisTree, AbstractUnindexedAxisTree):

    # {{{ instance attrs

    _node_map: idict

    def collect_buffers(self, visitor):
        return utils.reduce("|", map(visitor, self.node_map.values()), OrderedFrozenSet())

    def get_disk_cache_key(self, visitor) -> Hashable:
        node_map_key = {}
        for path, axis in self._node_map.items():
            node_map_key[visitor.relabel_path(path)] = visitor(axis)
        node_map_key = idict(node_map_key)
        return (type(self), node_map_key)

    get_instruction_executor_cache_key = get_disk_cache_key

    def __init__(self, node_map: Mapping[PathT, Node] | None | None = None) -> None:
        object.__setattr__(self, "_node_map", as_node_map(node_map))

    # }}}

    # {{{ interface impls

    node_map = pyop3.record.attr("_node_map")

    @cached_property
    def targets(self) -> idict[ConcretePathT, tuple[tuple[AxisTarget, ...], ...]]:
        from pyop3 import AxisVar

        targets_ = StrictlyUniqueDict({idict(): ((),)})
        for path, axis in self.node_map.items():
            if axis is None:
                continue

            for component in axis.components:
                path_ = path | {axis.label: component.label}
                expr = AxisVar(axis.linearize(component.label).regionless())
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

    def blocked(self, block_shape: Sequence[int, ...] | int) -> AxisTree:
        if len(block_shape) == 0:
            return self
        else:
            return self[self._block_indices(block_shape)].materialize()

    @property
    def comm(self):
        return utils.single_comm(self.nodes, "comm", allow_undefined=True) or MPI.COMM_SELF

    # TODO: rename to local_section
    def section(self, path: PathT, component: ComponentT, indices=idict()) -> PETSc.Section:
        # NOTE: This is the same as indexedaxistree but offsets are known to increase linearly
        from pyop3 import Dat, loop
        from pyop3.expr.visitors import replace_terminals

        path = as_path(path)
        component_label = as_component_label(component)
        axis = self.node_map[path]
        component = utils.just_one(c for c in axis.components if c.label == component_label)

        # IMPORTANT: If the tree contains constraints then the *local* section
        # is incorrect. This is because constrained DoFs are pushed to the back
        # of the array via axis component regions and therefore
        # 'section.getOffset(constrained_pt)' will give the wrong answer. At
        # present this doesn't seem to be causing any problems because we always
        # constrain all DoFs associated with a point and I would also guess that the
        # local section isn't actually used anywhere.
        # Since sections are not capable of handling interleaved layouts the answer
        # is either that the local section should be NULL or determined in a custom
        # way, but that the global section resulting from the 'invalid' section here
        # should be correct. This currently fails consistency checks inside PETSc.
        # --- UPDATE
        # This approach doesn't seem to work. I think we have to take the approach
        # of disregarding constrained DoFs in the local section. This means only
        # considering DoFs that live in the initial region.
        # if "constrained" in subtree._all_region_labels:
        #     cdat = Dat.zeros(self.regionless(), dtype=IntType)
        #     loop(
        #         p := self.with_region_label("constrained").iter(),
        #         cdat[p].assign(1),
        #         eager=True,
        #     )
        #     constrained = cdat.data_ro
        #     apply_constraints(section, sizes, constrained)

        # TODO: This is a hacky way to do this, better to just take the first region
        # set
        subpath = path | {axis.label: component_label}
        subtree = self.subtree(subpath)
        if "constrained" in subtree._all_region_labels:
            subtree = subtree.with_region_label("unconstrained")

        if subpath in self.leaf_paths:
            size_expr = 1
        else:
            size_expr = replace_terminals(subtree.size, indices)

        size_dat = Dat.empty(axis.linearize(component_label).regionless(), dtype=IntType)
        size_dat.assign(size_expr, eager=True, eager_strategy="compile")
        sizes = size_dat.buffer.data_ro

        section = PETSc.Section().create(comm=self.comm)
        section.setChart(0, component.local_size)
        for point in range(component.local_size):
            section.setDof(point, sizes[point])

        section.setUp()
        return section

    @cached_property
    def sf(self) -> StarForest:
        from pyop3.axis_tree.parallel import collect_star_forests, concatenate_star_forests

        has_sfs = bool(list(filter(None, (component.sf for axis in self.axes for component in axis.components))))
        if has_sfs:
            sfs = collect_star_forests(self)
            return concatenate_star_forests(sfs)
        else:
            return NullStarForest(self.local_size)

    # }}}

    @cached_method()
    def localize(self) -> AxisTree:
        node_map = {
            path: axis.localize() if axis else None
            for path, axis in self.node_map.items()
        }
        return type(self)(node_map)

    @cached_method()
    def regionless(self) -> AxisTree:
        node_map = {
            path: axis.regionless() if axis else None
            for path, axis in self.node_map.items()
        }
        return type(self)(node_map)

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
            linear_axis = axis.linearize(component_label)
            linear_axes.append(linear_axis)

        if linear_axes == self.nodes:
            return self

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

        return compute_layouts(self)

    @property
    def buffer_slice(self) -> slice:
        assert isinstance(self.local_size, numbers.Integral)
        return slice(0, self.local_size, 1)

    @property
    def buffer_size(self) -> int:
        return self.local_size

    # This is a PETSc-specific attribute
    @cached_property
    def global_numbering(self) -> Dat[IntType]:
        from pyop3 import Dat

        # debugging
        self.sf

        with temp_internal_comm(self.comm) as icomm:
            start = icomm.exscan(self.free.local_size) or 0
        numbering = np.arange(start, start + self.local_size, dtype=IntType)

        # set ghost+constrained entries to -1 to make sure they are overwritten
        numbering[self.free.local_size:] = -1
        self.sf.broadcast(numbering, MPI.REPLACE)
        return Dat(self, data=numbering, constant=True)


@pyop3.record.frozenrecord()
class IndexedAxisTree(AbstractNonUnitAxisTree, AbstractIndexedAxisTree):

    # {{{ instance attrs

    _node_map: idict[ConcretePathT, Axis]
    # NOTE: It is OK for unindexed to be None, then we just have a map-like thing
    _unindexed: AxisTree | None
    _targets: tuple[idict[ConcretePathT, tuple[AxisTarget, ...]], ...]

    def collect_buffers(self, visitor) -> OrderedFrozenSet:
        buffers = OrderedFrozenSet()
        for axis in self._node_map.values():
            buffers |= visitor(axis)
        for path, targetss in self._targets.items():
            for targets in targetss:
                for target in targets:
                    buffers |= visitor(target)
        return buffers

    def get_disk_cache_key(self, visitor) -> Hashable:
        raise AssertionError(
            "Indexed axis trees should not be present when we disk cache"
        )
        # below is old
        # When we disk cache things we have already pushed any symbolic
        # information in the targets into the actual expressions. We
        # therefore only care about the shape of things as that affects
        # loop extents.
        # return visitor(self.materialize())

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        node_map_key = {}
        for path, axis in self._node_map.items():
            relabeled_path = idict({
                visitor.renamer.add(axis_label, "Axis"): component_label
                for axis_label, component_label in path.items()
            })
            node_map_key[relabeled_path] = visitor(axis)
        node_map_key = idict(node_map_key)

        targets_key = {}
        for path, targetss in self._targets.items():
            relabeled_path = idict({
                visitor.renamer.add(axis_label, "Axis"): component_label
                for axis_label, component_label in path.items()
            })
            targets_key[relabeled_path] = tuple(
                tuple(visitor(target) for target in targets)
                for targets in targetss
            )
        targets_key = idict(targets_key)

        return (type(self), node_map_key, visitor(self._unindexed), targets_key)

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

        targets = complete_axis_targets(targets)

        object.__setattr__(self, "_node_map", node_map)
        object.__setattr__(self, "_unindexed", unindexed)
        object.__setattr__(self, "_targets", targets)
        self.__post_init__()

    def __post_init__(self) -> None:
        self.targets

    # }}}

    # {{{ interface impls

    node_map = pyop3.record.attr("_node_map")
    unindexed = pyop3.record.attr("_unindexed")

    @cached_property
    def targets(self) -> tuple[idict[ConcretePathT, tuple[AxisTarget, ...]], ...]:
        targets_ = StrictlyUniqueDict()
        for path, axis in self.node_map.items():
            targets_[path] = self._targets[path] + self._materialized.targets[path]
        return complete_axis_targets(targets_)

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

    @cached_method()
    def localize(self):
        return type(self)(
            self.materialize().localize(),
            targets=self.targets,
            unindexed=self.unindexed.localize(),
        )

    @cached_method()
    def regionless(self):
        return type(self)(
            self.materialize().regionless(),
            targets=self.targets,
            unindexed=self.unindexed.regionless(),
        )

    # TODO: Should have nest indices and nest labels as separate concepts.
    # The former is useful for buffers and the latter for trees
    @cached_property
    def nest_indices(self):
        return tuple(index for _, index in self._nest_info)

    @cached_property
    def nest_labels(self):
        return tuple(label for label, _ in self._nest_info)

    @cached_property
    def _nest_info(self) -> tuple:
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
            nest_indices_.append((component_label, component_index))
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

    def section(self, path: PathT, component: ComponentT, indices=idict()) -> PETSc.Section:
        # NOTE: This is the same as axistree but offsets are known not to increase linearly
        # clean this up once we know if works
        from pyop3 import Dat, loop
        from pyop3.expr.visitors import replace_terminals

        path = as_path(path)
        component_label = as_component_label(component)
        axis = self.node_map[path]
        component = utils.just_one(c for c in axis.components if c.label == component_label)

        # IMPORTANT: If the tree contains constraints then the *local* section
        # is incorrect. This is because constrained DoFs are pushed to the back
        # of the array via axis component regions and therefore
        # 'section.getOffset(constrained_pt)' will give the wrong answer. At
        # present this doesn't seem to be causing any problems because we always
        # constrain all DoFs associated with a point and I would also guess that the
        # local section isn't actually used anywhere.
        # Since sections are not capable of handling interleaved layouts the answer
        # is either that the local section should be NULL or determined in a custom
        # way, but that the global section resulting from the 'invalid' section here
        # should be correct. This currently fails consistency checks inside PETSc.
        # --- UPDATE
        # This approach doesn't seem to work. I think we have to take the approach
        # of disregarding constrained DoFs in the local section. This means only
        # considering DoFs that live in the initial region.
        # if "constrained" in subtree._all_region_labels:
        #     cdat = Dat.zeros(self.regionless(), dtype=IntType)
        #     loop(
        #         p := self.with_region_label("constrained").iter(),
        #         cdat[p].assign(1),
        #         eager=True,
        #     )
        #     constrained = cdat.data_ro
        #     apply_constraints(section, sizes, constrained)

        # TODO: This is a hacky way to do this, better to just take the first region
        # set
        subpath = path | {axis.label: component_label}
        subtree = self.materialize().subtree(subpath)
        if "constrained" in subtree._all_region_labels:
            subtree = subtree.with_region_label("unconstrained")

        if subpath in self.leaf_paths:
            size_expr = 1
        else:
            size_expr = replace_terminals(subtree.size, indices)

        offset_expr = replace_terminals(self.subst_layouts()[subpath], indices)

        size_dat = Dat.empty(axis.linearize(component_label).regionless(), dtype=IntType)
        offset_dat = Dat.empty(axis.linearize(component_label).regionless(), dtype=IntType)

        size_dat.assign(size_expr, eager=True, eager_strategy="compile")
        offset_dat.assign(offset_expr, eager=True, eager_strategy="compile")

        sizes = size_dat.buffer.data_ro
        offsets = offset_dat.buffer.data_ro

        section = PETSc.Section().create(comm=self.comm)
        section.setChart(0, component.local_size)
        for point in range(component.local_size):
            section.setDof(point, sizes[point])
            section.setOffset(point, offsets[point])
        return section

    # }}}

    @property
    def comm(self):
        return self.unindexed.comm

    @property
    def layouts(self):
        return self.unindexed.layouts

    def linearize(self, path: PathT, *, partial: bool = False) -> IndexedAxisTree:
        """Return the axis tree dropping all components not specified in the path."""
        path = as_path(path)

        linearized_axis_tree = self.materialize().linearize(path, partial=partial)

        if linearized_axis_tree == self.materialize():
            return self

        linearized_targets = {}
        for partial_path in accumulate_path(path):
            linearized_targets[partial_path] = self.targets[partial_path]
        for path_, target in self.targets.items():
            if path.items() < path_.items():
                linearized_targets[path_] = target

        return IndexedAxisTree(
            linearized_axis_tree, self.unindexed, targets=linearized_targets,
        )

    def materialize(self):
        """Return a new "unindexed" axis tree with the same shape."""
        return AxisTree(self.node_map)

    # TODO: how do we know if buffer_slice will produce the same object across all ranks?
    # Need to make forming a slice or a subset an active decision!
    # TODO: on_host decorator only required while `compile` strategy does not work for device offloading
    @cached_property
    @on_host
    def _buffer_indices(self) -> np.ndarray[IntType]:
        from pyop3 import Dat, do_loop

        if self.size == 0:
            return slice(0, 0)

        # NOTE: The below method might be better...
        # mask_dat = Dat.zeros(self.unindexed.localize(), dtype=bool, prefix="mask")
        # do_loop(p := self.index(), mask_dat[p].assign(1))
        # indices = just_one(np.nonzero(mask_dat.buffer.data_ro))

        indices_dat = Dat.full(self.materialize().regionless(), -1, dtype=IntType, prefix="indices")
        for leaf_path in self.leaf_paths:
            iterset = self.linearize(leaf_path)
            p = iterset.iter()
            offset_expr = just_one(self[p].leaf_subst_layouts.values())
            do_loop(p, indices_dat[p].assign(offset_expr))
        indices = indices_dat.buffer.data_ro
        indices = np.unique(np.sort(indices))

        if len(indices) > 0:
            assert min(indices) >= 0 and max(indices) <= self.unindexed.local_size

        return indices

    @cached_property
    def buffer_slice(self) -> slice | np.ndarray[int]:
        indices = self._buffer_indices

        # then convert to a slice if possible, do in Cython?
        slice_ = None
        n = len(indices)

        if n == 0:
            return slice(0, 0, 1)
        elif n == 1:
            start = indices[0]
            return slice(start, start+1, 1)
        else:
            step = indices[1] - indices[0]

            for i in range(1, n-1):
                new_step = indices[i+1] - indices[i]
                # non-const step, abort and use indices
                if new_step != step:
                    return indices

            return slice(indices[0], indices[-1]+1, step)

    @property
    def buffer_size(self) -> int:
        return self._buffer_indices.size

    # {{{ parallel

    # does this work?
    global_numbering = AxisTree.global_numbering

    # @cached_property
    # def global_numbering(self) -> Dat[IntType]:
    #     from pyop3 import Dat
    #
    #     assert False, "does this work? is it valid?"
    #
    #     return Dat(self.localize(), buffer=self.unindexed.global_numbering.buffer)

    # }}}

    # {{{ PyOP2 migration compat

    # mesh.exterior_facets is now an indexed axis tree
    @property
    def unique_markers(self):
        raise TypeError(
            "'unique_markers' is not a valid attribute in pyop3, you probably "
            "have to use 'mesh.facet_markers' instead"
        )

    # }}}



# TODO: Have an abstract indexed axis tree mixin type
@pyop3.record.frozenrecord()
class UnitIndexedAxisTree(AbstractUnitAxisTree, AbstractIndexedAxisTree):
    """An indexed axis tree representing something indexed down to a scalar."""

    # {{{ instance attrs

    _unindexed: AxisTree | None
    _targets: Any

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        targets_key = {}
        for path, targetss in self._targets.items():
            relabeled_path = idict({
                visitor.renamer.add(axis_label, "Axis"): component_label
                for axis_label, component_label in path.items()
            })
            targets_key[relabeled_path] = tuple(
                tuple(visitor(target) for target in targets)
                for targets in targetss
            )
        targets_key = idict(targets_key)

        return (type(self), visitor(self._unindexed), targets_key)

    def __init__(
        self,
        unindexed: AxisTree | None,
        *,
        targets,
    ):
        if idict() not in targets:
            targets = targets | {idict(): ((),)}

        assert targets.keys() == {idict()}
        object.__setattr__(self, "_unindexed", unindexed)
        object.__setattr__(self, "_targets", targets)
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # }}}

    # {{{ interface impls

    unindexed = pyop3.record.attr("_unindexed")

    @property
    def buffer_slice(self):
        raise NotImplementedError

    # }}}

    def getitem(self, indices, *, strict=False) -> UnitIndexedAxisTree:
        if utils.is_ellipsis_type(indices):
            return self
        else:
            raise InvalidIndexTargetException

    @cached_property
    def targets(self) -> tuple[idict[ConcretePathT, tuple[AxisTarget, ...]], ...]:
        return complete_axis_targets({
            idict(): self._targets[idict()] + self.materialize().targets[idict()]
        })

    @property
    def comm(self) -> MPI.Comm:
        return self.unindexed.comm

    def materialize(self):
        return UNIT_AXIS_TREE

    @cached_method()
    def localize(self):
        return type(self)(
            targets=self.targets,
            unindexed=self.unindexed.localize(),
        )

    @cached_method()
    def regionless(self):
        return type(self)(
            targets=self.targets,
            unindexed=self.unindexed.regionless(),
        )


    def as_axis(self) -> Axis:
        return Axis(0)

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

    @property
    def leaf_path(self):
        return idict()

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

    def with_context(self, context):
        return self

    # TODO: shared with other index tree
    @cached_property
    def nest_indices(self):
        return tuple(index for _, index in self._nest_info)

    @cached_property
    def nest_labels(self):
        return tuple(label for label, _ in self._nest_info)

    @cached_property
    def _nest_info(self):
        if idict() not in self._matching_target:
            return ()

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
            nest_indices_.append((component_label, component_index))
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

    matching_target = StrictlyUniqueDict()
    for source_path_ in source_paths:
        match_found = False
        for candidate_targets in target_set[source_path_]:
            target_path_ = target_path | merge_dicts(t.path for t in candidate_targets)
            if source_axes.node_map.get(source_path_):
                if not any(target_path_.items() <= leaf_path.items() for leaf_path in target_axes.leaf_paths):
                    continue  # incompatible paths, skip
                try:
                    submatching_target = _match_target_rec(source_axes, target_axes, target_set, source_path=source_path_, target_path=target_path_)
                except pyop3.exceptions.IncompatibleAxisTargetException:
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
            raise pyop3.exceptions.IncompatibleAxisTargetException
    return utils.freeze(matching_target)


# TODO: Make a __new__ that returns the single thing if only one tree provided
@pyop3.record.frozenrecord()
class AxisForest(AbstractAxisTreeLike):
    """A collection of equivalent axis trees.

    Axis forests are useful to describe circumstances where there are multiple
    viable axis trees for describing a layout. For instance, one can view
    the data layout for a function space as a set of DoFs per mesh strata, or
    as a flat set of nodes. These layouts cannot be transformed between each
    other and so must coexist.

    """

    # {{{ instance attrs

    _trees: tuple

    def get_instruction_executor_cache_key (self, visitor) -> Hashable:
        return (type(self), tuple(map(visitor, self.trees)))

    def __new__(
        cls,
        trees: Sequence[AbstractNonUnitAxisTree]
    ) -> AbstractAxisTreeLike:
        # eagerly consume iterators, as they are empty if reused
        if isinstance(trees, collections.abc.Iterator):
            trees = tuple(trees)

        # drop duplicates
        unique_trees = utils.unique(trees)

        # return singletons
        if len(unique_trees) == 1:
            return utils.just_one(unique_trees)
        else:
            # no argument modification, build as normal
            self = object.__new__(cls)
            object.__setattr__(self, "_trees", unique_trees)
            return self

    def __init__(self, *args, **kwargs) -> None:
        # To correctly handle generators, which get consumed at first use,
        # we do all initialisation inside __new__ instead of here
        assert hasattr(self, "_trees")
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # }}}

    # {{{ interface impls (AbstractAxisTreeLike)

    trees = pyop3.record.attr("_trees")

    @cached_property
    def unindexed(self) -> AbstractAxisTreeLike | None:
        unindexeds = utils.unique((t.unindexed for t in self.trees))
        if len(unindexeds) == 1:
            return utils.just_one(unindexeds)
        else:
            # TODO: when AxisForest(singleton) -> singleton then this logic can die
            if utils.some_but_not_all((t is None for t in unindexeds)):
                raise ValueError
            return AxisForest(unindexeds)

    @property
    def owned(self) -> AxisForest:
        return self.__record_init__(_trees=tuple(tree.owned for tree in self.trees))

    @property
    def unconstrained(self) -> AxisForest:
        # TODO: nodal axes have labels like {"owned", "unconstrained"} and {"ghost", "unconstrained"}
        # and so .unconstrained is ambiguous. Fixing it is tricky though so for now just discard the
        # rogue axis tree - it will be larger because nothing is getting dropped in the indexing.
        # Better fix: raise an exception about non-contiguous region numbering and drop in a generic way
        # Also: we usually want .free which gets both at once - this may not be needed
        new_trees = [tree.unconstrained for tree in self.trees]
        min_size = min(tree.local_size for tree in new_trees)
        new_trees = tuple(tree for tree in new_trees if tree.local_size == min_size)
        return self.__record_init__(_trees=new_trees)

    def with_region_labels(self, labels, **kwargs) -> AxisForest:
        return type(self)((tree.with_region_labels(labels, **kwargs) for tree in self.trees))

    @cached_property
    def region_sets(self) -> tuple[frozenset[str], ...]:
        return utils.single_valued(t.region_set for t in self.trees)

    @property
    def buffer_slice(self):
        return utils.single_valued(t.buffer_slice for t in self.trees)

    @property
    def buffer_size(self) -> int:
        return utils.single_valued(t.buffer_size for t in self.trees)

    @property
    def block_shape(self) -> tuple[int, ...]:
        # Must use the shortest available block shape
        block_shapes = tuple(tree.block_shape for tree in self.trees)
        min_block_shape_size = min(map(len, block_shapes))
        if min_block_shape_size == 0:
            return ()
        else:
            return utils.single_valued((
                tree.block_shape[-min_block_shape_size:] for tree in self.trees
            ))

    # }}}

    def __str__(self, /) -> str:
        sep = f"\n{'*'*80}\n"
        return sep.join(map(str, self.trees))

    def __getitem__(self, indices) -> AxisForest | AxisTree:
        return self.getitem(indices, strict=False)

    @cached_method(key=_getitem_cache_key)
    def getitem(self, indices, *, strict=False):
        if utils.is_ellipsis_type(indices):
            return self

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
    def comm(self) -> MPI.Comm:
        return utils.common_comm(self.trees, "comm")

    def materialize(self) -> AxisForest:
        return type(self)((tree.materialize() for tree in self.trees))

    def template_vec(self, block_shape):
        return self.trees[0].template_vec(block_shape)

    def localize(self) -> AxisForest:
        return type(self)((tree.localize() for tree in self.trees))

    def regionless(self) -> AxisForest:
        return type(self)((tree.regionless() for tree in self.trees))

    def prune(self) -> AxisForest:
        return type(self)((tree.prune() for tree in self.trees))

    def blocked(self, block_shape):
        return type(self)(map(operator.methodcaller("blocked", block_shape), self.trees))

    def restrict_nest(self, index):
        return type(self)((tree.restrict_nest(index) for tree in self.trees))

    @property
    def nest_indices(self):
        return utils.single_valued((tree.nest_indices for tree in self.trees))

    @property
    def nest_labels(self):
        return utils.single_valued((tree.nest_indices for tree in self.trees))

    @property
    def size(self):
        return self.trees[0].size

    @property
    def sf(self) -> AbstractStarForest:
        return utils.single_valued((tree.sf for tree in self.trees))

    @property
    def local_size(self) -> int:
        return utils.single_valued((tree.local_size for tree in self.trees))

    @property
    def local_max_size(self) -> int:
        return utils.single_valued((tree.local_max_size for tree in self.trees))

    @property
    def global_size(self) -> int:
        return utils.single_valued((tree.global_size for tree in self.trees))

    def with_context(self, context):
        return type(self)((tree.with_context(context) for tree in self.trees))

    @cached_property
    def global_numbering(self) -> Dat:
        from pyop3 import Dat

        buffers = [t.global_numbering.buffer for t in self.trees]
        utils.debug_assert(lambda: utils.is_single_valued(
            (b.data_ro_with_halos for b in buffers)
        ))
        return Dat(self, buffer=buffers[0])


@pyop3.record.frozenrecord()
class ContextSensitiveAxisTree(pyop3.obj.Pyop3Object, ContextSensitiveLoopIterable):

    # {{{ instance attrs

    trees: idict  # context to tree

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        trees_key = {}
        for path, tree in self.trees.items():
            trees_key[visitor.relabel_path(path)] = visitor(tree)
        trees_key = idict(trees_key)
        return (type(self), trees_key)

    def __init__(self, trees: Mapping):
        trees = idict(trees)

        object.__setattr__(self, "trees", trees)
        self.__post_init__()

    def __post_init__(self) -> None:
        assert isinstance(self.trees, Hashable)

    # }}}

    @property
    def context_map(self):  # old alias
        return self.trees

    @property
    def comm(self) -> MPI.Comm:
        return utils.single_comm(self.context_map.values(), "comm")

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
        from pyop3.index_tree import LoopIndex

        return LoopIndex(self)

    @cached_property
    def datamap(self):
        return merge_dicts(axes.datamap for axes in self.context_map.values())

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
    loop_vars=frozenset(),
):
    from pyop3 import NAN
    from pyop3.expr.visitors import replace_terminals, collect_loop_index_vars

    layouts_subst = {}
    # if strictly_all(x is None for x in [axis, path, target_path_acc, index_exprs_acc]):
    if path == idict():
        target_paths_and_exprs_acc = {idict(): targets[idict()]}

        accumulated_path = merge_dicts(t.path for ts in target_paths_and_exprs_acc.values() for t in ts)

        # layouts_subst[path] = replace(layouts[accumulated_path], linear_axes_acc, target_paths_and_exprs_acc)
        replace_map = merge_dicts(t.replace_map for ts in target_paths_and_exprs_acc.values() for t in ts)

        loop_vars |= utils.reduce("|", (set(collect_loop_index_vars(t.expr)) for ts in target_paths_and_exprs_acc.values() for t in ts), set())

        # If we have indexed using a different order to the initial axis tree then sometimes
        # the accumulated path is not valid. In this case do not emit a layout function.
        if accumulated_path in layouts and inner_loop_indices(axes, targets, path) <= loop_vars:
            layouts_subst[path] = replace_terminals(layouts[accumulated_path], replace_map)
        else:
            # if we haven't gone far enough down the tree to have found all of the loop
            # indices then we can't really say that we know what the layout function is.
            layouts_subst[path] = NAN

        if axes.is_empty or axes is UNIT_AXIS_TREE or isinstance(axes, UnitIndexedAxisTree):
            return layouts_subst

    axis = axes.node_map[path]
    for component in axis.components:
        path_ = path | {axis.label: component.label}

        target_paths_and_exprs_acc_ = target_paths_and_exprs_acc | {path_: targets[path_]}

        accumulated_path = merge_dicts(t.path for ts in target_paths_and_exprs_acc_.values() for t in ts)
        replace_map = merge_dicts(t.replace_map for ts in target_paths_and_exprs_acc_.values() for t in ts)
        loop_vars_ = loop_vars | utils.reduce("|", (set(collect_loop_index_vars(t.expr)) for ts in target_paths_and_exprs_acc_.values() for t in ts), set())

        # If we have indexed using a different order to the initial axis tree then sometimes
        # the accumulated path is not valid. In this case do not emit a layout function.
        # if accumulated_path in layouts:
        if accumulated_path in layouts and inner_loop_indices(axes, targets, path) <= loop_vars_:
            layouts_subst[path_] = replace_terminals(layouts[accumulated_path], replace_map)
        else:
            layouts_subst[path_] = NAN

        if axes.node_map[path_]:
            layouts_subst.update(
                subst_layouts(
                    axes,
                    targets,
                    layouts,
                    path=path_,
                    target_paths_and_exprs_acc=target_paths_and_exprs_acc_,
                    loop_vars=loop_vars_,
                )
            )
    return idict(layouts_subst)


# NOTE: likely very inefficient
def inner_loop_indices(axes, targets, path):
    from pyop3.expr.visitors import collect_loop_index_vars

    if path in axes.leaf_paths:
        return set()

    loop_index_vars = set()
    subtree = axes.linearize(path, partial=True)
    for subpath in subtree.node_map:
        for axis_target in targets[path | subpath]:
            loop_index_vars |= set(collect_loop_index_vars(axis_target.expr))
    return loop_index_vars



def prune_zero_sized_branches(axis_tree: AbstractNonUnitAxisTree, *, path=idict()) -> AxisTree:
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
def loopify_axis_tree(axis_tree: AbstractNonUnitAxisTree) -> tuple[AxisTree, Mapping]:
    from pyop3.expr.base import get_loop_tree

    loop_axes = OrderedSet()
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


def _iter_axis_tree(axis_tree: AbstractNonUnitAxisTree) -> GeneratorType[IteratorIndexT]:
    if isinstance(axis_tree, IndexedAxisTree):
        raise NotImplementedError("Need to consider targets")

    return _iter_axis_tree_rec(axis_tree, idict(), idict())


def _iter_axis_tree_rec(axis_tree: AbstractNonUnitAxisTree, path: ConcretePathT, indices: idict[AxisLabelT, int]) -> GeneratorType[IteratorIndexT]:
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
    return axis.__record_init__(components=tuple(replace_exprs(c, replace_map) for c in axis.components))


@replace_exprs.register(AxisComponent)
def _(component: AxisComponent, /, replace_map):
    return component.__record_init__(regions=tuple(replace_exprs(r, replace_map) for r in component.regions))


@replace_exprs.register(AxisComponentRegion)
def _(region: AxisComponentRegion, /, replace_map):
    from pyop3.expr.visitors import replace

    return region.__record_init__(size=replace(region.size, replace_map))


def gather_loop_indices_from_targets(targets):
    # NOTE: think this isn't really needed, remove with 'outer_loops'
    from pyop3.expr.visitors import collect_loop_index_vars

    loop_indices = OrderedSet()
    for axis_targetss in targets.values():
        for axis_targets in axis_targetss:
            for axis_target in axis_targets:
                for loop_var in collect_loop_index_vars(axis_target.expr):
                    loop_indices.add(loop_var.loop_index)
    return tuple(loop_indices)


def trim_axis_targets(targets, to_trim):
    return utils.freeze({
        path: [
            [
                axis_target
                for axis_target in axis_targets
                if axis_target.axis not in to_trim
            ]
            for axis_targets in axis_targetss
        ]
        for path, axis_targetss in targets.items()
    })


# ContextFreeSingleAxisTreeT = ???
# ContextFreeAxisTreeT = ContextFreeSingleAxisTreeT | AxisForest
# AxisTreeT = ContextFreeAxisTreeT | ContextSensitiveAxisTree


def matching_axis_tree(candidate: ContextFreeAxisTreeT, target: AxisTree | _UnitAxisTree) -> ContextFreeAxisTreeT:
    if isinstance(candidate, AxisForest):
        for candidate_ in candidate.trees:
            if axis_tree_is_valid_subset(candidate_, target):
                return candidate_
        else:
            raise AssertionError
    else:
        assert axis_tree_is_valid_subset(candidate, target)
        return candidate


def axis_tree_is_valid_subset(
    candidate: ContextFreeSingleAxisTreeT,
    target: ContextFreeSingleAxisTreeT,
) -> bool:
    """Return if one axis tree may be 'overlaid' on top of another.

    We consider an axis tree to be a valid subset if all of its leaf paths
    have a (unique) matching leaf path in the target tree.

    Parameters
    ----------
    candidate
        The axis tree that may be a subset.
    target
        The (buffer) axis tree to test against.

    Returns
    -------
    bool
        Whether ``candidate`` is a valid subset of ``target``.

    """
    target_leaf_paths = set(target.leaf_paths)
    for candidate_leaf_path in candidate.leaf_paths:
        match_found = False
        for target_leaf_path in target_leaf_paths:
            if is_subpath(candidate_leaf_path, target_leaf_path):
                match_found = True
                target_leaf_paths.remove(target_leaf_path)
                break
        if not match_found:
            return False
    return True


def complete_axis_targets(targets: idict[ConcretePathT, tuple[tuple]]) -> idict:
    new_targets = dict(targets)
    if idict() not in targets:
        new_targets[idict()] = ((),)
    # drop duplicates
    for path, candidate_axis_targets in targets.items():
        new_targets[path] = utils.unique(candidate_axis_targets)
    return utils.freeze(new_targets)
