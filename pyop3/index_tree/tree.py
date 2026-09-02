from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import functools
import itertools
import numbers
import types
import typing
from collections.abc import Hashable, Iterable, Mapping, Sequence
from functools import cached_property
from typing import Any

import numpy as np
import pymbolic as pym
from immutabledict import immutabledict as idict
from mpi4py import MPI

import pyop3.record
from pyop3 import utils
from pyop3.axis_tree.tree import (
    Axis,
    AxisComponent,
    AxisComponentRegion,
    AxisForest,
    AxisTree,
    GHOST_REGION_LABEL,
    OWNED_REGION_LABEL,
    UNIT_AXIS_TREE,
    AbstractNonUnitAxisTree,
    AxisTarget,
    IndexedAxisTree,
    UnitIndexedAxisTree,
)
from pyop3.collections import StrictlyUniqueDefaultDict, StrictlyUniqueDict, UniqueList
from pyop3.constants import DECIDE
from pyop3.dtypes import IntType
from pyop3.exceptions import InvalidIndexTargetException, Pyop3Exception
from pyop3.labeled_tree import (
    LabeledNodeComponent,
    LabeledTree,
    MultiComponentLabeledNode,
    MutableLabeledTreeMixin,
    accumulate_path,
)
from pyop3.sf import NullStarForest, StarForest


bsearch = pym.var("mybsearch")


class Index(MultiComponentLabeledNode):
    pass


# NOTE: index trees are not really labelled trees. The component labels are always
# nonsense. Instead I think they should just advertise a degree and then attach
# to matching index (instead of label).
@pyop3.record.frozenrecord()
class IndexTree(MutableLabeledTreeMixin, LabeledTree):

    # {{{ instance attrs

    node_map: idict
    _comm: MPI.Comm | None = dataclasses.field(hash=False)

    def __init__(
        self,
        node_map: Mapping[PathT, Node] | None = None,
        *,
        comm: MPI.Comm | None = None,
    ) -> None:
        node_map = self._prepare_node_map(node_map)
        object.__setattr__(self, "node_map", node_map)
        object.__setattr__(self, "_comm", comm)

    # }}}

    @property
    def comm(self) -> MPI.Comm:
        return self._comm if self._comm is not None else super().comm


    # {{{ factory methods

    @classmethod
    def from_iterable(cls, iterable: Iterable, comm: MPI.Comm | None = None) -> Self:
        # NOTE: This is a different parsing approach to that in pyop3/index_tree/parse.py
        # (which is more powerful)
        node_map = cls._node_map_from_iterable(iterable)
        return cls(node_map, comm=comm)

    @classmethod
    def from_nest(cls, nest: Any, *, comm: MPI.Comm | None = None) -> Self:
        node_map = cls._node_map_from_nest(nest)
        return cls(node_map, comm=comm)

    # NOTE: This sort of should live on the Index class and be attached here
    @functools.singledispatchmethod
    @classmethod
    def _as_index(cls, obj: Any, /) -> Index:
        utils.raise_missing_dispatch_handler(obj)

    @_as_index.register(Index)
    @classmethod
    def _(cls, index: Index, /) -> Index:
        return index

    _as_node = _as_index


class SliceComponent(LabeledNodeComponent, abc.ABC):
    @property
    @abc.abstractmethod
    def component(self):
        pass

    @property
    @abc.abstractmethod
    def is_full_slice(self) -> bool:
        pass


@pyop3.record.frozenrecord()
class AffineSliceComponent(SliceComponent):

    # {{{ instance attrs

    _component: ComponentLabelT
    start: numbers.Integral
    stop: numbers.Integral | None
    step: numbers.Integral
    label: ComponentLabelT

    def __init__(
        self,
        component: ComponentLabelT,
        start: numbers.Integral = 0,
        stop: numbers.Integral | None = None,
        step: numbers.Integral = 1,
        label: ComponentLabelT = DECIDE,
    ) -> None:
        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "label", label)

    # }}}

    # {{{ factory methods

    @classmethod
    def from_slice(cls, component: ComponentLabelT, slice_: slice) -> Self:
        start = slice_.start if slice_.start is not None else 0
        stop = slice_.stop
        step = slice_.step if slice_.step is not None else 1
        return cls(component, start, stop, step)

    # }}}

    # {{{ interface impls

    @property
    def component(self):
        return self._component

    @property
    def is_full_slice(self) -> bool:
        return self.start == 0 and self.stop is None and self.step == 1

    # }}}

    # as_range?
    # to_range?
    # should imply the returned type is different!
    def with_size(self, size: numbers.Integral | Dat | None = None) -> tuple:
        if size is None and self.stop is None:
            raise ValueError()

        start = self.start if self.start is not None else 0
        stop = self.stop if self.stop is not None else size
        step = self.step if self.step is not None else 1
        return start, stop, step


@pyop3.record.frozenrecord()
class SubsetSliceComponent(SliceComponent):

    _component: Any
    label: Any
    array: Any

    def __init__(self, component, array, *, label=None) -> None:
        from pyop3.expr import as_linear_buffer_expression

        array = as_linear_buffer_expression(array)

        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "array", array)

    # {{{ interface impls

    @property
    def component(self):
        return self._component

    @property
    def is_full_slice(self) -> bool:
        return False

    # }}}


@pyop3.record.frozenrecord()
class RegionSliceComponent(SliceComponent):
    """A slice component that takes all entries from a particular region.

    This class differs from an affine slice in that it 'consumes' the region
    label, and so breaks any recursive cycle where one might have something
    like `axes.owned.buffer_slice` (which accesses `axes.owned.buffer_slice`...).

    Note that 'region' can be a subset of the region label: e.g. "owned" matches {"owned", "unconstrained"}

    """

    # {{{ instance attrs

    _component: Any
    label: Any
    region: Any

    def __init__(self, component, region: Set, *, label=None) -> None:
        region = frozenset(region)

        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "region", region)

    # }}}

    # {{{ interface impls

    component = pyop3.record.attr("_component")

    @property
    def is_full_slice(self) -> bool:
        return False

    # }}}


class MapComponent(pyop3.obj.Object):

    __abstract_record_attrs = ("target_axis", "target_component", "arity", "label")

    @property
    def target_path(self) -> idict:
        return idict({self.target_axis: self.target_component})


# TODO: Implement AffineMapComponent
@pyop3.record.frozenrecord()
class TabulatedMapComponent(MapComponent):

    target_axis: Any
    target_component: Any
    array: Any
    arity: int
    label: Any

    def __init__(self, target_axis, target_component, array, *, label=DECIDE) -> None:
        from pyop3 import Dat
        from pyop3.expr import as_linear_buffer_expression

        if not isinstance(array, Dat):
            raise NotImplementedError
        assert array.axes.is_linear
        match array.axes.depth:
            case 1:
                arity = 1
            case 2:
                arity = array.axes.leaf_axis.size
            case _:
                raise ValueError

        array = as_linear_buffer_expression(array)

        object.__setattr__(self, "target_axis", target_axis)
        object.__setattr__(self, "target_component", target_component)
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "arity", arity)
        object.__setattr__(self, "label", label)

    # old alias
    @property
    def data(self):
        assert False, "old API"
        return self.array


# NOTE: I don't really remember why this type needs to exist
class AxisIndependentIndex(Index):
    @property
    @abc.abstractmethod
    def axes(self) -> AbstractIndexedAxisTree:
        pass

    @property
    def component_labels(self) -> tuple:
        return tuple(i for i, _ in enumerate(self.axes.leaf_paths))


class UnitIndex(AxisIndependentIndex):
    """An index with unit shape."""

    # {{{ interface impls

    @cached_property
    def axes(self) -> IndexedAxisTree:
        from pyop3.index_tree.apply import _index_axes_per_index

        if not self.is_context_free:
            raise ContextSensitiveException("Expected a context-free index")

        _, targets = _index_axes_per_index(self)
        return UnitIndexedAxisTree(unindexed=None, targets=targets)

    # }}}


class AbstractLoopIndex(pyop3.obj.Object):
    __abstract_record_attrs = ("iterset", "label")

    @abc.abstractmethod
    def context_free(self):
        pass

    # ick, remove
    @property
    def id(self):
        return self.label


@pyop3.record.frozenrecord()
class LoopIndex(AbstractLoopIndex):

    iterset: AbstractNonUnitAxisTree
    label: LabelT

    def get_instruction_executor_cache_key(self, visitor):
        return (
            type(self),
            visitor(self.iterset),
            visitor.renamer.add_type(type(self), self.label),
        )

    def __init__(self, iterset, label=None):
        if label is None:
            label = utils.unique_name(type(self).__name__)
        object.__setattr__(self, "iterset", iterset)
        object.__setattr__(self, "label", label)

    def context_free(self):
        if self.is_context_free:
            return LoopContextFreeLoopIndex(self.iterset, self.label)
        else:
            raise TypeError

    def is_context_free(self):
        return self.iterset.is_linear


@pyop3.record.frozenrecord()
class LoopContextFreeLoopIndex(AbstractLoopIndex, UnitIndex):
    """
    Parameters
    ----------
    iterset: AxisTree or ContextSensitiveAxisTree (!!!)
        Only add context later on

    """

    # {{{ instance attrs

    iterset: AbstractNonUnitAxisTree
    label: LabelT

    def get_disk_cache_key(self, visitor):
        return (
            type(self),
            visitor(self.iterset),
            visitor.renamer.add_type(type(self), self.label),
        )

    get_instruction_executor_cache_key = get_disk_cache_key

    def collect_buffers(self, visitor):
        return visitor(self.iterset)

    def __init__(self, iterset: AbstractNonUnitAxisTree, label=None) -> None:
        if label is None:
            label = utils.unique_name(type(self).__name__)
        object.__setattr__(self, "iterset", iterset)
        object.__setattr__(self, "label", label)

    def __record_post_init(self):
        assert self.iterset.is_linear

    # }}}

    @property
    def comm(self) -> MPI.Comm:
        return self.iterset.comm

    dtype = IntType

    # NOTE: should really just be 'degree' or similar, labels do not really make sense for
    # index trees
    @property
    def component_labels(self) -> tuple:
        return (0,)

    def context_free(self):
        return self

    is_context_free = True

    # TODO: don't think this is useful any more, certainly a confusing name
    @property
    def leaf_target_paths(self):
        """

        Unlike with maps and slices, loop indices are single-component (so return a 1-tuple)
        but that component can target differently labelled axes (so the tuple entry is an n-tuple).

        """
        from pyop3.index_tree.apply import collect_leaf_target_paths
        return collect_leaf_target_paths(self.iterset)


    # NOTE: This is confusing terminology. A loop index can be context-sensitive
    # in two senses:
    # 1. axes.index() is context-sensitive if axes is multi-component
    # 2. axes[p].index() is context-sensitive if p is context-sensitive
    # I think this can be resolved by considering axes[p] and axes as "iterset"
    # and handling that separately.
    def with_context(self, context, *args) -> LoopIndex:
        from pyop3.index_tree.parse import _as_context_free_indices
        return utils.just_one(_as_context_free_indices(self, context))


class InvalidIterationSetException(Pyop3Exception):
    pass


@pyop3.record.frozenrecord()
class ScalarIndex(UnitIndex):

    axis: AxisLabelT
    component: ComponentLabelT
    value: Any
    label: LabelT

    def __init__(self, axis, component, value, label=DECIDE) -> None:
        if label is DECIDE:
            label = f"{axis}__scalar"

        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "label", label)

    @property
    def leaf_target_paths(self):
        return ((idict({self.axis: self.component}),),)

    @property
    def component_labels(self) -> tuple:
        return ("0",)


if typing.TYPE_CHECKING:
    SliceComponentsT = (
        Sequence[SliceComponent]
        | SliceComponent
        | Mapping[ComponentLabelT, Any]
        | ComponentLabelT
    )


@pyop3.record.frozenrecord()
class Slice(Index):
    """

    A slice can be thought of as a map from a smaller space to the target space.

    Like maps it can also target multiple outputs. This is useful for multi-component
    axes.

    """

    # {{{ instance attrs

    axis: AxisLabelT
    components: SliceComponentsT
    label: AxisLabelT

    def __init__(
        self,
        axis: AxisLabelT,
        components: SliceComponentsT,
        *,
        label=DECIDE,
    ):
        from pyop3.index_tree.parse import _parse_slice_components

        if label == axis:
            # can only be set privately
            raise ValueError("The axis and slice labels should not match")

        components = _parse_slice_components(components)
        # Detect a full slice and relabel accordingly
        if (
            label is DECIDE
            and all(
                c.is_full_slice and c.label is DECIDE
                for c in components
            )
        ):
            label = axis
            components = tuple(
                c.record_new(label=c.component)
                for c in components
            )
        else:
            if label is DECIDE:
                # We want a deterministic yet private label for slices
                label = f"{axis}__slice"

            if any(c.label is DECIDE for c in components):
                if not all(c.label is DECIDE for c in components):
                    raise ValueError(
                        "Either none or all slice components can be labeled "
                        "DECIDE"
                    )

                if len(components) == 1:
                    component_labels = [None]
                else:
                    component_labels = range(len(components))
                components = tuple(
                    c.record_new(label=l)
                    for c, l in zip(components, component_labels, strict=True)
                )

        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "label", label)

    def __record_post_init(self) -> None:
        assert all(c.label is not DECIDE for c in self.components)

    # }}}

    @property
    def component_labels(self) -> tuple:
        return tuple(s.label for s in self.components)

    @cached_property
    def leaf_target_paths(self):
        # We return a collection of 1-tuples because each slice component
        # targets only a single (axis, component) pair. There are no
        # 'equivalent' target paths.
        return tuple(
            (idict({self.axis: subslice.component}),)
            for subslice in self.components
        )

    @property
    def expanded(self) -> tuple:
        return (self,)

    def restrict(self, paths):
        new_slice_components = []
        for path in paths:
            found = False
            for slice_component in self.components:
                if idict({self.label: slice_component.label}) == path:
                    new_slice_components.append(slice_component)
                    found = True
            if not found:
                raise ValueError("Invalid path provided")

        return type(self)(self.axis, new_slice_components, label=self.label)


class AbstractMap(pyop3.obj.Object):

    __abstract_record_attrs = ("connectivity",)

    @abc.abstractmethod
    def __call__(self, index, /, **kwargs):
         pass


@pyop3.record.frozenrecord()
class Map(AbstractMap):
    """

    Parameters
    ----------
    connectivity :
        The mappings from input to output for the map. This must be provided as
        an iterable of mappings because the map can both map from *entirely different*
        indices (e.g. multi-component loops that expand to different
        context-free indices) and *semantically equivalent* indices (e.g. a loop
        over ``axes[subset].index()`` has two possible sets of paths and index
        expressions and the map may map from one or both of these but the
        result should be the same). Accordingly, the ``connectivity`` argument
        should provide the different indices as different entries in the iterable,
        and the equivalent indices as different entries in each mapping.

        NOTE: I think this is dead now

        In fact I think to understand the situation we need to consider the following:

        closure(mesh.cells.index()) is hard because mesh.cells is an indexed view of mesh.points,
        and so the loop index carries information on both about. We can feasibly have
        closure(point) AND closure(cell) being separately valid mappings and we don't know
        which we want until we have a target set of axes to make a choice. We therefore want
        to propagate both for as long as possible.
        We could similarly imagine a scenario where closure(cell) yields POINTS, not cells,
        edges and vertices. What do we do then??? That is similar in that we get different
        axis trees that we want to propagate to the end!

        With this in mind, connectivity is therefore the map:

        {
            input_index_label: [
                [*possible component outputs],
                [*possible component outputs]
            ]
        }

        for example, closure gives
        {
            points: [
                [points],
            ]
            cells: [
                [cells, edges, vertices],
                [points],
            ]
            edges: [
                [edges, vertices],
                [points],
            ]
            ...
        }

        but this is really hard because indexing things now gives different AXIS TREES,
        not just different expressions! Indexing therefore must produce an axis forest...

    """

    connectivity: idict

    def __init__(self, connectivity) -> None:
        new_connectivity = {}
        for key, map_cptss in connectivity.items():
            new_map_cptss = []
            for map_cpts in map_cptss:
                if utils.strictly_all(mc.label is DECIDE for mc in map_cpts):
                    new_map_cpts = [
                        mc.record_new(label=i)
                        for i, mc in enumerate(map_cpts)
                    ]
                else:
                    new_map_cpts = map_cpts
                new_map_cptss.append(tuple(new_map_cpts))
            new_connectivity[key] = tuple(new_map_cptss)
        connectivity = idict(new_connectivity)

        object.__setattr__(self, "connectivity", connectivity)

    def __call__(self, index, /, **kwargs) -> CalledMap:
        return CalledMap(self, index, **kwargs)


@pyop3.record.frozenrecord()
class ScalarMap(AbstractMap):
    """An arity 1 map that does not produce an additional axis in the tree."""

    connectivity: idict
    """map connectivity. for each input path it can produce multiple equivalent targets
    (think points vs cells) but never more than one at a time. This differs from other
    map types where for instance the closure of a cell yields multiple result types.

    """

    def __init__(self, connectivity):
        connectivity = utils.freeze(connectivity)
        object.__setattr__(self, "connectivity", connectivity)

    def __record_post_init(self) -> None:
        from pyop3.expr import AxisVar

        # Make sure that 'connectivity' contains the right things
        for entries in self.connectivity.values():
            for entry in entries:
                assert isinstance(entry, MapComponent)
                assert entry.label is not DECIDE
                assert entry.arity == 1
                # hacky way to catch if we are passing in something flat or not
                assert isinstance(entry.array.layout, AxisVar)

    def __call__(self, index, /, **kwargs) -> UnitCalledMap:
        return UnitCalledMap(self, index, **kwargs)


# TODO: I think these parent types are no longer used/useful
class AbstractCalledMap(AxisIndependentIndex):

    __abstract_record_attrs = ("map", "index", "label")

    @property
    def connectivity(self):
        return self.map.connectivity

    # NOTE: nothing about this is specific to an index/map
    @property
    def leaf_target_paths(self) -> tuple:
        from pyop3.index_tree.apply import collect_leaf_target_paths
        return collect_leaf_target_paths(self.axes)

    @property
    def is_context_free(self) -> bool:
        return self.index.is_context_free


@pyop3.record.frozenrecord()
class CalledMap(AbstractCalledMap):

    # {{{ instance attrs

    map: Map
    index: Any
    label: Any

    def __init__(self, map, index, *, label=DECIDE) -> None:
        if label is DECIDE:
            label = utils.unique_name("map")

        object.__setattr__(self, "map", map)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "label", label)

    def __record_post_init(self) -> None:
        # Each leaf of the index wrapped by this map must have at least one
        # target that corresponds to a source for this map.
        for equiv_target_paths in self.index.leaf_target_paths:
            match_found = False
            for equiv_target_path in equiv_target_paths:
                if equiv_target_path in self.map.connectivity:
                    match_found = True
                    break
            if not match_found:
                raise pyop3.exceptions.InvalidMapTargetException(
                    "Cannot find a suitable candidate from the targets of the map index"
                )

    # }}}

    def iter(self, *, eager=False) -> LoopIndex:
        from pyop3.index_tree.apply import collect_leaf_targets
        from pyop3.index_tree.parse import as_index_forests

        if eager:
            raise NotImplementedError

        index_forests = as_index_forests(self)

        if self.is_context_free:
            index_forest = utils.just_one(index_forests.values())

            if len(index_forest) > 1:
                raise NotImplementedError("Need to think about this case")
            else:
                index_tree = utils.just_one(index_forest)

            iterset = index_axes(index_tree)
        else:
            context_map = {}
            for ctx, index_forest in as_index_forests(self).items():
                if len(index_forest) > 1:
                    raise NotImplementedError("Need to think about this case")
                else:
                    index_tree = utils.just_one(index_forest)

                context_map[ctx] = index_axes(index_tree, ctx)
            iterset = ContextSensitiveAxisTree(context_map)
        return LoopIndex(iterset)

    @cached_property
    def axes(self) -> IndexedAxisTree:
        from pyop3.index_tree.apply import collect_leaf_targets, _make_leaf_axis_from_called_map_new

        if not self.is_context_free:
            raise ContextSensitiveException("Expected a context-free index")

        input_axes = self.index.axes
        axes_ = input_axes.materialize()
        # Intermediate targets don't actually target anything
        targets = {
            input_path: ((),)
            for input_path in input_axes.node_map.keys()
        }
        for input_leaf_path, input_leaf_targets_per_leaf in zip(input_axes.leaf_paths, collect_leaf_targets(input_axes), strict=True):
            found = False
            for input_target in input_leaf_targets_per_leaf:
                input_target_path = utils.merge_dicts(t.path for t in input_target)

                if input_target_path in self.connectivity:
                    if len(self.connectivity[input_target_path]) > 1:
                        raise UnspecialisedCalledMapException(
                            "Multiple (equivalent) output paths are generated by the map. "
                            "This ambiguity makes it impossible to form an IndexTree."
                        )

                    output_spec = utils.just_one(self.connectivity[input_target_path])

                    # make a method
                    subaxis, subtargets = _make_leaf_axis_from_called_map_new(
                        self, output_spec, input_target
                    )

                    axes_ = axes_.add_axis(input_leaf_path, subaxis)
                    for subtarget_key, subtarget_value in subtargets.items():
                        targets[input_leaf_path | subtarget_key] = subtarget_value

                    found = True
                    break

            assert found

        targets = utils.freeze(targets)
        return IndexedAxisTree(axes_.node_map, None, targets=targets)


@pyop3.record.frozenrecord()
class UnitCalledMap(UnitIndex, AbstractCalledMap):

    map: UnitMap
    index: UnitMap | LoopIndex
    label: Any

    def __init__(self, map, index, *, label=DECIDE):
        if label is DECIDE:
            label = utils.unique_name("map")
        object.__setattr__(self, "map", map)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "label", label)


class LoopContextSensitive:
    """Class that looks different depending on the loop index branch."""
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

    @cached_property
    def loop_indices(self):
        # all branches must have the same loop indices
        return utils.single_valued(c.keys() for c in self.context_map)

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
