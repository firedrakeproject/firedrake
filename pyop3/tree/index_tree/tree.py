from __future__ import annotations

import abc
import collections
from collections.abc import Iterable
import dataclasses
import enum
import itertools
import functools
import math
import numbers
import sys
from collections import defaultdict
from functools import cached_property
from itertools import chain
from typing import Any, Collection, Hashable, Mapping, Sequence, Type, cast, Optional

import numpy as np
import pymbolic as pym
from pyop3.exceptions import InvalidIndexTargetException, Pyop3Exception
import pytools
from immutabledict import immutabledict as idict

from pyop3.tree.axis_tree import (
    Axis,
    AxisComponent,
    AxisComponentRegion,
    AxisTree,
    AxisForest,
    LoopIterable,
)
from pyop3.tree.axis_tree.tree import (
    UNIT_AXIS_TREE,
    complete_axis_targets,
    AbstractAxisTree,
    AxisTarget,
    ContextSensitiveLoopIterable,
    IndexedAxisTree,
    UnitIndexedAxisTree,
    OWNED_REGION_LABEL,
    GHOST_REGION_LABEL,
    find_matching_target,
    match_target,
)
from pyop3.dtypes import IntType
from pyop3.sf import NullStarForest, StarForest, local_sf, filter_petsc_sf
from pyop3.tree.labelled_tree import (
    as_node_map,
    LabelledNodeComponent,
    LabelledTree,
    MultiComponentLabelledNode,
    MutableLabelledTreeMixin,
    accumulate_path,
    filter_path,
)
from pyop3.utils import (
    Identified,
    Labelled,
    as_tuple,
    expand_collection_of_iterables,
    single_valued,
    just_one,
    merge_dicts,
    strictly_all,
)
from pyop3 import utils

import pyop3.extras.debug


bsearch = pym.var("mybsearch")

class Index(MultiComponentLabelledNode):
    pass


# NOTE: index trees are not really labelled trees. The component labels are always
# nonsense. Instead I think they should just advertise a degree and then attach
# to matching index (instead of label).
@utils.frozenrecord()
class IndexTree(MutableLabelledTreeMixin, LabelledTree):

    # {{{ instance attrs

    _node_map: idict

    def __init__(self, node_map: Mapping[PathT, Node] | None | None = None) -> None:
        object.__setattr__(self, "_node_map", as_node_map(node_map))

    # }}}

    # {{{ interface impls

    node_map = utils.attr("_node_map")

    @functools.singledispatchmethod
    @classmethod
    def as_node(cls, obj: Any) -> Index:
        raise TypeError(f"No handler defined for {type(obj).__name__}")

    @as_node.register(Index)
    @classmethod
    def _(cls, index: Index) -> Index:
        return index

    # }}}


class SliceComponent(LabelledNodeComponent, abc.ABC):
    @property
    @abc.abstractmethod
    def component(self):
        pass

    @property
    @abc.abstractmethod
    def is_full(self) -> bool:
        pass


@utils.frozenrecord()
class AffineSliceComponent(SliceComponent):

    _component: Any
    start: Any
    stop: Any
    step: Any
    _label: Any
    label_was_none: bool  # old

    # use None for the default args here since that agrees with Python slices
    def __init__(
        self,
        component,
        start: IntType | None = None,
        stop: IntType | None = None,
        step: IntType | None = None,
        *,
        label=None,
        label_was_none=None,
        **kwargs
    ):
        label_was_none = label_was_none or label is None

        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_label", label)

        # TODO: make None here and parse with `with_size()`
        object.__setattr__(self, "start", start if start is not None else 0)
        object.__setattr__(self, "stop", stop)
        # could be None here
        object.__setattr__(self, "step", step if step is not None else 1)

        # hack to force a relabelling
        object.__setattr__(self, "label_was_none", label_was_none)

    # {{{ interface impls

    @property
    def component(self):
        return self._component

    @property
    def label(self):
        return self._label

    @property
    def is_full(self) -> bool:
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


@utils.frozenrecord()
class SubsetSliceComponent(SliceComponent):

    _component: Any
    _label: Any
    array: Any

    def __init__(self, component, array, *, label=None):
        from pyop3.expr import as_linear_buffer_expression

        array = as_linear_buffer_expression(array)

        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "array", array)

    # {{{ interface impls

    @property
    def label(self):
        return self._label

    @property
    def component(self):
        return self._component

    @property
    def is_full(self) -> bool:
        return False

    # }}}


# alternative name, better or worse? I think worse
Subset = SubsetSliceComponent


@utils.frozenrecord()
class RegionSliceComponent(SliceComponent):
    """A slice component that takes all entries from a particular region.

    This class differs from an affine slice in that it 'consumes' the region
    label, and so breaks any recursive cycle where one might have something
    like `axes.owned.buffer_slice` (which accesses `axes.owned.buffer_slice`...).

    """

    # {{{ instance attrs

    _component: Any
    _label: Any
    region: Any

    def __init__(self, component, region: str, *, label=None) -> None:
        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "region", region)

    # }}}

    # {{{ interface impls

    component = utils.attr("_component")
    label = utils.attr("_label")

    @property
    def is_full(self) -> bool:
        return False

    # }}}


@dataclasses.dataclass(frozen=True)
class UnparsedSlice:
    """Placeholder object wrapping arbitrary slice types.

    This class is necessary because the special-casing of tuples in
    ``__getitem__`` by Python breaks the syntactic sugar we have for
    slices. For example consider an axis component with (tuple) label
    '(2, 1)'. We would like to be able to take this slice by executing:

        dat[(2, 1)]

    However, ``__getitem__`` turns this into the very different:

        dat[2, 1]

    """
    wrappee: Any  # TODO: Can specialise the type here


class MapComponent(pytools.ImmutableRecord, Labelled, abc.ABC):
    fields = {"target_axis", "target_component", "label"}

    def __init__(self, target_axis, target_component, *, label=None):
        pytools.ImmutableRecord.__init__(self)
        Labelled.__init__(self, label)
        self.target_axis = target_axis
        self.target_component = target_component

    @property
    @abc.abstractmethod
    def arity(self):
        pass

    @property
    def target_path(self) -> idict:
        return idict({self.target_axis: self.target_component})


# TODO: Implement AffineMapComponent
class TabulatedMapComponent(MapComponent):
    fields = MapComponent.fields | {"array", "arity"}

    def __init__(self, target_axis, target_component, array, *, arity=None, label=None):
        from pyop3.expr import as_linear_buffer_expression

        # determine the arity from the provided array
        if arity is None:
            arity = just_one(array.axes.leaf_component.regions).size

        super().__init__(target_axis, target_component, label=label)
        self.array = as_linear_buffer_expression(array)
        self._arity = arity

    @property
    def arity(self):
        return self._arity

    # old alias
    @property
    def data(self):
        return self.array

    @functools.cached_property
    def datamap(self):
        return self.array.datamap


class AxisIndependentIndex(Index):
    @property
    @abc.abstractmethod
    def axes(self) -> IndexedAxisTree:
        pass

    @property
    def component_labels(self) -> tuple:
        return tuple(i for i, _ in enumerate(self.axes.leaf_paths))


LoopIndexIdT = Hashable


class LoopIndex(Index):
    """
    Parameters
    ----------
    iterset: AxisTree or ContextSensitiveAxisTree (!!!)
        Only add context later on

    """
    dtype = IntType
    fields = Index.fields - {"label"} | {"id"}

    def __init__(self, iterset: AbstractAxisTree, *, id=None):
        self.iterset = iterset
        super().__init__(label=id)

    # TODO: This is very unclear...
    @property
    def id(self):
        return self.label

    @property
    def kernel_dtype(self):
        assert False, "old code"
        return self.dtype

    # NOTE: should really just be 'degree' or similar, labels do not really make sense for
    # index trees
    @property
    def component_labels(self) -> tuple:
        if not self.is_context_free:  # TODO: decorator?
            pyop3.extras.debug.warn_todo("Need a custom context free loop index type - the generic case cannot go in an index tree I think")
            # custom exception type
            # raise ValueError("only valid (context-free) in single component case")

        return (0,)

    @property
    def is_context_free(self):
        return len(self.iterset.leaf_paths) == 1

    @cached_property
    def axes(self) -> IndexedAxisTree:
        from pyop3.expr import LoopIndexVar
        from pyop3.expr.visitors import replace_terminals

        if not self.is_context_free:
            raise ContextSensitiveException("Expected a context-free index")

        _, targets = _index_axes_per_index(self)
        # # need to move the target bit to the outside
        # unpacked_targets = expand_compressed_target_paths(targets)

        return UnitIndexedAxisTree(unindexed=None, targets=targets)

    # TODO: don't think this is useful any more, certainly a confusing name
    @property
    def leaf_target_paths(self):
        """

        Unlike with maps and slices, loop indices are single-component (so return a 1-tuple)
        but that component can target differently labelled axes (so the tuple entry is an n-tuple).

        """
        return collect_leaf_target_paths(self.iterset)


    # NOTE: This is confusing terminology. A loop index can be context-sensitive
    # in two senses:
    # 1. axes.index() is context-sensitive if axes is multi-component
    # 2. axes[p].index() is context-sensitive if p is context-sensitive
    # I think this can be resolved by considering axes[p] and axes as "iterset"
    # and handling that separately.
    def with_context(self, context, *args) -> LoopIndex:
        from pyop3.tree.index_tree.parse import _as_context_free_indices
        return utils.just_one(_as_context_free_indices(self, context))


class InvalidIterationSetException(Pyop3Exception):
    pass


class ScalarIndex(Index):
    fields = {"axis", "component", "value"}

    def __init__(self, axis, component, value):
        super().__init__()
        self.axis = axis
        self.component = component
        self.value = value

    @property
    def leaf_target_paths(self):
        return ((idict({self.axis: self.component}),),)

    @property
    def component_labels(self) -> tuple:
        return ("0",)


# TODO I want a Slice to have "bits" like a Map/CalledMap does
class Slice(Index):
    """

    A slice can be thought of as a map from a smaller space to the target space.

    Like maps it can also target multiple outputs. This is useful for multi-component
    axes.

    """

    fields = Index.fields | {"axis", "components", "label"}

    def __init__(self, axis, components, *, label=None):
        components = as_tuple(components)
        if any(c.label is None for c in components):
            if not all(c.label is None for c in components):
                raise ValueError("Cannot have only some as None")
            components = tuple(c.__record_init__(_label=i) for i, c in enumerate(components))

        self.axis = axis
        self.components = components
        super().__init__(label=label)

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

    @property
    def datamap(self):
        return merge_dicts([s.datamap for s in self.components])


# class DuplicateIndexException(Pyop3Exception):
#     pass


class Map(pytools.ImmutableRecord):
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

    fields = {"connectivity", "name"}

    counter = 0

    def __init__(self, connectivity, name=None) -> None:
        # if not has_unique_entries(k for m in maps for k in m.connectivity.keys()):
        #     raise DuplicateIndexException("The keys for each map given to the multi-map may not clash")

        super().__init__()
        self.connectivity = idict(connectivity)

        # TODO delete entirely
        if name is None:
            # lazy unique name
            name = f"_Map_{self.counter}"
            self.counter += 1
        self.name = name

    def __call__(self, index):
        # If the input index is context-free then we should return something context-free
        # TODO: Should be encoded in some mixin type
        # if isinstance(index, ContextFreeIndex):
        # if isinstance(index, (ContextFreeIndex, ContextFreeCalledMap)):
        if False:
            return ContextFreeCalledMap(self, index)

            # equiv_domainss = tuple(frozenset(mappings.keys()) for mappings in self.connectivity)
            #
            # map_targets = []
            # empty = True
            # for equiv_call_index_targets in index.leaf_target_paths:
            #
            #     domain_index = None
            #     for call_index_target in equiv_call_index_targets:
            #         for i, equiv_domains in enumerate(equiv_domainss):
            #             if call_index_target in equiv_domains:
            #                 assert domain_index in {None, i}
            #                 domain_index = i
            #
            #     if domain_index is None:
            #         continue
            #
            #     empty = False
            #
            #     equiv_mappings = self.connectivity[domain_index]
            #     ntargets = single_valued(len(mcs) for mcs in equiv_mappings.values())
            #
            #     for itarget in range(ntargets):
            #         equiv_map_targets = []
            #         for call_index_target in equiv_call_index_targets:
            #             if call_index_target not in equiv_domainss[domain_index]:
            #                 continue
            #
            #             orig_component = equiv_mappings[call_index_target][itarget]
            #
            #             # We need to be careful with the slice here because the source
            #             # label needs to match the generated axis later on.
            #             orig_array = orig_component.array
            #             leaf_axis, leaf_component_label = orig_array.axes.leaf
            #             myslice = Slice(leaf_axis.label, [AffineSliceComponent(leaf_component_label, label=leaf_component_label)], label=self.name)
            #             newarray = orig_component.array[index, myslice]
            #
            #             indexed_component = orig_component.copy(array=newarray)
            #             equiv_map_targets.append(indexed_component)
            #         equiv_map_targets = tuple(equiv_map_targets)
            #         map_targets.append(equiv_map_targets)
            #
            # if empty:
            #     import warnings
            #     warnings.warn(
            #         "Provided index is not recognised by the map, so the "
            #         "resulting axes will be empty."
            #     )
            #
            # return ContextFreeCalledMap(self, index, map_targets)
        else:
            return CalledMap(self, index)

    @cached_property
    def datamap(self):
        data = {}
        for bit in self.connectivity.values():
            for map_cpt in bit:
                data.update(map_cpt.datamap)
        return idict(data)


class ContextSensitiveException(Pyop3Exception):
    """Exception raised when an index is sensitive to the loop index."""


class UnspecialisedCalledMapException(Pyop3Exception):
    """Exception raised when an unspecialised map is used in place of a specialised one.

    This is important for cases like closure(cell) where the result can be either
    a set of points, or sets of cells, edges, and vertices. We say that it is 'unspecialised'
    because it cannot be put into an `IndexTree` and instead should yield two trees as
    an `IndexForest`.

    """


class CalledMap(AxisIndependentIndex, Identified, Labelled, LoopIterable):
    fields = {"map", "from_index", "id", "label"}

    def __init__(self, map, from_index, *, id=None, label=None):
        Identified.__init__(self, id=id)
        Labelled.__init__(self, label=label)
        self.map = map
        self.index = from_index

        # this is an old alias
        self.from_index = from_index

    def __getitem__(self, indices):
        raise NotImplementedError("TODO")
        # figure out the current loop context, just a single loop index
        # from_index = self.from_index
        # while isinstance(from_index, CalledMap):
        #     from_index = from_index.from_index
        # existing_loop_contexts = tuple(
        #     freeze({from_index.id: path}) for path in from_index.paths
        # )
        #
        # index_forest = {}
        # for existing_context in existing_loop_contexts:
        #     axes = self.with_context(existing_context)
        #     index_forest.update(
        #         as_index_forest(indices, axes=axes, loop_context=existing_context)
        #     )
        #
        # array_per_context = {}
        # for loop_context, index_tree in index_forest.items():
        #     indexed_axes = index_axes(index_tree, loop_context, self.axes)
        #
        #     (
        #         target_paths,
        #         index_exprs,
        #         layout_exprs,
        #     ) = _compose_bits(
        #         self.axes,
        #         self.target_paths,
        #         self.index_exprs,
        #         None,
        #         indexed_axes,
        #         indexed_axes.target_paths,
        #         indexed_axes.index_exprs,
        #         indexed_axes.layout_exprs,
        #     )
        #
        #     array_per_context[loop_context] = Dat(
        #         indexed_axes,
        #         data=self.array,
        #         layouts=self.layouts,
        #         target_paths=target_paths,
        #         index_exprs=index_exprs,
        #         name=self.name,
        #         max_value=self.max_value,
        #     )
        # return ContextSensitiveMultiArray(array_per_context)

    def iter(self, *, eager=False) -> LoopIndex:
        from pyop3.tree.index_tree.parse import as_index_forests

        if eager:
            raise NotImplementedError

        index_forests = as_index_forests(self)

        if self.is_context_free:
            index_forest = just_one(index_forests.values())

            if len(index_forest) > 1:
                raise NotImplementedError("Need to think about this case")
            else:
                index_tree = just_one(index_forest)

            iterset = index_axes(index_tree)
        else:
            context_map = {}
            for ctx, index_forest in as_index_forests(self).items():
                if len(index_forest) > 1:
                    raise NotImplementedError("Need to think about this case")
                else:
                    index_tree = just_one(index_forest)

                context_map[ctx] = index_axes(index_tree, ctx)
            iterset = ContextSensitiveAxisTree(context_map)
        return LoopIndex(iterset)

    @property
    def name(self):
        return self.map.name

    @property
    def connectivity(self):
        return self.map.connectivity

    @cached_property
    def axes(self) -> IndexedAxisTree:
        if not self.is_context_free:
            raise ContextSensitiveException("Expected a context-free index")

        input_axes = self.index.axes
        axes_ = input_axes.materialize()
        targets = {}
        for input_leaf_path, input_leaf_targets_per_leaf in zip(input_axes.leaf_paths, collect_leaf_targets(input_axes), strict=True):
            found = False
            for input_target in input_leaf_targets_per_leaf:
                input_target_path = merge_dicts(t.path for t in input_target)

                if input_target_path in self.connectivity:
                    found = True

                    if len(self.connectivity[input_target_path]) > 1:
                        raise UnspecialisedCalledMapException(
                            "Multiple (equivalent) output paths are generated by the map. "
                            "This ambiguity makes it impossible to form an IndexTree."
                        )

                    output_spec = just_one(self.connectivity[input_target_path])

                    # make a method
                    subaxis, subtargets = _make_leaf_axis_from_called_map_new(
                        self, self.name, output_spec, input_target,
                    )

                    axes_ = axes_.add_axis(input_leaf_path, subaxis)
                    for subtarget_key, subtarget_value in subtargets.items():
                        targets[input_leaf_path | subtarget_key] = subtarget_value

                    # this axis is intermediate, it doesn't target anything
                    targets[input_leaf_path] = ((),)

                    break

            assert found

        targets = utils.freeze(targets)
        # if "closure" in str(axes_):
        #     breakpoint()
        return IndexedAxisTree(axes_.node_map, None, targets=targets)

    @property
    def is_context_free(self) -> bool:
        return self.index.is_context_free

    # NOTE: nothing about this is specific to an index
    @property
    def leaf_target_paths(self) -> tuple:
        return collect_leaf_target_paths(self.axes)

    @cached_property
    def expanded(self):
        """Return a `tuple` of maps specialised to possible inputs and outputs.

        This is necessary because the input index may match with multiple possible
        map inputs, and the map may have multiple possible outputs for each input.

        For example, closure(cell) matches inputs of points and cells, and has output
        cells, edges, and vertices, and separately points.

        """
        restricted_maps = []
        for index in self.call_index.expanded:
            for input_path in index.leaf_target_paths:
                for output_spec in self.connectivity[input_path]:
                    restricted_connectivity = {input_path: (output_spec,)}
                    restricted_map = Map(restricted_connectivity, self.name)(index)
                    restricted_maps.append(restricted_map)
        return tuple(restricted_maps)

    @property
    def _connectivity_dict(self):
        return idict(self.connectivity)

    # TODO cleanup
    def with_context(self, context, axes=None):
        raise NotImplementedError
        # maybe this line isn't needed?
        # cf_index = self.from_index.with_context(context, axes)
        cf_index = self.from_index
        leaf_target_paths = tuple(
            idict({mcpt.target_axis: mcpt.target_component})
            for path in cf_index.leaf_target_paths
            for mcpt in self.map.connectivity[path]
            # if axes is None we are *building* the axes from this map
            if axes is None
            or axes.is_valid_path(
                {mcpt.target_axis: mcpt.target_component}, complete=False
            )
        )
        if len(leaf_target_paths) == 0:
            raise RuntimeError
        return ContextFreeCalledMap(self.map, cf_index, leaf_target_paths, id=self.id)

    @property
    def name(self) -> str:
        return self.map.name


class ContextSensitiveCalledMap(ContextSensitiveLoopIterable):
    pass


class InvalidIndexException(Pyop3Exception):
    pass


@functools.singledispatch
def collect_index_target_paths(index: Index) -> tuple[tuple[idict[str, str], ...], ...]:
    raise TypeError(f"No handler defined for {type(index).__name__}")


@collect_index_target_paths.register(LoopIndex)
def _(loop_index: LoopIndex) -> tuple[tuple[idict[str, str], ...], ...]:
    return loop_index.leaf_target_paths
    # return (
    #     tuple(
    #         accumulate_target_path(iterset_target)
    #         for iterset_target in loop_index.iterset.paths_and_exprs
    #     ),
    # )


@collect_index_target_paths.register(ScalarIndex)
def _(scalar_index: ScalarIndex, /, *args, **kwargs):
    return scalar_index.leaf_target_paths


@collect_index_target_paths.register(Slice)
def _(slice_: Slice) -> tuple[tuple[idict[str, str]], ...]:
    return slice_.leaf_target_paths
    return tuple(
        (idict({slice_.axis: slice_component.component}),)
        for slice_component in slice_.components
    )


@collect_index_target_paths.register(CalledMap)
def _(called_map: CalledMap) -> tuple[tuple[idict[str, str]], ...]:
    return called_map.leaf_target_paths
    # duplicate of elsewhere
    leaf_target_paths_ = []
    for leaf_path in called_map.axes.leaf_paths:
        leaf_target_paths_per_target = []
        for leaf_targets_per_target in called_map.leaf_target_paths:
            leaf_target_paths_per_target.append(leaf_targets_per_target[leaf_path])
        leaf_target_paths_per_target = tuple(leaf_target_paths_per_target)
        leaf_target_paths_.append(leaf_target_paths_per_target)
    return tuple(leaf_target_paths_)
    # compressed_targets = []
    # for leaf_path in called_map.axes.leaf_paths:
    #     compressed_targets.append(tuple(t[leaf_path][0] for t in called_map.axes.targets))
    # return tuple(compressed_targets)


def match_target_paths_to_axis_tree(index_tree, orig_axes):
    target_axes_by_index, leaf_target_axes = match_target_paths_to_axis_tree_rec(index_tree, orig_axes, index_path=idict(), candidate_target_paths_acc=(idict(),))
    assert all(len(leaf_axes) == 0 for leaf_axes in leaf_target_axes), "Expected all axes to be consumed by now"
    return target_axes_by_index


def match_target_paths_to_axis_tree_rec(
    index_tree,
    orig_axes,
    *,
    index_path: ConcretePathT,
    candidate_target_paths_acc,
):
    index = index_tree.node_map[index_path]

    target_axes_by_index = {}
    leaf_target_axes = []
    index_target_paths = collect_index_target_paths(index)
    for equivalent_index_target_paths, index_component_label in zip(index_target_paths, index.component_labels, strict=True):
        equivalent_index_target_paths = list(equivalent_index_target_paths)

        index_path_ = index_path | {index.label: index_component_label}

        candidate_target_paths_acc_ = tuple(
            candidate_path | index_target_path
            for candidate_path in candidate_target_paths_acc 
            for index_target_path in equivalent_index_target_paths
        )
        if not index_tree.node_map[index_path_]:
            # At a leaf, can now determine the axes that are referenced by the path.
            # We only expect a single match from all the collected candidate paths.
            if not any(
                candidate_path in orig_axes.node_map
                for candidate_path in candidate_target_paths_acc_
            ):
                raise InvalidIndexTargetException("Candidates do not target the axis tree")

            full_target_axes = utils.single_valued(
                orig_axes.visited_nodes(candidate_path)
                for candidate_path in candidate_target_paths_acc_
                if candidate_path in orig_axes.node_map
            )
            # convert to a dict so entries can be popped off as we go up
            sub_leaf_target_axess = (dict(full_target_axes),)
        else:
            sub_target_axes_by_index, sub_leaf_target_axess = match_target_paths_to_axis_tree_rec(index_tree, orig_axes, index_path=index_path_, candidate_target_paths_acc=candidate_target_paths_acc_)
            target_axes_by_index |= sub_target_axes_by_index

        # Look at what all the leaves think the axes that are pointed to by this
        # index are and make sure they are consistent.
        selected_axess = tuple(
            idict({
                axis: component_label
                for axis, component_label in sub_leaf_target_axes.items()
                if any(axis.label in index_target_path for index_target_path in equivalent_index_target_paths)
            })
            for sub_leaf_target_axes in sub_leaf_target_axess
        )

        # all subtrees must agree on what this axis represents
        selected_axes = utils.single_valued(selected_axess)
        # remove the selected axes from the leaf paths so they cannot be reused
        for sub_leaf_target_axes in sub_leaf_target_axess:
            for axis in selected_axes.keys():
                sub_leaf_target_axes.pop(axis)

        target_axes_by_index[index_path_] = selected_axes
        leaf_target_axes.extend(sub_leaf_target_axess)

    target_axes_by_index = idict(target_axes_by_index)
    leaf_target_axes = tuple(leaf_target_axes)
    return target_axes_by_index, leaf_target_axes


@functools.singledispatch
def _index_axes_per_index(index: Index, /, *args, **kwargs) -> tuple[AxisTree, tuple, tuple[LoopIndex, ...]]:
    """TODO.

    Case 1: loop indices

    Assume we have ``axis[p]`` with ``p`` a `ContextFreeLoopIndex`.
    If p came from other_axis[::2].iter(), then it has *2* possible
    target paths and expressions: over the indexed or unindexed trees.
    Therefore when we index axis with p we must account for this, hence all
    indexing operations return a tuple of possible, equivalent, targets.

    Then, when we combine it all together, if we imagine having 2 loop indices
    like this, then we need the *product* of them to enumerate all possible
    targets.

    """
    raise TypeError(f"No handler provided for {type(index)}")


@_index_axes_per_index.register(LoopIndex)
def _(loop_index: LoopIndex, /, *args, **kwargs):
    """
    This function should return {None: [(path0, expr0), (path1, expr1)]}
    where path0 and path1 are "equivalent"
    This entails in inversion of loop_index.iterset.targets which has the form
    [
      {key: (path0, expr0), ...},
      {key: (path1, expr1), ...}
    ]
    """
    from pyop3.expr import LoopIndexVar
    from pyop3.expr.visitors import replace_terminals

    iterset = loop_index.iterset
    assert iterset.is_linear

    # Example:
    # If we assume that the loop index has target expressions
    #     AxisVar("a") * 2     and       AxisVar("b")
    # then this will return
    #     LoopIndexVar(p, "a") * 2      and LoopIndexVar(p, "b")
    # new_targets: dict[ConcretePathT, list[list[AxisTarget]]] = {idict(): []}
    replace_map = {
        axis.label: LoopIndexVar(loop_index, axis.localize())
        for axis, _ in iterset.visited_nodes(iterset.leaf_path)
    }

    iterset_targets = utils.just_one(collect_leaf_targets(iterset))
    new_targets = utils.freeze({
        idict(): [
            [
                AxisTarget(
                    axis_target.axis,
                    axis_target.component,
                    replace_terminals(axis_target.expr, replace_map),
                )
                for axis_target in axis_targets
            ]
            for axis_targets in iterset_targets
        ]
    })

    return (UNIT_AXIS_TREE, new_targets)


@_index_axes_per_index.register(ScalarIndex)
def _(index: ScalarIndex, /, target_axes, **kwargs):
    targets = utils.freeze({
        idict(): [[
            AxisTarget(index.axis, index.component, index.value),
        ]]
    })
    return (UNIT_AXIS_TREE, targets)


@_index_axes_per_index.register(Slice)
def _(slice_: Slice, /, target_axes, *, seen_target_exprs):
    from pyop3.expr import AxisVar
    from pyop3.expr.visitors import replace_terminals, collect_axis_vars
    from pyop3.expr import CompositeDat
    from pyop3.expr.visitors import get_shape, get_loop_axes, materialize_composite_dat


    # If we are just taking a component from a multi-component array,
    # e.g. mesh.points["cells"], then relabelling the axes just leads to
    # needless confusion. For instance if we had
    #
    #     myslice0 = Slice("mesh", AffineSliceComponent("cells", step=2))
    #
    # then mesh.points[myslice0] would work but mesh.points["cells"][myslice0]
    # would fail.
    # As a counter example, if we have non-trivial subsets then this sort of
    # relabelling is essential for things to make sense. If we have two subsets:
    #
    #     subset0 = Slice("mesh", Subset("cells", [1, 2, 3]))
    #
    # and
    #
    #     subset1 = Slice("mesh", Subset("cells", [4, 5, 6]))
    #
    # then mesh.points[subset0][subset1] is confusing, should subset1 be
    # assumed to work on the already sliced axis? This can be a major source of
    # confusion for things like interior facets in Firedrake where the first slice
    # happens in one function and the other happens elsewhere. We hit situations like
    #
    #     mesh.interior_facets[interior_facets_I_want]
    #
    # conflicts with
    #
    #     mesh.interior_facets[facets_I_want]
    #
    # where one subset is given with facet numbering and the other with interior
    # facet numbering. The labels are the same so identifying this is really difficult.
    #
    # We fix this here by requiring that non-full slices perform a relabelling and
    # full slices do not. A full slice is defined to be a slice where all of the
    # components are affine with start 0, stop None and step 1. The components must
    # also not already have a label since that would take precedence.
    #
    # TODO: Just have a special type for this!
    is_full = all(
        isinstance(s, AffineSliceComponent) and s.is_full and s.label_was_none
        for s in slice_.components
    )
    # NOTE: We should be able to eagerly return here?

    if is_full:
        axis_label = slice_.axis
    else:
        axis_label = slice_.label

    components = []
    for slice_component in slice_.components:
        targets = target_axes[idict({slice_.label: slice_component.label})]
        target_axis, target_component_label = just_one(targets.items())
        target_component = just_one(
            c for c in target_axis.components if c.label == target_component_label
        )

        # Loop over component regions and compute their sizes one by one.
        #
        # If the indexing operation is unordered then the assumption of
        # contiguous numbering is broken and so the existing regions must be discarded.
        # For example, if we have the two regions:
        #
        #     {"owned": 3, "ghost": 2}
        #
        # and permute them with the array [3, 4, 0, 2, 1], then it is no longer the
        # case that "owned" points preceded "ghost" points and so extracting the
        # "owned" region is no longer a trivial slice. We therefore choose to discard
        # this information.

        # TODO: Might be clearer to combine these steps
        regions = _prepare_regions_for_slice_component(slice_component, target_component.regions)
        indexed_regions = _index_regions(slice_component, regions, parent_exprs=seen_target_exprs)

        # if isinstance(slice_component, RegionSliceComponent):
        #     breakpoint()

        if isinstance(target_component.sf, StarForest):
            # It is not possible to have a star forest attached to a
            # component with variable extent
            assert isinstance(target_component.local_size, numbers.Integral)

            if isinstance(slice_component, RegionSliceComponent):
                if slice_component.region not in {"owned", "ghost"}:
                    raise NotImplementedError("Need to have a special type for selecting owned/ghost, what follows assumes that we are only getting 'owned' or 'ghost'")

                region_index = target_component.region_labels.index(slice_component.region)
                steps = utils.steps([r.local_size for r in target_component.regions], drop_last=False)
                start, stop = steps[region_index:region_index+2]
                indices = np.arange(start, stop, dtype=IntType)
                sf = None
            else:
                if isinstance(slice_component, AffineSliceComponent):
                    indices = np.arange(*slice_component.with_size(target_component.local_size), dtype=IntType)
                else:
                    assert isinstance(slice_component, SubsetSliceComponent)
                    # evaluate the subset to get the correct indices
                    subset_axes = utils.just_one(get_shape(slice_component.array))
                    subset_loop_axes = get_loop_axes(slice_component.array)
                    if subset_loop_axes:
                        raise NotImplementedError
                    subset_expr = CompositeDat(subset_axes, {subset_axes.leaf_path: slice_component.array})
                    indices = materialize_composite_dat(subset_expr).buffer.buffer.data_ro

                if isinstance(target_component.sf, StarForest):
                    # the issue is here when we are dealing with subsets (as opposed to region slices)
                    # I have just implemented a new attempt that uses another bit of the PETSc API
                    petsc_sf = filter_petsc_sf(target_component.sf.sf, indices, 0, target_component.local_size)
                    sf = StarForest(petsc_sf, target_component.sf.comm)
                else:
                    assert isinstance(target_component.sf, NullStarForest)
                    sf = NullStarForest(indices.size)
        else:
            sf = None

        if is_full:
            component_label = slice_component.component
        else:
            # TODO: Ideally the default labels here would be integers if not
            # somehow provided. Perhaps the issue stems from the fact that the label
            # attribute is used for two things: identifying paths in the index tree
            # and labelling the resultant axis component.
            component_label = slice_component.label

        component = AxisComponent(indexed_regions, label=component_label, sf=sf)
        components.append(component)

    axis = Axis(components, label=axis_label)

    # now do target expressions
    targets = {}
    for slice_component, axis_component in zip(slice_.components, axis.components, strict=True):
        index_path = idict({slice_.label: slice_component.label})
        target_axis, target_component_label = utils.just_one(target_axes[index_path].items())
        target_component = just_one(
            c for c in target_axis.components if c.label == target_component_label
        )

        # NOTE: is the localize() really needed?
        linear_axis = axis.linearize(axis_component.label).localize()

        if isinstance(slice_component, RegionSliceComponent):
            if slice_component.region in {OWNED_REGION_LABEL, GHOST_REGION_LABEL}:
                region_index = target_component.region_labels.index(slice_component.region)
                steps = utils.steps([r.size for r in target_component.regions], drop_last=False)
            else:
                region_index = target_component.region_labels.index(slice_component.region)
                steps = utils.steps([r.size for r in target_component.regions], drop_last=False)
            slice_expr = AxisVar(linear_axis) + steps[region_index]
        elif isinstance(slice_component, AffineSliceComponent):
            slice_expr = AxisVar(linear_axis) * slice_component.step + slice_component.start
        else:
            assert isinstance(slice_component, Subset)
            # replace the index information in the subset buffer
            try:
                subset_axis_var = just_one(collect_axis_vars(slice_component.array.layout))
            except ValueError:
                subset_axis_var = just_one(av for av in collect_axis_vars(slice_component.array.layout) if av.axis_label == slice_.label)

            if subset_axis_var.axis.label != linear_axis.label:
                replace_map = {subset_axis_var.axis.label: AxisVar(linear_axis)}
                slice_expr = replace_terminals(slice_component.array, replace_map, assert_modified=True)
            else:
                # FIXME: this isn't nice, should the labels ever match here?
                # labels match, strict=True will cause replace to fail
                slice_expr = slice_component.array
        slice_expr = replace_terminals(slice_expr, seen_target_exprs)

        targets[idict({axis.label: axis_component.label})] = [[
            AxisTarget(slice_.axis, slice_component.component, slice_expr),
        ]]

    axes = axis.as_tree()
    targets = utils.freeze(targets)
    return (axes, targets)


@_index_axes_per_index.register(CalledMap)
def _(called_map: CalledMap, *args, **kwargs):
    return called_map.axes.materialize(), called_map.axes.targets


def _make_leaf_axis_from_called_map_new(map_, map_name, output_spec, input_paths_and_exprs):
    from pyop3 import Dat
    from pyop3.expr.visitors import replace_terminals
    from pyop3.expr.buffer import LinearDatBufferExpression

    components = []
    replace_map = merge_dicts(
        t.replace_map for t in input_paths_and_exprs
    )
    for map_output in output_spec:
        # NOTE: This should be done more eagerly.
        arity = map_output.arity
        if not isinstance(arity, numbers.Integral):
            assert isinstance(arity, LinearDatBufferExpression)
            # arity = arity[map_.index]
            arity = replace_terminals(map_output.arity, replace_map, assert_modified=True)
        component = AxisComponent(arity, label=map_output.label)
        components.append(component)
    axis = Axis(components, label=map_name)

    targets = {}
    for component, map_output in zip(components, output_spec, strict=True):
        if not isinstance(map_output, TabulatedMapComponent):
            raise NotImplementedError("Currently we assume only arrays here")

        target_axis = map_output.target_axis
        target_component = map_output.target_component
        expr = replace_terminals(map_output.array, replace_map, assert_modified=True)
        axis_target = AxisTarget(target_axis, target_component, expr)
        targets[idict({axis.label: component.label})] = ((axis_target,),)
    targets = idict(targets)

    return (axis, targets)


def index_axes(
    index_tree: Union[IndexTree, Ellipsis],
    loop_context: Mapping | None = None,
    orig_axes: AxisTree | AxisForest | None = None,
# ) -> AxisForest:
    ):
    """Build an axis tree from an index tree.

    Parameters
    ----------
    axes :
        An axis tree that is being indexed. This argument is not always needed
        if, say, we are constructing the iteration set for the expression
        ``map(p).index()``. If not provided then some indices (e.g. unbounded
        slices) will no longer work.

    Returns
    -------
    AxisTree :
        The new axis tree.

    plus target paths and target exprs

    """
    if orig_axes is None:
        raise NotImplementedError("TODO")

    if orig_axes is not None:
        assert isinstance(orig_axes, (AxisTree, IndexedAxisTree))

    if utils.is_ellipsis_type(index_tree):
        if orig_axes is not None:
            return orig_axes
        else:
            raise ValueError

    # Determine the target axes addressed by the index tree. Since the index
    # tree defines the shape of the resulting indexed axis tree, each index
    # must map to a unique initial axis.
    target_axes = match_target_paths_to_axis_tree(index_tree, orig_axes)

    # Unpack the target paths from
    # 
    #     {index1: [component1, component2], index2: [component3]}
    #
    # to
    # 
    #     ({index1: component1, index2: component3},
    #      {index1: component2, index2: component3})
    #
    # (where each 'component' is also a tuple of *equivalent targets*).
    # target_paths = expand_collection_of_iterables(target_paths_compressed)

    # Resolve the symbolic targets into actual axes of the original tree
    # axis_tree_targets = match_target_paths_to_axis_tree(index_tree, target_paths, orig_axes)
    # axis_tree_targets = []
    # for index_targets in target_paths:
    #     # Of the many combinations of targets addressable by the provided index tree
    #     # only one is expected to actually match the given axis tree.
    #     axis_tree_target = matching_target(index_targets, orig_axes)
    #     axis_tree_targets.append(axis_tree_target)

    # Re-compress the result so it is easier to use in subsequent tree
    # traversals. That is, convert something like
    # 
    #     ({index1: target1, index2: target3},
    #      {index1: target2, index2: target3})
    #
    # to
    # 
    #     {index1: [target1, target2], index2: [target3]}
    #
    # (where each 'component' is also a tuple of *equivalent targets*).

    # construct the new, indexed, axis tree
    indexed_axes, indexed_targets = make_indexed_axis_tree(index_tree, target_axes)

    indexed_targets = complete_axis_targets(indexed_targets)

    # If the original axis tree is unindexed then no composition is required.
    if orig_axes is None or isinstance(orig_axes, AxisTree):
        if indexed_axes is UNIT_AXIS_TREE:
            return UnitIndexedAxisTree(
                orig_axes,
                targets=indexed_targets,
            )
        else:
            return IndexedAxisTree(indexed_axes, orig_axes, targets=indexed_targets)

    if orig_axes is None:
        raise NotImplementedError("Need to think about this case")

    matching_target = match_target(indexed_axes, orig_axes, indexed_targets)
    fullmap = _index_info_targets_axes(indexed_axes, matching_target, orig_axes)
    composed_targets = compose_targets(orig_axes, orig_axes.targets, indexed_axes, matching_target, fullmap)

    # TODO: reorder so the if statement captures the composition and this line is only needed once
    if indexed_axes is UNIT_AXIS_TREE:
        retval = UnitIndexedAxisTree(
            orig_axes.unindexed,
            targets=composed_targets,
        )
    else:
        retval = IndexedAxisTree(
            indexed_axes.node_map,
            orig_axes.unindexed,
            targets=composed_targets,
        )
        # if "closure" in str(retval.subst_layouts()):
        #     breakpoint()

    # debugging
    retval._matching_target
    return retval


def collect_index_tree_target_paths(index_tree: IndexTree) -> idict:
    return collect_index_tree_target_paths_rec(index_tree, index=index_tree.root)


def collect_index_tree_target_paths_rec(index_tree: IndexTree, *, index: Index) -> idict[Index, Any]:
    # target_paths = {index: collect_index_target_paths(index)}
    # # TODO: index_tree.child?
    # for subindex in filter(None, index_tree.node_map[index.id]):
    #     target_paths |= collect_index_tree_target_paths_rec(index_tree, index=subindex)
    # return idict(target_paths)
    target_paths = {index: []}
    index_target_paths = collect_index_target_paths(index)
    # TODO: index_tree.child?
    for target_path, subindex in zip(index_target_paths, index_tree.node_map[index.id], strict=True):
        if subindex is None:
            target_paths[index].append((target_path, None))
        else:
            subtarget_paths = collect_index_tree_target_paths_rec(index_tree, index=subindex)
            target_paths[index].append((target_path, subtarget_paths))
    return idict(target_paths)


def make_indexed_axis_tree(index_tree: IndexTree, target_axes):
    return _make_indexed_axis_tree_rec(
        index_tree,
        target_axes,
        index_path=idict(),
        expr_replace_map=idict(),
    )


def _make_indexed_axis_tree_rec(index_tree: IndexTree, target_axes, *, index_path: ConcretePathT, expr_replace_map):
    index = index_tree.node_map[index_path]

    index_axis_tree, per_index_targets = _index_axes_per_index(
        index, target_axes,
        seen_target_exprs=expr_replace_map,
    )

    targets: dict[ConcretePathT, tuple[AxisTarget, ...]] \
        = utils.StrictlyUniqueDefaultDict(tuple, per_index_targets)

    axis_tree = index_axis_tree
    for leaf_path, index_component_label in zip(
        index_axis_tree.leaf_paths, index.component_labels, strict=True
    ):
        index_path_ = index_path | {index.label: index_component_label}
        subindex = index_tree.node_map[index_path_]
        if subindex is None:
            continue

        expr_replace_map_ = (
            expr_replace_map
            | merge_dicts(t.replace_map for ts in per_index_targets[leaf_path] for t in ts)
        )

        # trim current path from 'target_axes' so subtrees can understand things
        target_axes_ = {
            filter_path(orig_path, index_path_): target
            for orig_path, target in target_axes.items()
        }

        subaxis_tree, subtargets = _make_indexed_axis_tree_rec(
            index_tree,
            target_axes_,
            index_path=index_path_,
            expr_replace_map=expr_replace_map_,
        )

        leaf_axis_key = leaf_path
        axis_tree = axis_tree.add_subtree(leaf_axis_key, subaxis_tree)

        for subpath, subtargets in subtargets.items():
            if subpath == idict():
                # product needed
                new_targets = []
                for AAA in targets.pop(leaf_path):
                    for BBB in subtargets:
                        new_targets.append(AAA + BBB)
                targets[leaf_path] = new_targets
            else:
                targets[leaf_path | subpath] = subtargets
    targets = utils.freeze(targets)

    return (axis_tree, targets)


def compose_targets(orig_axes, orig_targets, indexed_axes, indexed_target, fullmap, *, axis_path=idict()):
    """

    Traverse ``indexed_axes``, picking up bits from indexed_target_paths and keep
    trying to address orig_axes.paths with it. If there is a hit then we take that
    bit of the original target path into the new location.

    We *do not* accumulate things as we go. The final result should be the map

    { (indexed_axis, component) -> ((target_path1 | target_path2, ...), (targetexpr1 | targetexpr2)), ... }

    Things are complicated by the fact that not all of the targets from indexed_target_paths
    will resolve. Imagine axisB[p] where p is from axisA[::2].iter(). p targets 2 things and
    only one will match with axisB. We need to check for this outside the function.

    ---

    """
    from pyop3.expr.visitors import replace_terminals

    assert not orig_axes.is_empty

    composed_target = utils.StrictlyUniqueDict()

    if not axis_path:
        # special handling for entries that are not tied to a specific axis
        initially_empty_axis_targets = []
        expr_replace_map = merge_dicts(t.replace_map for t in indexed_target[idict()])

        for axis_targets in orig_targets[idict()]:
            XXX = []
            for axis_target in axis_targets:
                composed_expr = replace_terminals(axis_target.expr, expr_replace_map)
                composed_axis_target = AxisTarget(axis_target.axis, axis_target.component, composed_expr)
                XXX.append(composed_axis_target)
            initially_empty_axis_targets.append(XXX)

        # then from the indexed axes
        YYY = [initially_empty_axis_targets]
        for target_path in fullmap[idict()]:
            ZZZ = []
            for orig_axis_targets in orig_targets[target_path]:
                AAA = []
                for orig_axis_target in orig_axis_targets:
                    composed_expr = replace_terminals(orig_axis_target.expr, expr_replace_map)
                    composed_axis_target = AxisTarget(
                        orig_axis_target.axis, orig_axis_target.component, composed_expr
                    )
                    AAA.append(composed_axis_target)
                ZZZ.append(AAA)
            YYY.append(ZZZ)

        merged = []
        for debug in itertools.product(*YYY):
            merged.append(sum(debug, start=[]))

        # else:
        #     composed_target[idict()] = ((),)
        composed_target[idict()] = utils.freeze(merged)

        if indexed_axes.is_empty or indexed_axes is UNIT_AXIS_TREE:
            return idict(composed_target)

    axis = indexed_axes.node_map[axis_path]
    for component in axis.components:
        path_ = axis_path | {axis.label: component.label}

        # TODO: use merge_targets, but also need to do a subst
        # merged = merge_targets()
        # some of these cannot be combined, and others can!
        AAA = []
        indexed_axis_targets = indexed_target[path_]
        expr_replace_map = merge_dicts(t.replace_map for t in indexed_axis_targets)
        for target_path in fullmap[path_]:
            BBB = []  # cannot be mixed
            for orig_axis_targets in orig_targets[target_path]:
                composed_axis_targets = []
                for orig_axis_target in orig_axis_targets:
                    composed_expr = replace_terminals(orig_axis_target.expr, expr_replace_map)
                    composed_axis_target = AxisTarget(
                        orig_axis_target.axis, orig_axis_target.component, composed_expr
                    )
                    composed_axis_targets.append(composed_axis_target)
                BBB.append(composed_axis_targets)
            AAA.append(BBB)

        # also used in leaf_target_paths, generalise
        merged = []
        for debug in itertools.product(*AAA):
            merged.append(utils.reduce("+", debug, []))

        composed_target[path_] = utils.freeze(merged)

        if indexed_axes.node_map[path_]:
            composed_target_paths_ = compose_targets(
                orig_axes,
                orig_targets,
                indexed_axes,
                indexed_target,
                fullmap,
                axis_path=path_,
            )
            for mykey, myvalue in composed_target_paths_.items():
                composed_target[path_ | mykey] = myvalue

    return idict(composed_target)


class MyBadError(Exception):
    pass


def _index_info_targets_axes(indexed_axes, target, orig_axes) -> bool:
    """Return whether the index information targets the original axis tree.

    This is useful for when multiple interpretations of axis information are
    provided (e.g. with loop indices) and we want to filter for the right one.

    ---

    UPDATE

    Look at the full target tree to resolve ambiguity in indexing things. For example
    consider a mixed space. A slice over the mesh is not clear as it may refer to the
    axis of either space. Here we construct the full path and pull out the axes that
    are actually desired.

    raises an exception if things don't match (which we expect to happen)

    """
    result = {}
    for indexed_leaf_path in indexed_axes.leaf_paths:
        # first get the actual axes that are visited
        axis_targets = []
        for indexed_leaf_path_acc in accumulate_path(indexed_leaf_path):
            axis_targets.extend(target[indexed_leaf_path_acc])
        leaf_target_path = merge_dicts(t.path for t in axis_targets)

        if leaf_target_path not in orig_axes.node_map:
            raise MyBadError(
                "This means that the leaf of an indexed axis tree doesn't target the original axes")

        # now construct the mapping to specific *full* axis paths, not path elements
        # we need to look at the node map to get the right ordering as target_path_acc
        # is in indexed order, not the order in the original tree
        ordered_target_path = utils.just_one(
            tp
            for tp in orig_axes.node_map.keys()
            if tp == leaf_target_path
        )
        partial_to_full_path_map = {}
        acc = idict()
        for ax, c in ordered_target_path.items():
            acc = acc | {ax: c}
            partial_to_full_path_map[ax, c] = acc

        for indexed_leaf_path_acc in accumulate_path(indexed_leaf_path):
            indexed_axis_targets = target[indexed_leaf_path_acc]
            target_path = merge_dicts(t.path for t in indexed_axis_targets)

            full_target_paths = []
            for target_axis, target_component in target_path.items():
                full_axis_targets_ = partial_to_full_path_map[target_axis, target_component]
                full_target_paths.append(full_axis_targets_)
            result[indexed_leaf_path_acc] = tuple(full_target_paths)
    return idict(result)


# TODO: just get rid of this, assuming the new system works
def expand_compressed_target_paths(compressed_target_paths):
    return expand_collection_of_iterables(compressed_target_paths)


@dataclasses.dataclass(frozen=True)
class IndexIteratorEntry:
    index: LoopIndex
    source_path: idict
    target_path: idict
    source_exprs: idict
    target_exprs: idict

    @property
    def loop_context(self):
        return idict({self.index.id: (self.source_path, self.target_path)})

    @property
    def replace_map(self):
        return idict(
            {self.index.id: merge_dicts([self.source_exprs, self.target_exprs])}
        )

    @property
    def target_replace_map(self):
        return idict(
            {
                self.index.id: {ax: expr for ax, expr in self.target_exprs.items()},
            }
        )

    @property
    def source_replace_map(self):
        return idict(
            {
                self.index.id: {ax: expr for ax, expr in self.source_exprs.items()},
            }
        )


class ArrayPointLabel(enum.IntEnum):
    CORE = 0
    ROOT = 1
    LEAF = 2


class IterationPointType(enum.IntEnum):
    CORE = 0
    ROOT = 1
    LEAF = 2


# TODO This should work for multiple loop indices. One should really pass a loop expression.
def partition_iterset(index: LoopIndex, arrays):
    """Split an iteration set into core, root and leaf index sets.

    The distinction between these is as follows:

    * CORE: May be iterated over without any communication at all.
    * ROOT: Requires a leaf-to-root reduction (i.e. up-to-date SF roots).
    * LEAF: Requires a root-to-leaf broadcast (i.e. up-to-date SF leaves) and also up-to-date roots.

    The partitioning algorithm basically loops over the iteration set and marks entities
    in turn. Any entries whose stencils touch an SF leaf are marked LEAF and any that do
    not touch leaves but do roots are marked ROOT. Any remaining entities do not require
    the SF and are marked CORE.

    """
    from pyop3 import Mat

    # take first
    # if index.iterset.depth > 1:
    #     raise NotImplementedError("Need a good way to sniff the parallel axis")
    paraxis = index.iterset.root

    # FIXME, need indices per component
    if len(paraxis.components) > 1:
        raise NotImplementedError

    # at a minimum this should be done per multi-axis instead of per array
    is_root_or_leaf_per_array = {}
    for array in arrays:
        # skip matrices
        # really nasty hack for now to handle indexed mats
        if isinstance(array, Mat) or not hasattr(array, "buffer"):
            continue

        # skip purely local arrays
        if not array.buffer.is_distributed:
            continue

        sf = array.buffer.sf  # the dof sf

        # mark leaves and roots
        is_root_or_leaf = np.full(sf.size, ArrayPointLabel.CORE, dtype=np.uint8)
        is_root_or_leaf[sf.iroot] = ArrayPointLabel.ROOT
        is_root_or_leaf[sf.ileaf] = ArrayPointLabel.LEAF

        is_root_or_leaf_per_array[array.name] = is_root_or_leaf

    labels = np.full(paraxis.size, IterationPointType.CORE, dtype=np.uint8)
    # for p in index.iterset.iter():
    #     # hack because I wrote bad code and mix up loop indices and itersets
    #     p = dataclasses.replace(p, index=index)
    # for p in index.iter():
    #     parindex = p.source_exprs[paraxis.label]
    #     assert isinstance(parindex, numbers.Integral)
    #
    #     for array in arrays:
    #         # same nasty hack
    #         if isinstance(array, (Mat, Sparsity)) or not hasattr(array, "buffer"):
    #             continue
    #         # skip purely local arrays
    #         if not array.buffer.is_distributed:
    #             continue
    #         if labels[parindex] == IterationPointType.LEAF:
    #             continue
    #
    #         # loop over stencil
    #         array = array.with_context({index.id: (p.source_path, p.target_path)})
    #
    #         for q in array.axes.iter({p}):
    #             # offset = array.axes.offset(q.target_exprs, q.target_path)
    #             offset = array.axes.offset(q.source_exprs, q.source_path, loop_exprs=p.replace_map)
    #
    #             point_label = is_root_or_leaf_per_array[array.name][offset]
    #             if point_label == ArrayPointLabel.LEAF:
    #                 labels[parindex] = IterationPointType.LEAF
    #                 break  # no point doing more analysis
    #             elif point_label == ArrayPointLabel.ROOT:
    #                 assert labels[parindex] != IterationPointType.LEAF
    #                 labels[parindex] = IterationPointType.ROOT
    #             else:
    #                 assert point_label == ArrayPointLabel.CORE
    #                 pass

    parcpt = just_one(paraxis.components)  # for now

    # I don't think this is working - instead everything touches a leaf
    # core = just_one(np.nonzero(labels == IterationPointType.CORE))
    # root = just_one(np.nonzero(labels == IterationPointType.ROOT))
    # leaf = just_one(np.nonzero(labels == IterationPointType.LEAF))
    # core = np.asarray([], dtype=IntType)
    # root = np.asarray([], dtype=IntType)
    # leaf = np.arange(paraxis.size, dtype=IntType)

    # hack to check things
    core = np.asarray([0], dtype=IntType)
    root = np.asarray([1], dtype=IntType)
    leaf = np.arange(2, paraxis.size, dtype=IntType)

    subsets = []
    for data in [core, root, leaf]:
        # Constant? no, rank_equal=False
        # Parameter?
        size = Dat(
            AxisTree(Axis(1)), data=np.asarray([len(data)]), dtype=IntType
        )
        subset = Dat(
            Axis([AxisComponent(size, parcpt.label)], paraxis.label), data=data
        )
        subsets.append(subset)
    subsets = tuple(subsets)
    return "not used", subsets

    # make a new iteration set over just these indices
    # index with just core (arbitrary)

    # need to use the existing labels here
    mysubset = Slice(
        paraxis.label,
        [Subset(parcpt.label, subsets[0], label=parcpt.label)],
        label=paraxis.label,
    )
    new_iterset = index.iterset[mysubset]

    return index.copy(iterset=new_iterset), subsets


@functools.singledispatch
def _prepare_regions_for_slice_component(slice_component, regions) -> tuple[AxisComponentRegion, ...]:
    raise TypeError


@_prepare_regions_for_slice_component.register(RegionSliceComponent)
def _(region_component: RegionSliceComponent, regions):
    return tuple(regions)


@_prepare_regions_for_slice_component.register(AffineSliceComponent)
def _(affine_component: AffineSliceComponent, regions):
    assert affine_component.step != 0
    return tuple(regions) if affine_component.step > 0 else tuple(reversed(regions))


@_prepare_regions_for_slice_component.register(Subset)
def _(subset: Subset, regions) -> tuple:
    # We must lose all region information if we are not accessing entries in order
    if len(regions) > 1 and not subset.array.buffer.buffer.ordered:
        size = sum(r.size for r in regions)
        return (AxisComponentRegion(size),)
    else:
        return regions


@functools.singledispatch
def _index_regions(*args, **kwargs) -> tuple[AxisComponentRegion, ...]:
    raise TypeError


@_index_regions.register(RegionSliceComponent)
def _(region_component: RegionSliceComponent, regions, *, parent_exprs) -> tuple[AxisComponentRegion, ...]:
    from pyop3.expr.visitors import replace_terminals as expr_replace

    selected_region = utils.just_one(
        region
        for region in regions
        if region.label == region_component.region
    )

    # Substitute any parent expressions into the region size. This is necessary
    # for region slices of trees that are both multi-region and ragged. For
    # instance, consider the axis tree:
    #
    #   { mesh: (owned: 3, ghost: 2) }
    #     { dofs: (unconstrained: [1, 1, 0, 1, 0], unconstrained: [0, 0, 1, 0, 1]) }
    #
    # If we wish to take only the ghost points, then the ragged arrays for
    # the dof axis need to be truncated.
    size = expr_replace(selected_region.size, parent_exprs)
    selected_region = selected_region.__record_init__(label=None, size=size)
    return (selected_region,)


@_index_regions.register(AffineSliceComponent)
def _(affine_component: AffineSliceComponent, regions, *, parent_exprs) -> tuple[AxisComponentRegion, ...]:
    """
    Examples
    --------
    {"a": 3, "b": 2}[::]   -> {"a": 3, "b": 2} ( [0, 1, 2, 3, 4] )
    {"a": 3, "b": 2}[::2]  -> {"a": 2, "b": 1} ( [0, 2, 4] )
    {"a": 3, "b": 2}[1::]  -> {"a": 2, "b": 2} ( [1, 2, 3, 4] )
    {"a": 3, "b": 2}[1::2] -> {"a": 1, "b": 1} ( [1, 3] )
    {"a": 3, "b": 2}[:3:]  -> {"a": 3, "b": 0} ( [0, 1, 2] )
    {"a": 3, "b": 2}[:4:2] -> {"a": 2, "b": 0} ( [0, 2] )

    """
    from pyop3.expr import conditional
    from pyop3.expr.visitors import replace_terminals as expr_replace, min_

    if affine_component.is_full:
        indexed_regions = []
        for region in regions:
            size = expr_replace(region.size, parent_exprs)
            indexed_region = region.__record_init__(size=size)
            indexed_regions.append(indexed_region)
        return tuple(indexed_regions)

    size = sum(r.size for r in regions)
    start, stop, step = affine_component.with_size(size)

    # utils.debug_assert(lambda: min_value(start) >= 0)

    # TODO: This check doesn't always hold. For example if we have the arities of
    # facets and are expecting interior facets but there aren't any. Then the max
    # value here is 1 not 2. We could avoid this by letting buffers define, instead
    # of computing, a max_value.
    # utils.debug_assert(lambda: max_value(stop) <= max_value(size))

    # For single region components we can simplify things because we know that
    # the slice is always in bounds for the region.
    if len(regions) == 1:
        region = utils.just_one(regions)
        region_size = utils.ceildiv((stop - start), step)
        region_size = expr_replace(region_size, parent_exprs)
        indexed_region = AxisComponentRegion(region_size, region.label)
        return (indexed_region,)

    indexed_regions = []
    loc = 0
    offset = start
    for region in regions:
        lower_bound = loc
        upper_bound = loc + region.size
        # This really requires more exposition but the basic idea
        # is we need to stride over the regions in turn and collect the
        # relevant pieces of each one. In particular we need to know the
        # size of the new, indexed region, and where we need to start
        # from when we look at the next region (the 'offset').
        #
        # The code below is equivalent to the following but adapted to work for
        # ragged things.
        #
        #     # out-of-bounds, just move forwards
        #     if upper_bound < start or lower_bound >= stop:
        #         region_size = 0
        #         offset -= region.size
        #     else:
        #         region_size = ceildiv((min(region.size, stop-loc) - offset), step)
        #         offset = (offset + region.size) % step
        if start == stop:
            out_of_bounds = True
        else:
            out_of_bounds = (upper_bound < start) | (lower_bound >= stop)
        region_size = conditional(out_of_bounds, 0, utils.ceildiv((min_(region.size, stop-loc) - offset), step))
        offset = conditional(out_of_bounds, offset-region.size, (offset+region.size) % step)

        # Make sure that we apply any parent indexing to the size expression
        # (important if we are dealing with ragged things).
        region_size_debug = region_size
        region_size = expr_replace(region_size, parent_exprs)

        indexed_region = AxisComponentRegion(region_size, region.label)
        indexed_regions.append(indexed_region)
        loc += region.size
    return tuple(indexed_regions)


@_index_regions.register(SubsetSliceComponent)
def _(subset: SubsetSliceComponent, regions, **kwargs) -> tuple:
    """
    IMPORTANT: This function will do a full search of the set of indices.

    Examples
    --------
    {"a": 3, "b": 2}[0,1,2,3,4] -> {"a": 3, "b": 2}
    {"a": 3, "b": 2}[0,1,2]     -> {"a": 3, "b": 0}
    {"a": 3, "b": 2}[1,4]       -> {"a": 1, "b": 1}
    {"a": 3, "b": 2}[3,4]       -> {"a": 0, "b": 2}

    """
    from pyop3 import Scalar

    indices = subset.array.buffer.buffer.data_ro

    indexed_regions = []
    loc = 0
    lower_index = 0
    for region in regions:
        upper_index = np.searchsorted(indices, loc+region.local_size)
        size = upper_index - lower_index

        if isinstance(region.size, numbers.Integral):
            size_ = size
        else:
            size_ = Scalar(size, constant=True)
        indexed_region = AxisComponentRegion(size_, region.label)
        indexed_regions.append(indexed_region)

        loc += region.local_size
        lower_index = upper_index
    return tuple(indexed_regions)


def convert_region_to_affine_slice(region_slice: RegionSliceComponent, axis_component: AxisComponent) -> AffineSliceComponent:
    region_index = axis_component.region_labels.index(region_slice.label)
    region_sizes = utils.steps(region.size for region in axis_component.regions)
    return AffineSliceComponent(start=region_sizes[region_index], stop=region_sizes[region_index+1])


def as_slice(label: ComponentLabelT) -> UnparsedSlice:
    return UnparsedSlice(label)


def collect_leaf_targets(axes):
    """
    Returns
    -------
    An iterable of generators, one per leaf.

    Notes
    -----
    This function is a generator because often the result does not need to be
    exhaustively searched.

    """
    return tuple(
        _collect_leaf_targets_per_leaf(axes, leaf_path, None, utils.UniqueList())
        for leaf_path in axes.leaf_paths
    )


def _collect_leaf_targets_per_leaf(axes, leaf_path, path, targets):
    if path is None:
        path_ = idict()
    else:
        axis = axes.node_map[path]
        path_ = path | {axis.label: leaf_path[axis.label]}

    for axis_targets in axes.targets[path_]:
        with utils.stack(targets, axis_targets):
            if axes.node_map[path_]:
                yield from _collect_leaf_targets_per_leaf(axes, leaf_path, path_, targets)
            else:
                yield tuple(targets)


def collect_leaf_target_paths(axes):
    return tuple(
        _collect_leaf_target_paths_per_leaf(axes, leaf_path)
        for leaf_path in axes.leaf_paths
    )


def _collect_leaf_target_paths_per_leaf(axes, leaf_path):
    leaf_targets = _collect_leaf_targets_per_leaf(axes, leaf_path, None, utils.UniqueList())
    for leaf_target in leaf_targets:
        yield merge_dicts(t.path for t in leaf_target)
