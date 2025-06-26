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
from typing import Any, Collection, Hashable, Mapping, Sequence, Type, cast, Optional

import numpy as np
import pymbolic as pym
from pyop3.tensor.dat import ArrayBufferExpression, as_linear_buffer_expression
from pyop3.exceptions import Pyop3Exception
import pytools
from immutabledict import immutabledict

from pyop3.tensor import Dat
from pyop3.axtree import (
    Axis,
    AxisComponent,
    AxisTree,
    AxisForest,
    AxisVar,
    LoopIterable,
)
from pyop3.axtree.layout import _as_int
from pyop3.axtree.tree import (
    UNIT_AXIS_TREE,
    AbstractAxisTree,
    ContextSensitiveLoopIterable,
    AxisComponentRegion,
    IndexedAxisTree,
    UnitIndexedAxisTree,
    LoopIndexVar,
    OWNED_REGION_LABEL,
    GHOST_REGION_LABEL,
)
from pyop3.dtypes import IntType
# from pyop3.expr_visitors import replace_terminals, replace as expr_replace
from pyop3.lang import KernelArgument
from pyop3.sf import StarForest
from pyop3.tree import (
    ConcretePathT,
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

class InvalidIndexTargetException(Pyop3Exception):
    """Exception raised when we try to match index information to a mismatching axis tree."""


class Index(MultiComponentLabelledNode):
    pass


# NOTE: index trees are not really labelled trees. The component labels are always
# nonsense. Instead I think they should just advertise a degree and then attach
# to matching index (instead of label).
class IndexTree(MutableLabelledTreeMixin, LabelledTree):

    # {{{ interface impls

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
    fields = LabelledNodeComponent.fields | {"component"}

    def __init__(self, component, *, label=None):
        super().__init__(label)
        self.component = component

    @property
    @abc.abstractmethod
    def is_full(self) -> bool:
        pass


class AffineSliceComponent(SliceComponent):
    fields = SliceComponent.fields | {"start", "stop", "step", "label_was_none"}

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

        super().__init__(component, label=label, **kwargs)
        # TODO: make None here and parse with `with_size()`
        self.start = start if start is not None else 0
        self.stop = stop
        # could be None here
        self.step = step if step is not None else 1

        # hack to force a relabelling
        self.label_was_none = label_was_none

    @property
    def datamap(self) -> immutabledict:
        return immutabledict()

    @property
    def is_full(self) -> bool:
        return self.start == 0 and self.stop is None and self.step == 1

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


class SubsetSliceComponent(SliceComponent):
    fields = SliceComponent.fields | {"array"}

    def __init__(self, component, array, **kwargs):
        array = as_linear_buffer_expression(array)

        self.array = array
        super().__init__(component, **kwargs)

    @property
    def is_full(self) -> bool:
        return False


# alternative name, better or worse? I think worse
Subset = SubsetSliceComponent


class RegionSliceComponent(SliceComponent):
    """A slice component that takes all entries from a particular region.

    This class differs from an affine slice in that it 'consumes' the region
    label, and so breaks any recursive cycle where one might have something
    like `axes.owned.buffer_slice` (which accesses `axes.owned.buffer_slice`...).

    """
    def __init__(self, component, region: str, *, label=None) -> None:
        super().__init__(component, label=label)
        self.region = region

    @property
    def is_full(self) -> bool:
        return False


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
    def target_path(self) -> immutabledict:
        return immutabledict({self.target_axis: self.target_component})


# TODO: Implement AffineMapComponent
class TabulatedMapComponent(MapComponent):
    fields = MapComponent.fields | {"array", "arity"}

    def __init__(self, target_axis, target_component, array, *, arity=None, label=None):
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


# class ContextFreeIndex(Index, ContextFree, abc.ABC):
#     # The following is unimplemented but may prove useful
#     # @property
#     # def axes(self):
#     #     return self._tree.axes
#     #
#     # @property
#     # def target_paths(self):
#     #     return self._tree.target_paths
#     #
#     # @cached_property
#     # def _tree(self):
#     #     """
#     #
#     #     Notes
#     #     -----
#     #     This method will deliberately not work for slices since slices
#     #     require additional existing axis information in order to be valid.
#     #
#     #     """
#     #     return as_index_tree(self)
#     @abc.abstractmethod
#     def restrict(self, paths):
#         """Return a restricted index with only the paths provided.
#
#         The resulting index will have its components ordered as given by ``paths``.
#
#         """
#
#
#
# class ContextSensitiveIndex(Index, ContextSensitive, abc.ABC):
#     def __init__(self, context_map, *, id=None):
#         Index.__init__(self, id)
#         ContextSensitive.__init__(self, context_map)



class LoopIndex(Index, KernelArgument):
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

    # TODO: Prefer this as a traversal
    # TODO: Include iterset outer loops
    @property
    def outer_loops(self):
        return (self,)

    @property
    def is_context_free(self):
        return len(self.iterset.leaf_paths) == 1

    @cached_property
    def axes(self) -> IndexedAxisTree:
        from pyop3.expr_visitors import replace_terminals

        if not self.is_context_free:
            raise ContextSensitiveException("Expected a context-free index")

        # NOTE: same as _index_axes_index

        # Example:
        # If we assume that the loop index has target expressions
        #     AxisVar("a") * 2     and       AxisVar("b")
        # then this will return
        #     LoopIndexVar(p, "a") * 2      and LoopIndexVar(p, "b")
        replace_map = {
            immutabledict(): (
                self.iterset.leaf_path,
                {axis.label: LoopIndexVar(self.id, axis) for axis, _ in self.iterset.visited_nodes(self.iterset.leaf_path)},
            )
        }
        replace_map = replace_map[immutabledict()][1]

        targets = []
        for equivalent_targets in self.iterset.paths_and_exprs:
            new_path = {}
            new_exprs = {}
            for (orig_path, orig_exprs) in equivalent_targets.values():
                new_path.update(orig_path)
                for axis_label, orig_expr in orig_exprs.items():
                    new_exprs[axis_label] = replace_terminals(orig_expr, replace_map)
            new_path = immutabledict(new_path)
            new_exprs = immutabledict(new_exprs)
            targets.append(immutabledict({immutabledict(): (new_path, new_exprs)}))
        return UnitIndexedAxisTree(unindexed=None, targets=targets)

    # TODO: don't think this is useful any more, certainly a confusing name
    @property
    def leaf_target_paths(self):
        """

        Unlike with maps and slices, loop indices are single-component (so return a 1-tuple)
        but that component can target differently labelled axes (so the tuple entry is an n-tuple).

        """
        assert self.is_context_free

        equivalent_paths = []
        for targets_acc in self.iterset.targets_acc:
            equivalent_path, _ = targets_acc[self.iterset.leaf_path]
            equivalent_paths.append(equivalent_path)
        equivalent_paths = tuple(equivalent_paths)

        return (equivalent_paths,)


    # @cached_property
    # def local_index(self):
    #     return LocalLoopIndex(self)

    # @property
    # def i(self):
    #     return self.local_index

    # @property
    # def paths(self):
    #     return tuple(self.iterset.path(*leaf) for leaf in self.iterset.leaves)
    #
    # NOTE: This is confusing terminology. A loop index can be context-sensitive
    # in two senses:
    # 1. axes.index() is context-sensitive if axes is multi-component
    # 2. axes[p].index() is context-sensitive if p is context-sensitive
    # I think this can be resolved by considering axes[p] and axes as "iterset"
    # and handling that separately.
    def with_context(self, context, *args) -> LoopIndex:
        from pyop3.itree.parse import _as_context_free_indices
        return utils.just_one(_as_context_free_indices(self, context))

    # unsure if this is required
    @property
    def datamap(self):
        return self.iterset.datamap


class InvalidIterationSetException(Pyop3Exception):
    pass


class ScalarIndex(Index):
    fields = {"axis", "component", "value", "id"}

    def __init__(self, axis, component, value):
        super().__init__()
        self.axis = axis
        self.component = component
        self.value = value

    @property
    def leaf_target_paths(self):
        return ((immutabledict({self.axis: self.component}),),)

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
            components = tuple(c.copy(label=i) for i, c in enumerate(components))

        self.axis = axis
        self.components = components
        super().__init__(label=label)

    @property
    @utils.deprecated("components")
    def slices(self):
        return self.components

    @property
    def component_labels(self) -> tuple:
        return tuple(s.label for s in self.slices)

    @cached_property
    def leaf_target_paths(self):
        # We return a collection of 1-tuples because each slice component
        # targets only a single (axis, component) pair. There are no
        # 'equivalent' target paths.
        return tuple(
            (immutabledict({self.axis: subslice.component}),)
            for subslice in self.slices
        )

    @property
    def expanded(self) -> tuple:
        return (self,)

    def restrict(self, paths):
        new_slice_components = []
        for path in paths:
            found = False
            for slice_component in self.slices:
                if immutabledict({self.label: slice_component.label}) == path:
                    new_slice_components.append(slice_component)
                    found = True
            if not found:
                raise ValueError("Invalid path provided")

        return type(self)(self.axis, new_slice_components, label=self.label)

    @property
    def datamap(self):
        return merge_dicts([s.datamap for s in self.slices])


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
        self.connectivity = immutabledict(connectivity)

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
        return immutabledict(data)


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
        from pyop3.itree.parse import as_index_forests

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

    @property
    def outer_loops(self):
        return self.index.outer_loops

    @cached_property
    def axes(self) -> IndexedAxisTree:
        if not self.is_context_free:
            raise ContextSensitiveException("Expected a context-free index")

        input_axes = self.index.axes
        axes_ = input_axes.materialize()
        targets = {}
        for input_leaf_path in input_axes.leaf_paths:
            equivalent_inputs = input_axes.targets

            found = False
            for input_targets in equivalent_inputs:

                # traverse the input axes and collect the final target path
                if len(input_targets) == 1:
                    input_path, input_exprs = input_targets[immutabledict()]
                else:
                    raise NotImplementedError

                if input_path in self.connectivity:
                    if found:
                        raise UnspecialisedCalledMapException(
                            "Multiple (equivalent) input paths have been found that "
                            "are accepted by the map. This ambiguity makes it impossible "
                            "to form an IndexTree."
                        )

                    if len(self.connectivity[input_path]) > 1:
                        raise UnspecialisedCalledMapException(
                            "Multiple (equivalent) output paths are generated by the map. "
                            "This ambiguity makes it impossible to form an IndexTree."
                        )

                    found = True
                    output_spec = just_one(self.connectivity[input_path])

                    # make a method
                    subaxis, subtargets = _make_leaf_axis_from_called_map_new(
                        self, self.name, output_spec, input_targets,
                    )

                    axes_ = axes_.add_axis(input_leaf_path, subaxis)
                    targets.update(subtargets)

            if not found:
                assert False

        targets = immutabledict(targets)

        # Since maps are necessarily restricted to a single interpretation we only
        # have one possible interpretation of the targets.
        targets = (targets,)
        return IndexedAxisTree(axes_.node_map, None, targets=targets)

    @property
    def is_context_free(self) -> bool:
        return self.index.is_context_free

    # NOTE: nothing about this is specific to an index
    @property
    def leaf_target_paths(self) -> tuple:
        targets_acc = just_one(self.axes.targets_acc)

        paths = []
        for leaf_path in self.axes.leaf_paths:
            equivalent_paths = (targets_acc[leaf_path][0],)
            paths.append(equivalent_paths)
        return tuple(paths)

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


    @cached_property
    def _source_paths(self):
        return tuple(p for p, _ in self.connectivity)

    @property
    def _connectivity_dict(self):
        return immutabledict(self.connectivity)

    # TODO cleanup
    def with_context(self, context, axes=None):
        raise NotImplementedError
        # maybe this line isn't needed?
        # cf_index = self.from_index.with_context(context, axes)
        cf_index = self.from_index
        leaf_target_paths = tuple(
            immutabledict({mcpt.target_axis: mcpt.target_component})
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
def collect_index_target_paths(index: Index) -> tuple[tuple[immutabledict[str, str], ...], ...]:
    raise TypeError(f"No handler defined for {type(index).__name__}")


@collect_index_target_paths.register(LoopIndex)
def _(loop_index: LoopIndex) -> tuple[tuple[immutabledict[str, str], ...], ...]:
    def accumulate_target_path(iterset_target):
        return merge_dicts(path for (path, _) in iterset_target.values())

    # TODO: It would be nice to have a better attribute through which to
    # collect this
    return (
        tuple(
            accumulate_target_path(iterset_target)
            for iterset_target in loop_index.iterset.paths_and_exprs
        ),
    )


@collect_index_target_paths.register(ScalarIndex)
def _(scalar_index: ScalarIndex, /, *args, **kwargs):
    return scalar_index.leaf_target_paths


@collect_index_target_paths.register(Slice)
def _(slice_: Slice) -> tuple[tuple[immutabledict[str, str]], ...]:
    return tuple(
        (immutabledict({slice_.axis: slice_component.component}),)
        for slice_component in slice_.slices
    )


@collect_index_target_paths.register(CalledMap)
def _(called_map: CalledMap) -> tuple[tuple[immutabledict[str, str]], ...]:
    compressed_targets = []
    for leaf_path in called_map.axes.leaf_paths:
        compressed_targets.append(tuple(t[leaf_path][0] for t in called_map.axes.targets))
    return tuple(compressed_targets)


def match_target_paths_to_axis_tree(index_tree, orig_axes):
    target_axes_by_index, leaf_target_axes = match_target_paths_to_axis_tree_rec(index_tree, orig_axes, index_path=immutabledict(), candidate_target_paths_acc=(immutabledict(),))
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

        index_path_ = index_path | {index.label: index_component_label}

        candidate_target_paths_acc_ = tuple(
            candidate_path | index_target_path
            for candidate_path in candidate_target_paths_acc 
            for index_target_path in equivalent_index_target_paths
        )
        if index_tree.node_map[index_path_] is None:
            # At a leaf, can now determine the axes that are referenced by the path.
            # We only expect a single match from all the collected candidate paths.
            full_target_axes = just_one(
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
            immutabledict({
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

    target_axes_by_index = immutabledict(target_axes_by_index)
    leaf_target_axes = tuple(leaf_target_axes)

    return target_axes_by_index, leaf_target_axes


@functools.singledispatch
def _index_axes_index(index: Index, /, *args, **kwargs) -> tuple[AxisTree, tuple, tuple[LoopIndex, ...]]:
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


@_index_axes_index.register(LoopIndex)
def _(loop_index: LoopIndex, /, target_axes, **kwargs):
    """
    This function should return {None: [(path0, expr0), (path1, expr1)]}
    where path0 and path1 are "equivalent"
    This entails in inversion of loop_index.iterset.targets which has the form
    [
      {key: (path0, expr0), ...},
      {key: (path1, expr1), ...}
    ]
    """
    from pyop3.expr_visitors import replace_terminals

    axes = UNIT_AXIS_TREE

    # selected_axes = target_axes[loop_index]

    # Example:
    # If we assume that the loop index has target expressions
    #     AxisVar("a") * 2     and       AxisVar("b")
    # then this will return
    #     LoopIndexVar(p, "a") * 2      and LoopIndexVar(p, "b")
    replace_map = {
        axis.label: LoopIndexVar(loop_index.id, axis)
        for axis, _ in loop_index.iterset.visited_nodes(loop_index.iterset.leaf_path)
    }
    targets = {immutabledict(): []}
    for equivalent_targets in loop_index.iterset.paths_and_exprs:
        new_path = {}
        new_exprs = {}
        for (orig_path, orig_exprs) in equivalent_targets.values():
            new_path.update(orig_path)
            for axis_label, orig_expr in orig_exprs.items():
                new_exprs[axis_label] = replace_terminals(orig_expr, replace_map)
        new_path = immutabledict(new_path)
        new_exprs = immutabledict(new_exprs)
        targets[immutabledict()].append((new_path, new_exprs))
    targets[immutabledict()] = tuple(targets[immutabledict()])
    targets = immutabledict(targets)

    outer_loops = loop_index.iterset.outer_loops + (loop_index,)

    return (axes, targets, outer_loops)


@_index_axes_index.register(ScalarIndex)
def _(index: ScalarIndex, *args, **kwargs):
    target_path_and_exprs = immutabledict({immutabledict(): ((just_one(just_one(index.leaf_target_paths)), immutabledict({index.axis: index.value})),)})
    return (UNIT_AXIS_TREE, target_path_and_exprs, ())


@_index_axes_index.register(Slice)
def _(slice_: Slice, /, target_axes, *, seen_target_exprs):
    from pyop3.expr_visitors import replace_terminals, collect_axis_vars

    # TODO: move this code
    from firedrake.cython.dmcommon import filter_sf


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
        for s in slice_.slices
    )
    # NOTE: We should be able to eagerly return here?

    if is_full:
        axis_label = slice_.axis
    else:
        axis_label = slice_.label

    components = []
    component_paths = []
    component_exprs = []
    for slice_component in slice_.slices:
        targets = target_axes[immutabledict({slice_.label: slice_component.label})]
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

        # # DEBUG: using _all_regions breaks things because we only express owned-ness
        # # lazily from the SFs. We therefore have to special case the regionslice for
        # # owned around here.
        if (
            isinstance(slice_component, RegionSliceComponent)
            and slice_component.region in {OWNED_REGION_LABEL, GHOST_REGION_LABEL}
        ):
            orig_regions = target_component._all_regions
        else:
            orig_regions = target_component.regions

        # TODO: Might be clearer to combine these steps
        regions = _prepare_regions_for_slice_component(slice_component, orig_regions)
        indexed_regions = _index_regions(slice_component, regions, parent_exprs=seen_target_exprs)

        orig_size = sum(r.size for r in orig_regions)
        indexed_size = sum(r.size for r in indexed_regions)

        if target_component.sf is not None:
            # If we are specially filtering the owned entries we want to drop the SF
            # to disallow things like `axes.owned.owned`.
            if (
                isinstance(slice_component, RegionSliceComponent)
                and slice_component.region in {OWNED_REGION_LABEL, GHOST_REGION_LABEL}
            ):
                sf = None
            else:
                if isinstance(slice_component, RegionSliceComponent):
                    region_index = target_component._all_region_labels.index(slice_component.region_label)
                    steps = utils.steps([r.size for r in target_component._all_regions], drop_last=False)
                    start, stop = steps[region_index:region_index+2]
                    indices = np.arange(start, stop, dtype=IntType)
                elif isinstance(slice_component, AffineSliceComponent):
                    indices = np.arange(*slice_component.with_size(orig_size), dtype=IntType)
                else:
                    assert isinstance(slice_component, SubsetSliceComponent)
                    indices = slice_component.array.buffer.data_ro

                petsc_sf = filter_sf(target_component.sf.sf, indices, 0, orig_size)
                sf = StarForest(petsc_sf, indexed_size)
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

        component_paths.append(slice_component.component)

    axis = Axis(components, label=axis_label)

    # now do target expressions
    for slice_component in slice_.slices:
        targets = target_axes[immutabledict({slice_.label: slice_component.label})]
        target_axis, target_component_label = just_one(targets.items())
        target_component = just_one(
            c for c in target_axis.components if c.label == target_component_label
        )

        if isinstance(slice_component, RegionSliceComponent):
            if slice_component.region in {OWNED_REGION_LABEL, GHOST_REGION_LABEL}:
                region_index = target_component._all_region_labels.index(slice_component.region)
                steps = utils.steps([r.size for r in target_component._all_regions], drop_last=False)
            else:
                region_index = target_component.region_labels.index(slice_component.region)
                steps = utils.steps([r.size for r in target_component.regions], drop_last=False)
            slice_expr = AxisVar(axis) + steps[region_index]
        elif isinstance(slice_component, AffineSliceComponent):
            slice_expr = AxisVar(axis) * slice_component.step + slice_component.start
        else:
            assert isinstance(slice_component, Subset)
            # replace the index information in the subset buffer
            subset_axis_var = just_one(collect_axis_vars(slice_component.array.layout))
            replace_map = {subset_axis_var.axis_label: AxisVar(axis)}
            slice_expr = replace_terminals(slice_component.array, replace_map)
        component_exprs.append(slice_expr)

    targets = {}
    for slice_component, component, path, expr in zip(slice_.components, components, component_paths, component_exprs, strict=True):
        mytargets = target_axes[immutabledict({slice_.label: slice_component.label})]
        target_axis, target_component_label = just_one(mytargets.items())

        target_path = immutabledict({slice_.axis: path})
        target_expr = immutabledict({slice_.axis: expr})
        # use a 1-tuple here because there are no 'equivalent' layouts for slices
        # like there are for (e.g.) loop indices
        targets[immutabledict({axis.label: component.label})] = ((target_path, target_expr),)

    axes = axis.as_tree()
    targets = immutabledict(targets)
    outer_loops = ()
    return (axes, targets, outer_loops)


@_index_axes_index.register(CalledMap)
def _(called_map: CalledMap, *args, **kwargs):
    # compress the targets here
    compressed_targets = {}
    for leaf_path in called_map.axes.leaf_paths:
        compressed_targets[leaf_path] = tuple(t[leaf_path] for t in called_map.axes.targets)

    return called_map.axes.materialize(), compressed_targets, called_map.outer_loops


def _make_leaf_axis_from_called_map_new(map_, map_name, output_spec, input_paths_and_exprs):
    from pyop3.expr_visitors import replace_terminals

    components = []
    replace_map = merge_dicts(t for _, t in input_paths_and_exprs.values())
    for map_output in output_spec:
        # NOTE: This should be done more eagerly.
        arity = map_output.arity
        if not isinstance(arity, numbers.Integral):
            assert isinstance(arity, Dat)
            arity = arity[map_.index]
        # arity = replace_terminals(map_output.arity, replace_map)
        component = AxisComponent(arity, label=map_output.label)
        components.append(component)
    axis = Axis(components, label=map_name)

    targets = {}
    for component, map_output in zip(components, output_spec, strict=True):
        if not isinstance(map_output, TabulatedMapComponent):
            raise NotImplementedError("Currently we assume only arrays here")

        target_path = immutabledict({map_output.target_axis: map_output.target_component})

        # myvar = just_one(collect_axis_vars(map_output.array.layout))
        # replace_map = {myvar.axis_label: AxisVar(axis.label)}

        # FIXME: I don't really need this stuff, 
        # paths_and_exprs = input_paths_and_exprs | {"anything": ("anything", {leaf_axis: AxisVar(leaf_axis.label})}
        # replace_map = merge_dicts(t for _, t in paths_and_exprs.values())
        replace_map = merge_dicts(t for _, t in input_paths_and_exprs.values())
        target_exprs = immutabledict({map_output.target_axis: replace_terminals(map_output.array, replace_map)})
        targets[immutabledict({axis.label: component.label})] = (target_path, target_exprs)
    targets = immutabledict(targets)

    return (axis, targets)


def index_axes(
    index_tree: Union[IndexTree, Ellipsis],
    loop_context: Mapping | None = None,
    orig_axes: Optional[Union[AxisTree, AxisForest]] = None,
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

    if index_tree is Ellipsis:
        if orig_axes is not None:
            return orig_axes
        else:
            raise ValueError

    # Determine the target axes addressed by the index tree
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
    # compressed_target_axes = defaultdict(list)
    # for index in axis_tree_targets[0].keys():
    #     for axis_tree_target in axis_tree_targets:
    #         compressed_target_axes[index].append(axis_tree_target[index])
    # compressed_target_axes = immutabledict(compressed_target_axes)

    # construct the new, indexed, axis tree
    indexed_axes, target_paths_and_exprs_compressed, outer_loops = make_indexed_axis_tree(index_tree, target_axes)

    indexed_target_paths_and_exprs = expand_collection_of_iterables(target_paths_and_exprs_compressed)

    # If the original axis tree is unindexed then no composition is required.
    if orig_axes is None or isinstance(orig_axes, AxisTree):
        if indexed_axes is UNIT_AXIS_TREE:
            return UnitIndexedAxisTree(
                orig_axes,
                targets=indexed_target_paths_and_exprs,
                layout_exprs={},
                outer_loops=outer_loops,
            )
        else:
            return IndexedAxisTree(
                indexed_axes.node_map,
                orig_axes,
                targets=indexed_target_paths_and_exprs + (indexed_axes._source_path_and_exprs,),
                layout_exprs={},
                outer_loops=outer_loops,
            )

    if orig_axes is None:
        raise NotImplementedError("Need to think about this case")

    all_target_paths_and_exprs = []
    for orig_path in orig_axes.targets:

        match_found = False
        # TODO: would be more intuitive to find match first, instead of looping
        for indexed_path_and_exprs in indexed_target_paths_and_exprs:
            try:
                indexed_path_and_exprs_fixup = _index_info_targets_axes(indexed_axes, indexed_path_and_exprs, orig_axes)
            except MyBadError:
                # does not match, continue
                continue

            assert not match_found, "don't expect multiple hits"
            target_path_and_exprs = compose_targets(orig_axes, orig_path, indexed_axes, indexed_path_and_exprs_fixup)
            match_found = True

            all_target_paths_and_exprs.append(target_path_and_exprs)
        assert match_found, "must hit once"

    # If we have full slices we can get duplicate targets here. This is completely expected
    # but we make assumptions that an indexed tree has unique targets so we filter them here
    # NOTE: really bad code, could use ordered set or similar
    all_target_paths_and_exprs += [immutabledict(indexed_axes._source_path_and_exprs)]
    filtered = []
    for x in all_target_paths_and_exprs:
        if x not in filtered:
            filtered.append(x)
    all_target_paths_and_exprs = filtered

    # TODO: reorder so the if statement captures the composition and this line is only needed once
    if indexed_axes is UNIT_AXIS_TREE:
        return UnitIndexedAxisTree(
            orig_axes.unindexed,
            targets=indexed_target_paths_and_exprs,
            layout_exprs={},
            outer_loops=outer_loops,
        )
    else:
        return IndexedAxisTree(
            indexed_axes.node_map,
            orig_axes.unindexed,
            targets=all_target_paths_and_exprs,
            layout_exprs={},
            outer_loops=outer_loops,
        )


def collect_index_tree_target_paths(index_tree: IndexTree) -> immutabledict:
    return collect_index_tree_target_paths_rec(index_tree, index=index_tree.root)


def collect_index_tree_target_paths_rec(index_tree: IndexTree, *, index: Index) -> immutabledict[Index, Any]:
    # target_paths = {index: collect_index_target_paths(index)}
    # # TODO: index_tree.child?
    # for subindex in filter(None, index_tree.node_map[index.id]):
    #     target_paths |= collect_index_tree_target_paths_rec(index_tree, index=subindex)
    # return immutabledict(target_paths)
    target_paths = {index: []}
    index_target_paths = collect_index_target_paths(index)
    # TODO: index_tree.child?
    for target_path, subindex in zip(index_target_paths, index_tree.node_map[index.id], strict=True):
        if subindex is None:
            target_paths[index].append((target_path, None))
        else:
            subtarget_paths = collect_index_tree_target_paths_rec(index_tree, index=subindex)
            target_paths[index].append((target_path, subtarget_paths))
    return immutabledict(target_paths)


def make_indexed_axis_tree(index_tree: IndexTree, target_axes):
    return make_indexed_axis_tree_rec(index_tree, target_axes, index_path=immutabledict(), seen_target_exprs=immutabledict())


def make_indexed_axis_tree_rec(index_tree: IndexTree, target_axes, *, index_path: ConcretePathT, seen_target_exprs):
    index = index_tree.node_map[index_path]

    index_axis_tree, index_target_paths_and_exprs, index_outer_loops = _index_axes_index(
        index, target_axes,
        seen_target_exprs=seen_target_exprs,
    )

    target_paths_and_exprs = dict(index_target_paths_and_exprs)

    # # merge target paths and exprs (ugly but currently necessary)
    # # we want this to look like: {axis_key: [targets1, targets2]}
    # # where targets1 and targets2 are *distinct* and refer to leaves of the axis tree
    # # These then consist of the tuple targets1 := (path-and-exprs1, path-and-exprs2)
    # # where those are 'equivalent'.
    # target_paths_and_exprs = {index: []}
    # breakpoint()
    # for equivalent_paths, (axis_key, equivalent_exprs) in zip(target_paths[index], index_target_exprs.items(), strict=True):
    #     equivalents = []
    #     for path, exprs in zip(equivalent_paths, equivalent_exprs, strict=True):
    #         equivalents.append((path, exprs))
    #     target_paths_and_exprs[axis_key].append(equivalents)

    axis_tree = index_axis_tree
    outer_loops = list(index_outer_loops)
    for leaf_path, index_component_label in zip(
        index_axis_tree.leaf_paths, index.component_labels, strict=True
    ):
        index_path_ = index_path | {index.label: index_component_label}
        subindex = index_tree.node_map[index_path_]
        if subindex is None:
            continue

        # leaf_key = (leaf[0].id, leaf[1]) if leaf is not None else None
        leaf_key = leaf_path
        seen_target_exprs_ = seen_target_exprs | merge_dicts(exprs for (_, exprs) in index_target_paths_and_exprs[leaf_key])

        # trim current path from 'target_axes' so subtrees can understand things
        target_axes_ = {
            filter_path(orig_path, index_path_): target
            for orig_path, target in target_axes.items()
        }

        subaxis_tree, subtarget_paths_and_exprs, sub_outer_loops = make_indexed_axis_tree_rec(
            index_tree,
            target_axes_,
            index_path=index_path_,
            seen_target_exprs=seen_target_exprs_,
        )

        # leaf_axis_key = (leaf[0], leaf[1]) if leaf is not None else None
        leaf_axis_key = leaf_path
        axis_tree = axis_tree.add_subtree(leaf_axis_key, subaxis_tree)

        # If a subtree has no shape (e.g. if it is a loop index) then append
        # index information to the existing 'None' entry.
        if immutabledict() in subtarget_paths_and_exprs:
            subtarget_paths_and_exprs = dict(subtarget_paths_and_exprs)
            target_paths_and_exprs[immutabledict()] += subtarget_paths_and_exprs.pop(immutabledict())

        for mykey, myvalue in subtarget_paths_and_exprs.items():
            target_paths_and_exprs[leaf_key | mykey] = myvalue

        outer_loops += sub_outer_loops
    outer_loops = tuple(outer_loops)

    return (axis_tree, target_paths_and_exprs, outer_loops)


def compose_targets(orig_axes, orig_target_paths_and_exprs, indexed_axes, indexed_target_paths_and_exprs, *, axis_path=immutabledict()):
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
    from pyop3.expr_visitors import replace_terminals

    # if len(axis_path) == 1:
    #     breakpoint()

    assert not orig_axes.is_empty

    composed_target_paths_and_exprs = {}

    if not axis_path:

        # special handling for None entries
        none_mapped_target_path = {}
        none_mapped_target_exprs = {}

        orig_none_mapped_target_path, orig_none_mapped_target_exprs = orig_target_paths_and_exprs.get(axis_path, ({}, {}))

        myreplace_map = indexed_target_paths_and_exprs.get(immutabledict(), ({}, {}))[1]
        none_mapped_target_path |= orig_none_mapped_target_path
        for orig_axis_label, orig_index_expr in orig_none_mapped_target_exprs.items():
            none_mapped_target_exprs[orig_axis_label] = replace_terminals(orig_index_expr, myreplace_map)

        # Now add any extra 'None-indexed' axes.
        target_axis_paths, replace_map = indexed_target_paths_and_exprs.get(immutabledict(), [(), immutabledict()])

        for target_axis_path in target_axis_paths:
            orig_target_path, orig_target_exprs = orig_target_paths_and_exprs.get(target_axis_path, (immutabledict(), immutabledict()))

            none_mapped_target_path |= orig_target_path
            for orig_axis_label, orig_index_expr in orig_target_exprs.items():
                none_mapped_target_exprs[orig_axis_label] = replace_terminals(orig_index_expr, myreplace_map)

        # Only store if non-empty
        if strictly_all((none_mapped_target_path, none_mapped_target_exprs)):
            composed_target_paths_and_exprs[immutabledict()] = (
                immutabledict(none_mapped_target_path), immutabledict(none_mapped_target_exprs)
            )

        if indexed_axes.is_empty or indexed_axes is UNIT_AXIS_TREE:
            return immutabledict(composed_target_paths_and_exprs)

    axis = indexed_axes.node_map[axis_path]

    for component in axis.components:
        path_ = axis_path | {axis.label: component.label}

        target_axis_paths, replace_map = indexed_target_paths_and_exprs.get(path_, ((), None))

        for target_axis_path in target_axis_paths:
            orig_target_path, orig_target_exprs = orig_target_paths_and_exprs.get(target_axis_path, (immutabledict(), immutabledict()))

            # now index exprs
            new_exprs = {}
            for orig_axis_label, orig_index_expr in orig_target_exprs.items():
                new_exprs[orig_axis_label] = replace_terminals(orig_index_expr, replace_map)

            composed_target_paths_and_exprs[path_] = (orig_target_path, immutabledict(new_exprs))

        if indexed_axes.node_map[path_]:
            composed_target_paths_ = compose_targets(
                orig_axes,
                orig_target_paths_and_exprs,
                indexed_axes,
                indexed_target_paths_and_exprs,
                axis_path=path_,
            )
            for mykey, myvalue in composed_target_paths_.items():
                composed_target_paths_and_exprs[path_ | mykey] = myvalue

    return immutabledict(composed_target_paths_and_exprs)


class MyBadError(Exception):
    pass


def matching_target(index_targets, orig_axes: AbstractAxisTree) -> Any | None:
    assert False, "old code"
    """TODO

    This is useful for when multiple interpretations of axis information are
    provided (e.g. with loop indices) and we want to filter for the right one.


    Look at the full target tree to resolve ambiguity in indexing things. For example
    consider a mixed space. A slice over the mesh is not clear as it may refer to the
    axis of either space. Here we construct the full path and pull out the axes that
    are actually desired.

    """
    matching_candidates = []
    matching_target_axess = []
    for candidate in expand_collection_of_iterables(index_targets):
        breakpoint()  # this is going bad because of duplicates
        full_target_path = merge_dicts(candidate.values())

        if full_target_path not in orig_axes.leaf_paths:
            # not a match
            continue

        matching_candidates.append(candidate)
        matching_target_axes = orig_axes.path_with_nodes(orig_axes._node_from_path(full_target_path))
        matching_target_axess.append(matching_target_axes)
    matching_candidate = just_one(matching_candidates)
    matching_target_axes = just_one(matching_target_axess)

    matching_target_axes_by_index = defaultdict(list)
    for axis, component_label in matching_target_axes.items():
        index = just_one(
            idx
            for idx, target_paths in matching_candidate.items()
            if (axis.label, component_label) in target_paths.items()
        )
        matching_target_axes_by_index[index].append((axis, component_label))
    return immutabledict(matching_target_axes_by_index)


def match_index_targets(indexed_axes, index_info, orig_axes):
    assert False, "just an idea for now"
    return _match_index_targets_rec(indexed_axes=indexed_axes, index_info=index_info, orig_axes=orig_axes, indexed_path=immutabledict(), candidate_matches=())


def _match_index_targets_rec(*, indexed_axes, index_info, orig_axes, indexed_path, candidate_matches):
    result = {}
    for indexed_leaf_path in indexed_axes.leaf_paths:

        # 1. get the actual axes that are visited
        none_target_path, _ = index_info.get(immutabledict(), (immutabledict(), immutabledict()))
        target_path_acc = dict(none_target_path)
        for ipath_acc in accumulate_path(indexed_leaf_path):
            target_path, _ = index_info.get(ipath_acc, (immutabledict(), immutabledict()))
            target_path_acc |= target_path
        target_path_acc = immutabledict(target_path_acc)

        if not target_path_acc in orig_axes.node_map:
            raise MyBadError(
                "This means that the leaf of an indexed axis tree doesn't target the original axes")

        # now construct the mapping to specific *full* axis paths, not path elements
        # we need to look at the node map to get the right ordering as target_path_acc
        # is in indexed order, not the order in the original tree
        ordered_target_path = utils.just_one(
            tp
            for tp in orig_axes.node_map.keys()
            if tp == target_path_acc
        )
        partial_to_full_path_map = {}
        acc = immutabledict()
        for ax, c in ordered_target_path.items():
            acc = acc | {ax: c}
            partial_to_full_path_map[ax, c] = acc

        if immutabledict() in index_info:
            # The current index information is currently done per index, rather
            # than with the tree as a whole. To convert these partial results
            # into ones with full path information

            # FIXME: exprs too
            partial_target_path, target_exprs = index_info[immutabledict()]

            target_ids = []
            for target_axis, target_component in partial_target_path.items():
                full_path = partial_to_full_path_map[target_axis, target_component]
                target_ids.append(full_path)
            result[immutabledict()] = (tuple(target_ids), target_exprs)

        ipath_acc = immutabledict()
        for indexed_axis_label, indexed_component_label in indexed_leaf_path.items():
            ipath_acc = ipath_acc | {indexed_axis_label: indexed_component_label}
            target_path, target_exprs = index_info.get(ipath_acc, (immutabledict(), immutabledict()))

            target_ids = []
            for target_axis, target_component in target_path.items():
                full_path = partial_to_full_path_map[target_axis, target_component]
                target_ids.append(full_path)
            result[ipath_acc] = (tuple(target_ids), target_exprs)

    return immutabledict(result)


def _index_info_targets_axes(indexed_axes, index_info, orig_axes) -> bool:
    """Return whether the index information targets the original axis tree.

    This is useful for when multiple interpretations of axis information are
    provided (e.g. with loop indices) and we want to filter for the right one.

    ---

    UPDATE

    Look at the full target tree to resolve ambiguity in indexing things. For example
    consider a mixed space. A slice over the mesh is not clear as it may refer to the
    axis of either space. Here we construct the full path and pull out the axes that
    are actually desired.

    """
    result = {}
    for indexed_leaf_path in indexed_axes.leaf_paths:

        # 1. get the actual axes that are visited
        none_target_path, _ = index_info.get(immutabledict(), (immutabledict(), immutabledict()))
        target_path_acc = dict(none_target_path)
        for ipath_acc in accumulate_path(indexed_leaf_path):
            target_path, _ = index_info.get(ipath_acc, (immutabledict(), immutabledict()))
            target_path_acc |= target_path
        target_path_acc = immutabledict(target_path_acc)

        if target_path_acc not in orig_axes.node_map:
            raise MyBadError(
                "This means that the leaf of an indexed axis tree doesn't target the original axes")

        # if orig_axes.depth == 2:
        #     breakpoint()
        # now construct the mapping to specific *full* axis paths, not path elements
        # we need to look at the node map to get the right ordering as target_path_acc
        # is in indexed order, not the order in the original tree
        ordered_target_path = utils.just_one(
            tp
            for tp in orig_axes.node_map.keys()
            if tp == target_path_acc
        )
        partial_to_full_path_map = {}
        acc = immutabledict()
        for ax, c in ordered_target_path.items():
            acc = acc | {ax: c}
            partial_to_full_path_map[ax, c] = acc

        # if len(ordered_target_path) > 1:
        #     breakpoint()
        #
        # if len(partial_to_full_path_map) > 1:
        #     breakpoint()

        if immutabledict() in index_info:
            # The current index information is currently done per index, rather
            # than with the tree as a whole. To convert these partial results
            # into ones with full path information

            partial_target_path, target_exprs = index_info[immutabledict()]

            full_target_paths = []
            for target_axis, target_component in partial_target_path.items():
                full_target_path = partial_to_full_path_map[target_axis, target_component]
                full_target_paths.append(full_target_path)
            result[immutabledict()] = (tuple(full_target_paths), target_exprs)

        for ipath_acc in accumulate_path(indexed_leaf_path):
            target_path, target_exprs = index_info.get(ipath_acc, (immutabledict(), immutabledict()))

            target_ids = []
            for target_axis, target_component in target_path.items():
                full_path = partial_to_full_path_map[target_axis, target_component]
                target_ids.append(full_path)
            result[ipath_acc] = (tuple(target_ids), target_exprs)

    return immutabledict(result)


# TODO: just get rid of this, assuming the new system works
def expand_compressed_target_paths(compressed_target_paths):
    return expand_collection_of_iterables(compressed_target_paths)


@dataclasses.dataclass(frozen=True)
class IndexIteratorEntry:
    index: LoopIndex
    source_path: immutabledict
    target_path: immutabledict
    source_exprs: immutabledict
    target_exprs: immutabledict

    @property
    def loop_context(self):
        return immutabledict({self.index.id: (self.source_path, self.target_path)})

    @property
    def replace_map(self):
        return immutabledict(
            {self.index.id: merge_dicts([self.source_exprs, self.target_exprs])}
        )

    @property
    def target_replace_map(self):
        return immutabledict(
            {
                self.index.id: {ax: expr for ax, expr in self.target_exprs.items()},
            }
        )

    @property
    def source_replace_map(self):
        return immutabledict(
            {
                self.index.id: {ax: expr for ax, expr in self.source_exprs.items()},
            }
        )


def iter_axis_tree(
    loop_index: LoopIndex,
    axes: AxisTree,
    target_paths,
    index_exprs,
    outer_loops=(),
    axis=None,
    path=immutabledict(),
    indices=immutabledict(),
    target_path=None,
    index_exprs_acc=None,
    no_index=False,
):
    assert False, "old code"
    from pyop3.expr_visitors import evaluate as eval_expr

    # this is a hack, sometimes things are indexed
    if no_index:
        indices = indices | merge_dicts(
            iter_entry.source_exprs for iter_entry in outer_loops
        )
        outer_replace_map = {}
    else:
        outer_replace_map = merge_dicts(
            iter_entry.replace_map for iter_entry in outer_loops
        )
    if target_path is None:
        assert index_exprs_acc is None
        target_path = target_paths.get(None, immutabledict())

        # Substitute the index exprs, which map target to source, into
        # indices, giving target index exprs
        myindex_exprs = index_exprs.get(None, immutabledict())
        # evaluator = ExpressionEvaluator(indices, outer_replace_map)
        new_exprs = {}
        for axlabel, index_expr in myindex_exprs.items():
            new_exprs[axlabel] = eval_expr(index_expr, indices)
        index_exprs_acc = immutabledict(new_exprs)

    if axes.is_empty:
        source_path = immutabledict()
        source_exprs = immutabledict()
        yield IndexIteratorEntry(
            loop_index, source_path, target_path, source_exprs, index_exprs_acc
        )
        return

    axis = axis or axes.root

    for component in axis.components:
        # for efficiency do these outside the loop
        path_ = path | {axis.label: component.label}
        target_path_ = target_path | target_paths.get((axis.id, component.label), {})
        myindex_exprs = index_exprs.get((axis.id, component.label), immutabledict())
        subaxis = axes.child(axis, component)

        # bit of a hack, I reckon this can go as we can just get it from component.count
        # inside as_int
        if isinstance(component.count, Dat):
            mypath = component.count.axes.target_path.get(None, {})
            myindices = component.count.axes.target_exprs.get(None, {})
            if not component.count.axes.is_empty:
                for cax, ccpt in component.count.axes.path_with_nodes(
                    *component.count.axes.leaf
                ).items():
                    mypath.update(component.count.axes.target_path.get((cax.id, ccpt), {}))
                    myindices.update(
                        component.count.axes.target_exprs.get((cax.id, ccpt), {})
                    )

            mypath = immutabledict(mypath)
            myindices = immutabledict(myindices)
            replace_map = indices
        else:
            mypath = immutabledict()
            myindices = immutabledict()
            replace_map = None

        for pt in range(
            _as_int(
                component.count,
                replace_map,
                loop_indices=outer_replace_map,
            )
        ):
            new_exprs = {}
            # evaluator = ExpressionEvaluator(
            #     indices | {axis.label: pt}, outer_replace_map
            # )
            for axlabel, index_expr in myindex_exprs.items():
                # new_index = evaluator(index_expr)
                # new_index = eval_expr(index_expr, path_, indices | {axis.label: pt})
                new_index = eval_expr(index_expr, indices | {axis.label: pt})
                new_exprs[axlabel] = new_index
            index_exprs_ = index_exprs_acc | new_exprs
            indices_ = indices | {axis.label: pt}
            if subaxis:
                yield from iter_axis_tree(
                    loop_index,
                    axes,
                    target_paths,
                    index_exprs,
                    outer_loops,
                    subaxis,
                    path_,
                    indices_,
                    target_path_,
                    index_exprs_,
                )
            else:
                yield IndexIteratorEntry(
                    loop_index, path_, target_path_, indices_, index_exprs_
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
    if len(regions) > 1 and not subset.array.buffer.ordered:
        size = sum(r.size for r in regions)
        return (AxisComponentRegion(size),)
    else:
        return regions


@functools.singledispatch
def _index_regions(*args, **kwargs) -> tuple[AxisComponentRegion, ...]:
    raise TypeError


@_index_regions.register(RegionSliceComponent)
def _(region_component: RegionSliceComponent, regions, *, parent_exprs) -> tuple[AxisComponentRegion, ...]:
    selected_region = just_one(filter(lambda r: r.label == region_component.region, regions))
    return (AxisComponentRegion(selected_region.size),)


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
    from pyop3.expr_visitors import replace_terminals as expr_replace


    size = sum(r.size for r in regions)
    start, stop, step = affine_component.with_size(size)

    if any(isinstance(r.size, (Dat, ArrayBufferExpression)) for r in regions):
        if len(regions) > 1:
            raise NotImplementedError("Only single-region ragged components are supported")
        region = just_one(regions)

        replace_map = {axis_label: expr for (axis_label, expr) in parent_exprs.items()}
        stop = expr_replace(stop, replace_map)
        return (AxisComponentRegion(stop, region.label),)

    indexed_regions = []
    loc = 0
    offset = start
    for region in regions:
        lower_bound = loc
        upper_bound = loc + region.size
        if upper_bound < start or lower_bound >= stop:
            region_size = 0
            offset -= region.size
        else:
            region_size = math.ceil((min(region.size, stop-loc) - offset) / step)
            offset = (offset + region.size) % step

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
    {"a": 3, "b": 2}[0,1,2,3,4] -> {"a": 3, "b": 0}
    {"a": 3, "b": 2}[0,1,2]     -> {"a": 3, "b": 0}
    {"a": 3, "b": 2}[1,4]       -> {"a": 1, "b": 1}
    {"a": 3, "b": 2}[3,4]       -> {"a": 0, "b": 2}

    """
    indices = subset.array.buffer.data_ro

    indexed_regions = []
    loc = 0
    lower_index = 0
    for region in regions:
        upper_index = np.searchsorted(indices, loc+region.size)
        size = upper_index - lower_index
        indexed_region = AxisComponentRegion(size, region.label)
        indexed_regions.append(indexed_region)

        loc += region.size
        lower_index = upper_index
    return tuple(indexed_regions)
