from __future__ import annotations

import collections
import itertools
import functools
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

from immutabledict import immutabledict as idict

from pyop3 import utils
from pyop3.dtypes import IntType
from pyop3.expr.tensor.dat import Dat
from pyop3.tree.axis_tree import AxisTree
from pyop3.tree.axis_tree.tree import AbstractAxisTree, IndexedAxisTree
from pyop3.exceptions import InvalidIndexTargetException, Pyop3Exception
from pyop3.tree.index_tree.tree import CalledMap, IndexTree, LoopIndex, Slice, AffineSliceComponent, ScalarIndex, Index, Map, SubsetSliceComponent, UnparsedSlice
from pyop3.utils import OrderedSet, debug_assert, expand_collection_of_iterables, strictly_all, single_valued, just_one

import pyop3.extras.debug


class IncompletelyIndexedException(Pyop3Exception):
    """Exception raised when an axis tree is incompletely indexed by an index tree/forest."""


# NOTE: Now really should be plural: 'forests'
# NOTE: Is this definitely the case? I think at the moment I always return just a single
# tree per context.
def as_index_forests(forest: Any, /, axes: AbstractAxisTree | None = None, *, strict: bool = False) -> idict:
    """Return a collection of index trees, split by loop context.

    Parameters
    ----------
    forest :
        The object representing an indexing operation.
    axes :
        The axis tree to which the indexing is being applied.
    strict :
        Flag indicating whether or not additional slices should be added
        implicitly. If `False` then extra slices are added to fill up any
        unindexed shape. If `True` then providing an insufficient set of
        indices will raise an exception.

    Returns
    -------
    index_forest
        A mapping from loop contexts to a tuple of equivalent index trees. Loop
        contexts are represented by the mapping ``{loop index id: iterset path}``.

        Multiple index trees are needed because maps are able to yield multiple
        equivalent index trees.
    """
    if axes is None and strict:
        raise ValueError("Cannot do strict checking if no axes are provided to match against")

    if forest is Ellipsis:
        return idict({idict(): (forest,)})

    forests = {}
    compressed_loop_contexts = collect_loop_contexts(forest)
    # We do not care about the ordering of `loop_context` (though we *do* care about
    # the order of iteration).
    for loop_context in expand_collection_of_iterables(compressed_loop_contexts):
        forest_ = _as_index_forest(forest, axes, loop_context)
        matched_forest = []

        found_match = False
        for index_tree in forest_:
            if axes is not None:
                if strict:
                    # Make sure that 'axes' are completely indexed by each of the index
                    # forests. Note that, since the index trees in a forest represent
                    # 'equivalent' indexing operations, only one of them is expected to work.
                    if not _index_tree_completely_indexes_axes(index_tree, axes):
                        continue
                else:
                    # Add extra slices to make sure that index tree targets
                    # all the axes in 'axes'
                    index_tree = complete_index_tree(index_tree, axes)
                    debug_assert(lambda: _index_tree_completely_indexes_axes(index_tree, axes))

            # Each of the index trees in a forest are considered
            # 'equivalent' in that they represent semantically
            # equivalent operations, differing only in the axes that
            # they target. For example, the loop index
            #
            #     p = axis[::2].iter()
            #
            # will target *both* the unindexed `axis`, as well as the
            # intermediate indexed axis `axis[::2]`. There are therefore
            # multiple index trees in play.
            #
            # For maps it is possible for us to have clashes in the target axes
            # (e.g. cells -> vertices and owned cells -> vertices).
            # If we ever hit this we will need to think a bit.
            matched_forest.append(index_tree)
            found_match = True

        if not found_match:
            raise IncompletelyIndexedException(
                "Index forest does not correctly index the axis tree"
            )

        forests[loop_context] = tuple(matched_forest)
    return idict(forests)


# old alias, remove
as_index_forest = as_index_forests


@functools.singledispatch
def collect_loop_contexts(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@collect_loop_contexts.register(IndexTree)
def _(index_tree: IndexTree, /) -> OrderedSet:
    loop_contexts = OrderedSet()
    for index in index_tree.nodes:
        loop_contexts |= collect_loop_contexts(index)

    assert len(loop_contexts) < 2, "By definition an index tree cannot be context-sensitive"
    return loop_contexts


@collect_loop_contexts.register(LoopIndex)
def _(loop_index: LoopIndex, /) -> OrderedSet:
    if not isinstance(loop_index.iterset, AbstractAxisTree):
        raise NotImplementedError("Need to think about context-sensitive itersets and add them here")

    return OrderedSet({
        (loop_index.id, loop_index.iterset.leaf_paths)})

    # return OrderedSet({
    #     (
    #         loop_index.id,
    #         tuple(
    #             tuple(
    #                 leaf_target_paths_per_target[leaf_path]
    #                 for leaf_target_paths_per_target in loop_index.iterset.leaf_target_paths
    #             )
    #             for leaf_path in loop_index.iterset.leaf_paths
    #         )
    #     )
    # })


@collect_loop_contexts.register(CalledMap)
def _(called_map: CalledMap, /) -> OrderedSet:
    return collect_loop_contexts(called_map.index)


@collect_loop_contexts.register(numbers.Number)
@collect_loop_contexts.register(str)
@collect_loop_contexts.register(slice)
@collect_loop_contexts.register(Slice)
@collect_loop_contexts.register(ScalarIndex)
@collect_loop_contexts.register(Dat)
@collect_loop_contexts.register(UnparsedSlice)
def _(index: Any, /) -> OrderedSet:
    return OrderedSet()


@collect_loop_contexts.register(Sequence)
def _(seq: Sequence, /) -> OrderedSet:
    loop_contexts = OrderedSet()
    for item in seq:
        loop_contexts |= collect_loop_contexts(item)
    return loop_contexts


@functools.singledispatch
def _as_index_forest(obj: Any, /, *args, **kwargs) -> tuple[IndexTree]:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_as_index_forest.register(IndexTree)
def _(index_tree: IndexTree, /, *args, **kwargs) -> tuple[IndexTree]:
    return (index_tree,)


@_as_index_forest.register(Index)
def _(index: Index, /, axes, loop_context) -> tuple[IndexTree]:
    cf_indices = _as_context_free_indices(index, loop_context, axis_tree=axes, path=idict())
    return tuple(IndexTree(cf_index) for cf_index in cf_indices)


@_as_index_forest.register(tuple)
def _(seq: tuple, /, axes, loop_context) -> tuple[IndexTree]:
    # The indices can contain a mixture of 'true' indices (i.e. subclasses of
    # `Index`) and 'sugar' indices (e.g. integers, strings and slices). The former
    # may be used in any order since they declare the axes they target whereas
    # the latter are order dependent.
    index_nests = _index_forest_from_iterable(seq, axes, loop_context, path=idict())
    return tuple(map(IndexTree.from_nest, index_nests))


def _index_forest_from_iterable(indices, axes, loop_context, *, path):
    index, *subindices = indices

    if isinstance(index, IndexTree):
        cf_index_tree = as_context_free_index_tree(index, loop_context)
        if not subindices:
            return (cf_index_tree.to_nest(),)
        else:
            raise NotImplementedError
    else:
        cf_indices = _as_context_free_indices(index, loop_context, axis_tree=axes, path=path)

    if not subindices:
        return cf_indices

    index_nests = []
    for cf_index in cf_indices:
        subnestss = {}
        for component_index in range(cf_index.degree):
            # 'leaf_target_paths' is a tuple of tuples due to having both equivalent
            # targets (e.g. cells and nodes) and multiple components. If there are
            # equivalent targets then we cannot uniquely parse (Python) slices
            # properly and so we set 'path' to 'None' to indicate this.
            if len(cf_index.leaf_target_paths) > 1:
                path_ = None  # disable Python slice parsing
            else:
                path_ = path | cf_index.leaf_target_paths[0][component_index]

            # Each index can produce multiple index trees because of equivalent
            # targets, so we have to collect all of them.
            subnests = _index_forest_from_iterable(subindices, axes, loop_context, path=path_)
            subnestss[component_index] = subnests

        # Now combine all combinations of the possible subtrees
        for subnests in itertools.product(*subnestss.values()):
            index_nest = {cf_index: subnests}
            index_nests.append(index_nest)
    return tuple(index_nests)


@_as_index_forest.register(slice)
@_as_index_forest.register(list)
@_as_index_forest.register(str)
@_as_index_forest.register(numbers.Integral)
@_as_index_forest.register(Dat)
@_as_index_forest.register(UnparsedSlice)
def _(index: Any, /, axes, loop_context) -> tuple[IndexTree]:
    desugared = _desugar_index(index, axes=axes, path=idict())
    return _as_index_forest(desugared, axes, loop_context)


@functools.singledispatch
def _desugar_index(obj: Any, /, *args, **kwargs) -> Index:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_desugar_index.register(UnparsedSlice)
def _(unparsed: UnparsedSlice, /, *, axes, path) -> Index:
    return _desugar_index(unparsed.wrappee, axes=axes, path=path)


@_desugar_index.register(numbers.Integral)
def _(num: numbers.Integral, /, *, axes, path) -> Index:
    if path is None:
        raise RuntimeError("Cannot parse integers here due to ambiguity")

    try:
        axis = axes.node_map[path]
    except KeyError:
        raise InvalidIndexTargetException

    # single-component axis - return a scalar index
    if len(axis.components) == 1 and axis.component.label is None:
        component = just_one(axis.components)
        index = ScalarIndex(axis.label, component.label, num)

    # match on component label
    else:
        component = just_one(c for c in axis.components if c.label == num)
        if component.size == 1:
            index = ScalarIndex(axis.label, component.label, 0)
        else:
            index = Slice(axis.label, [AffineSliceComponent(component.label, label=component.label)], label=axis.label)

    return index


@_desugar_index.register(slice)
def _(slice_: slice, /, *, axes, path) -> Slice:
    if path is None:
        raise RuntimeError("Cannot parse Python slices here due to ambiguity")
    slice_is_full = slice_.start in {None, 0} and slice_.stop is None and slice_.step in {None, 1}

    try:
        axis = axes.node_map[path]
    except KeyError:
        raise InvalidIndexTargetException

    if len(axis.components) == 1:
        if slice_is_full:
            return Slice(
                axis.label,
                [AffineSliceComponent(axis.component.label, label=axis.component.label)],
                label=axis.label,
            )
        else:
            return Slice(
                axis.label,
                [AffineSliceComponent(axis.component.label, slice_.start, slice_.stop, slice_.step)]
            )
    elif slice_is_full:
        # just take everything, keep the labels around (for now, eventually want a special type for this)
        return Slice(
            axis.label,
            [AffineSliceComponent(component.label, label=component.label) for component in axis.components],
            label=axis.label,
        )
    else:
        # badindexexception?
        # NOTE: We could in principle match multi-component things if the component
        # labels form a continuous sequence of integers
        raise ValueError(
            "Cannot slice multi-component things using generic slices, ambiguous"
        )


@_desugar_index.register(list)
def _(list_: list, /, *, axes, path) -> Slice:
    if path is None:
        raise RuntimeError("Cannot parse a list here due to ambiguity")

    try:
        axis = axes.node_map[path]
    except KeyError:
        raise InvalidIndexTargetException

    if len(axis.components) == 1:
        dat = Dat.from_sequence(list_, IntType)
        return _desugar_index(dat, axes=axes, path=path)
    else:
        return Slice(
            axis.label,
            [
                AffineSliceComponent(component_label, label=component_label)
                for component_label in list_
            ],
            label=axis.label,
        )

@_desugar_index.register(Dat)
def _(dat: Dat, /, *, axes, path) -> Slice:
    if path is None:
        raise RuntimeError("Cannot parse Python slices here due to ambiguity")
    axis = axes.node_map[path]

    if len(axis.components) == 1:
        slice_cpt = SubsetSliceComponent(axis.component.label, dat)
        return Slice(axis.label, [slice_cpt])
    else:
        # badindexexception?
        # NOTE: We could in principle match multi-component things if the component
        # labels form a continuous sequence of integers
        raise ValueError(
            "Cannot slice multi-component things using generic slices, ambiguous"
        )

@_desugar_index.register(str)
@_desugar_index.register(tuple)
def _(label: str, /, *, axes, path) -> Index:
    # take a full slice of a component with a matching label
    axis = axes.node_map[path]
    component = just_one(c for c in axis.components if c.label == label)

    if component.size == 1:
        return ScalarIndex(axis.label, component.label, 0)
    else:
        return Slice(axis.label, [AffineSliceComponent(component.label, label=component.label)], label=axis.label)


# TODO: This function needs overhauling to work in more cases.
def complete_index_tree(index_tree: IndexTree, axes: AxisTree) -> IndexTree:
    return _complete_index_tree_rec(index_tree=index_tree, axes=axes, path=idict(), possible_target_paths_acc=(idict(),))


def _complete_index_tree_rec(
    *, index_tree: IndexTree, axes: AxisTree, path: ConcretePathT, possible_target_paths_acc,
) -> IndexTree:
    """Add extra slices to the index tree to match the axes.

    Notes
    -----
    This function is currently only capable of adding additional slices if
    they are "innermost".

    """
    index = index_tree.node_map[path]
    complete_index_tree = IndexTree(index)

    for component_label, equivalent_target_paths in zip(
        index.component_labels, index.leaf_target_paths, strict=True
    ):
        possible_target_paths_acc_ = tuple(
            possible_target_path | target_path
            for possible_target_path in possible_target_paths_acc
            for target_path in equivalent_target_paths
        )

        path_ = path | {index.label: component_label}
        if index_tree.node_map[path_]:
            complete_sub_index_tree = _complete_index_tree_rec(
                index_tree=index_tree,
                axes=axes,
                path=path_,
                possible_target_paths_acc=possible_target_paths_acc_,
            )
        else:
            # At the bottom of the index tree, add any extra slices if needed.
            complete_sub_index_tree = _complete_index_tree_with_slices(
                axes=axes, target_paths=possible_target_paths_acc_, axis_path=idict()
            )

        complete_index_tree = complete_index_tree.add_subtree(
            {index.label: component_label}, complete_sub_index_tree,
        )

    return complete_index_tree


def _complete_index_tree_with_slices(*, axes, target_paths, axis_path: ConcretePathT) -> IndexTree:
    axis = axes.node_map[axis_path]

    # If the label of the current axis exists in any of the target paths then
    # that means that an index already exists that targets that axis, and
    # hence no slice need be produced. At the same time, we can also trim
    # the target paths since we know that we can exclude any that do not
    # use that axis label.
    matching_target_paths = tuple(target_path for target_path in target_paths if axis.label in target_path)

    if len(matching_target_paths) == 0:
        # axis not found, need to emit a slice
        slice_ = Slice(
            axis.label, [AffineSliceComponent(c.label) for c in axis.components]
        )
        index_tree = IndexTree(slice_)

        for axis_component, slice_component_label in zip(
            axis.components, slice_.component_labels, strict=True
        ):
            axis_path_ = axis_path | {axis.label: axis_component.label}
            if axes.node_map[axis_path_]:
                sub_index_tree = _complete_index_tree_with_slices(axes=axes, target_paths=target_paths, axis_path=axis_path_)
                index_tree = index_tree.add_subtree({slice_.label: slice_component_label}, sub_index_tree)

        return index_tree
    else:
        # If the axis is found in 'target_paths' then this means that it has
        # been addressed by the index tree and hence a slice isn't needed.
        # We simply follow the path of the tree that is addressed and recurse.
        axis_component_label = utils.single_valued((
            target_path[axis.label] for target_path in matching_target_paths
        ))
        axis_path_ = axis_path | {axis.label: axis_component_label}
        if axes.node_map[axis_path_]:
            return _complete_index_tree_with_slices(axes=axes, target_paths=matching_target_paths, axis_path=axis_path_)
        else:
            # at the bottom, no more slices needed
            return IndexTree()


def  _index_tree_completely_indexes_axes(index_tree: IndexTree, axes, *, index_path=idict(), possible_target_paths_acc=None) -> bool:
    """Return whether the index tree completely indexes the axis tree.

    This is done by traversing the index tree and collecting the possible target
    paths. At the leaf of the tree we then check whether or not any of the
    possible target paths correspond to a valid path to a leaf of the axis tree.

    """
    if index_path == idict():
        possible_target_paths_acc = (idict(),)

    index = index_tree.node_map[index_path]
    for component_label, equivalent_target_paths in zip(
        index.component_labels, index.leaf_target_paths, strict=True
    ):
        index_path_ = index_path | {index.label: component_label}

        possible_target_paths_acc_ = tuple(
            possible_target_path_acc | possible_target_path
            for possible_target_path_acc in possible_target_paths_acc
            for possible_target_path in equivalent_target_paths
        )

        if index_tree.node_map[index_path_]:
            if not _index_tree_completely_indexes_axes(
                index_tree,
                axes,
                index_path=index_path_,
                possible_target_paths_acc=possible_target_paths_acc_,
            ):
                return False
        else:
            if all(tp not in axes.leaf_paths for tp in possible_target_paths_acc_):
                return False
    return True


@functools.singledispatch
def _as_context_free_indices(obj: Any, /, loop_context: Mapping, **kwargs) -> Index:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_as_context_free_indices.register(slice)
@_as_context_free_indices.register(numbers.Integral)
@_as_context_free_indices.register(UnparsedSlice)
def _(obj, /, loop_context: Mapping, *, axis_tree: AbstractAxisTree, path: ConcretePathT) -> tuple[Slice]:
    return (_desugar_index(obj, axes=axis_tree, path=path),)


@_as_context_free_indices.register(Slice)
@_as_context_free_indices.register(ScalarIndex)
def _(index, /, loop_context: Mapping, **kwargs) -> tuple[Index]:
    return (index,)


@_as_context_free_indices.register(LoopIndex)
def _(loop_index: LoopIndex, /, loop_context, **kwargs) -> tuple[LoopIndex]:
    if loop_index.is_context_free:
        return (loop_index,)
    else:
        iterset = loop_index.iterset
        path = utils.just_one(
            path_
            for path_ in loop_context[loop_index.id]
            if path_ in iterset.leaf_paths
        )
        linear_iterset = iterset.linearize(path)
        # leaf_path = utils.just_one(
        #     for path in loop_context[loop_index.id]
        # )
        #
        # # TODO: Somewhere we are using a set when we shouldn't so we have to search
        # # for the right path when it should really be the first/last entry.
        # leaf = None
        # for path in loop_context[loop_index.id]:
        #     try:
        #         leaf_ = loop_index.iterset._node_from_path(path)
        #         assert leaf is None
        #         leaf = leaf_
        #     except:
        #         continue
        # assert leaf is not None
        #
        # slices = [
        #     Slice(axis_label, [AffineSliceComponent(component_label, label=component_label)], label=axis_label)
        #     for axis_label, component_label in loop_index.iterset.path(leaf, ordered=True)
        # ]
        #
        # # TODO: should accept the iterable directly
        # slices_tree = IndexTree.from_iterable(slices)
        #
        # linear_iterset = loop_index.iterset[slices_tree]
        return (loop_index.copy(iterset=linear_iterset),)


@_as_context_free_indices.register(CalledMap)
def _(called_map, /, loop_context, **kwargs):
    cf_maps = []
    cf_indices = _as_context_free_indices(called_map.from_index, loop_context)

    # loop over semantically equivalent indices
    for cf_index in cf_indices:

        # imagine that we have
        #
        #   {
        #      x -> [[a], [b, c]],
        #      y -> [[a], [d]],
        #   }
        #
        # ie x maps to *either* [a] or [b, c] and y maps to either [a] or [d]
        # then we want to end up with
        #
        #   {
        #     x -> [[a]],  # (should be [a], need a type to capture the extra brackets)
        #     y -> [[a]],  # (should be [a])
        #   }
        #   and
        #   {
        #     x -> [[b, c]],
        #     y -> [[a]],
        #   }
        #   etc
        #
        # In effect for a concrete set of inputs having a concrete set of outputs
        possibilities = []
        for equivalent_input_paths in cf_index.leaf_target_paths:
            found = False
            for input_path in equivalent_input_paths:
                if input_path in called_map.connectivity:
                    found = True
                    for output_spec in called_map.connectivity[input_path]:
                        possibilities.append((input_path, output_spec))
            if not found:
                breakpoint()
            assert found, "must be at least one matching path"

        for input_path, output_spec in possibilities:
            # TODO: Introduce new type here so we don't need the 1-tuple, also assert single input path...
            restricted_connectivity = {input_path: (output_spec,)}
            restricted_map = Map(restricted_connectivity, called_map.name)(cf_index)
            cf_maps.append(restricted_map)
    return tuple(cf_maps)


def as_context_free_index_tree(index_tree: IndexTree, loop_context) -> IndexTree:
    index_forests = as_index_forests(index_tree)
    loop_context_, index_forest = just_one(index_forests.items())
    assert loop_context_ == loop_context
    return just_one(index_forest)
