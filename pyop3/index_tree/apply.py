from __future__ import annotations

import functools
import itertools
import numbers
from collections.abc import Mapping

import numpy as np
from immutabledict import immutabledict as idict

import pyop3.axis_tree
import pyop3.collections
import pyop3.dtypes
import pyop3.exceptions
import pyop3.index_tree.tree
import pyop3.labeled_tree
import pyop3.sf
from pyop3 import utils
from pyop3.index_tree.tree import (
    AffineSliceComponent,
    CalledMap,
    LoopIndex,
    RegionSliceComponent,
    ScalarIndex,
    Slice,
    Slice,
    SubsetSliceComponent,
    TabulatedMapComponent,
    UnitCalledMap,
)


def index_axes(
    index_tree: pyop3.index_tree.tree.IndexTree | Ellipsis,
    loop_context: Mapping | None = None,
    orig_axes: pyop3.axis_tree.AxisTree | pyop3.axis_tree.AxisForest | None = None,
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
        assert isinstance(orig_axes, (pyop3.axis_tree.AxisTree, pyop3.axis_tree.IndexedAxisTree))

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

    # Now construct the new, indexed, axis tree. To make sure that we get unique
    # labels we compute the 'index_count' (which is the number of times that the
    # tree has been indexed from the original unindexed one). Currently we use
    # the axis targets as a nasty proxy for this
    index_count = max(map(len, orig_axes.targets.values()))
    indexed_axes, indexed_targets = make_indexed_axis_tree(index_tree, target_axes, index_count=index_count)

    indexed_targets = pyop3.axis_tree.tree.complete_axis_targets(indexed_targets)

    # If the original axis tree is unindexed then no composition is required.
    if orig_axes is None or isinstance(orig_axes, pyop3.axis_tree.AxisTree):
        if indexed_axes is pyop3.axis_tree.UNIT_AXIS_TREE:
            return pyop3.axis_tree.UnitIndexedAxisTree(
                orig_axes,
                targets=indexed_targets,
            )
        else:
            return pyop3.axis_tree.IndexedAxisTree(indexed_axes, orig_axes, targets=indexed_targets)

    if orig_axes is None:
        raise NotImplementedError("Need to think about this case")

    matching_target = pyop3.axis_tree.tree.match_target(indexed_axes, orig_axes, indexed_targets)
    fullmap = _index_info_targets_axes(indexed_axes, matching_target, orig_axes)
    composed_targets = compose_targets(orig_axes, orig_axes.targets, indexed_axes, matching_target, fullmap)

    # TODO: reorder so the if statement captures the composition and this line is only needed once
    if indexed_axes is pyop3.axis_tree.UNIT_AXIS_TREE:
        retval = pyop3.axis_tree.UnitIndexedAxisTree(
            orig_axes.unindexed,
            targets=composed_targets,
        )
    else:
        retval = pyop3.axis_tree.IndexedAxisTree(
            indexed_axes.node_map,
            orig_axes.unindexed,
            targets=composed_targets,
        )
    return retval


def make_indexed_axis_tree(index_tree: IndexTree, target_axes, index_count: int):
    return _make_indexed_axis_tree_rec(
        index_tree,
        target_axes,
        index_path=idict(),
        expr_replace_map=idict(),
        index_count=index_count,
    )


def _make_indexed_axis_tree_rec(index_tree: IndexTree, target_axes, *, index_path: ConcretePathT, expr_replace_map, index_count):
    index = index_tree.node_map[index_path]

    index_axis_tree, per_index_targets = _index_axes_per_index(
        index, target_axes,
        seen_target_exprs=expr_replace_map,
        index_count=index_count,
    )

    targets: dict[ConcretePathT, tuple[AxisTarget, ...]] \
        = pyop3.collections.StrictlyUniqueDefaultDict(tuple, per_index_targets)

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
            | utils.merge_dicts(t.replace_map for ts in per_index_targets[leaf_path] for t in ts)
        )

        # trim current path from 'target_axes' so subtrees can understand things
        target_axes_ = {
            pyop3.labeled_tree.filter_path(orig_path, index_path_): target
            for orig_path, target in target_axes.items()
        }

        subaxis_tree, subtargets = _make_indexed_axis_tree_rec(
            index_tree,
            target_axes_,
            index_path=index_path_,
            expr_replace_map=expr_replace_map_,
            index_count=index_count,
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

    composed_target = pyop3.collections.StrictlyUniqueDict()

    if not axis_path:
        # special handling for entries that are not tied to a specific axis
        initially_empty_axis_targets = []
        expr_replace_map = utils.merge_dicts(t.replace_map for t in indexed_target[idict()])

        for axis_targets in orig_targets[idict()]:
            XXX = []
            for axis_target in axis_targets:
                composed_expr = replace_terminals(axis_target.expr, expr_replace_map)
                composed_axis_target = pyop3.axis_tree.AxisTarget(axis_target.axis, axis_target.component, composed_expr)
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
                    composed_axis_target = pyop3.axis_tree.AxisTarget(
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

        if indexed_axes.is_empty or indexed_axes is pyop3.axis_tree.UNIT_AXIS_TREE:
            return idict(composed_target)

    axis = indexed_axes.node_map[axis_path]
    for component in axis.components:
        path_ = axis_path | {axis.label: component.label}

        # TODO: use merge_targets, but also need to do a subst
        # merged = merge_targets()
        # some of these cannot be combined, and others can!
        AAA = []
        indexed_axis_targets = indexed_target[path_]
        expr_replace_map = utils.merge_dicts(t.replace_map for t in indexed_axis_targets)
        for target_path in fullmap[path_]:
            BBB = []  # cannot be mixed
            for orig_axis_targets in orig_targets[target_path]:
                composed_axis_targets = []
                for orig_axis_target in orig_axis_targets:
                    composed_expr = replace_terminals(orig_axis_target.expr, expr_replace_map)
                    composed_axis_target = pyop3.axis_tree.AxisTarget(
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
        for indexed_leaf_path_acc in pyop3.labeled_tree.accumulate_path(indexed_leaf_path):
            axis_targets.extend(target[indexed_leaf_path_acc])
        leaf_target_path = utils.merge_dicts(t.path for t in axis_targets)

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

        for indexed_leaf_path_acc in pyop3.labeled_tree.accumulate_path(indexed_leaf_path):
            indexed_axis_targets = target[indexed_leaf_path_acc]
            target_path = utils.merge_dicts(t.path for t in indexed_axis_targets)

            full_target_paths = []
            for target_axis, target_component in target_path.items():
                full_axis_targets_ = partial_to_full_path_map[target_axis, target_component]
                full_target_paths.append(full_axis_targets_)
            result[indexed_leaf_path_acc] = tuple(full_target_paths)
    return idict(result)


# TODO: just get rid of this, assuming the new system works
def expand_compressed_target_paths(compressed_target_paths):
    return expand_collection_of_iterables(compressed_target_paths)


@functools.singledispatch
def _prepare_regions_for_slice_component(slice_component, regions) -> tuple[AxisComponentRegion, ...]:
    raise TypeError


@_prepare_regions_for_slice_component.register
def _(region_component: RegionSliceComponent, regions):
    return tuple(regions)


@_prepare_regions_for_slice_component.register
def _(affine_component: AffineSliceComponent, regions):
    assert affine_component.step != 0
    return tuple(regions) if affine_component.step > 0 else tuple(reversed(regions))


@_prepare_regions_for_slice_component.register
def _(subset: SubsetSliceComponent, regions) -> tuple:
    # We must lose all region information if we are not accessing entries in order
    if len(regions) > 1 and not subset.array.buffer.ordered:
        size = sum(r.size for r in regions)
        return (pyop3.axis_tree.AxisComponentRegion(size),)
    else:
        return regions


@functools.singledispatch
def _index_regions(*args, **kwargs) -> tuple[AxisComponentRegion, ...]:
    raise TypeError


@_index_regions.register
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
    selected_region = selected_region.record_new(label=None, size=size)
    return (selected_region,)


@_index_regions.register
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
    from pyop3.expr.visitors import min_
    from pyop3.expr.visitors import replace_terminals as expr_replace

    if affine_component.is_full_slice:
        indexed_regions = []
        for region in regions:
            size = expr_replace(region.size, parent_exprs)
            indexed_region = region.record_new(size=size)
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
        indexed_region = pyop3.axis_tree.AxisComponentRegion(region_size, region.label)
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

        indexed_region = pyop3.axis_tree.AxisComponentRegion(region_size, region.label)
        indexed_regions.append(indexed_region)
        loc += region.size
    return tuple(indexed_regions)


@_index_regions.register
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

    indices = subset.array.buffer.data_ro

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
        indexed_region = pyop3.axis_tree.AxisComponentRegion(size_, region.label)
        indexed_regions.append(indexed_region)

        loc += region.local_size
        lower_index = upper_index
    return tuple(indexed_regions)


def convert_region_to_affine_slice(region_slice: RegionSliceComponent, axis_component: pyop3.axis_tree.AxisComponent) -> AffineSliceComponent:
    region_index = axis_component.region_labels.index(region_slice.label)
    region_sizes = utils.steps(region.size for region in axis_component.regions)
    return AffineSliceComponent(start=region_sizes[region_index], stop=region_sizes[region_index+1])


def match_target_paths_to_axis_tree(index_tree, orig_axes):
    """Traverse the index tree to determine which axes it targets."""
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
    index_target_paths = index.leaf_target_paths
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
                raise pyop3.exceptions.InvalidIndexTargetException("Candidates do not target the axis tree")

            full_target_axes = utils.single_valued(
                orig_axes.visited_nodes(candidate_path)
                for candidate_path in candidate_target_paths_acc_
                # if candidate_path in orig_axes.node_map
                if candidate_path in orig_axes.leaf_paths
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
def _index_axes_per_index(index: Index, /, *args, **kwargs) -> tuple[pyop3.axis_tree.AxisTree, tuple, tuple[LoopIndex, ...]]:
    """TODO.

    Case 1: loop indices

    Assume we have ``axis[p]`` with ``p`` a linear loop index.
    If p came from other_axis[::2].iter(), then it has *2* possible
    target paths and expressions: over the indexed or unindexed trees.
    Therefore when we index axis with p we must account for this, hence all
    indexing operations return a tuple of possible, equivalent, targets.

    Then, when we combine it all together, if we imagine having 2 loop indices
    like this, then we need the *product* of them to enumerate all possible
    targets.

    """
    raise TypeError(f"No handler provided for {type(index)}")


@_index_axes_per_index.register
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
        axis.label: LoopIndexVar(loop_index, axis.regionless())
        for axis, _ in iterset.visited_nodes(iterset.leaf_path)
    }

    iterset_targets = utils.just_one(collect_leaf_targets(iterset))
    new_targets = utils.freeze({
        idict(): [
            [
                pyop3.axis_tree.AxisTarget(
                    axis_target.axis,
                    axis_target.component,
                    replace_terminals(axis_target.expr, replace_map),
                )
                for axis_target in axis_targets
            ]
            for axis_targets in iterset_targets
        ]
    })

    return (pyop3.axis_tree.UNIT_AXIS_TREE, new_targets)


@_index_axes_per_index.register
def _(index: ScalarIndex, /, target_axes, **kwargs):
    targets = utils.freeze({
        idict(): [[
            pyop3.axis_tree.AxisTarget(index.axis, index.component, index.value),
        ]]
    })
    return (pyop3.axis_tree.UNIT_AXIS_TREE, targets)


@_index_axes_per_index.register
def _(slice_: Slice, /, target_axes, *, seen_target_exprs, index_count: int):
    from pyop3.expr import AxisVar
    from pyop3.expr.visitors import (
        collect_axis_vars,
        get_loop_axes,
        get_shape,
        replace_terminals,
    )


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
    # full slices do not.

    components = []
    for slice_component in slice_.components:
        targets = target_axes[idict({slice_.label: slice_component.label})]
        target_axis, target_component_label = utils.just_one(targets.items())
        target_component = utils.just_one(
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

        if isinstance(target_component.sf, pyop3.sf.StarForest):
            # It is not possible to have a star forest attached to a
            # component with variable extent
            assert isinstance(target_component.local_size, numbers.Integral)

            if isinstance(slice_component, RegionSliceComponent):
                region_index = target_component.region_labels.index(slice_component.region)
                steps = utils.steps([r.local_size for r in target_component.regions], drop_last=False)
                start, stop = steps[region_index:region_index+2]
                indices = np.arange(start, stop, dtype=pyop3.dtypes.IntType)
                sf = None
            else:
                if isinstance(slice_component, AffineSliceComponent):
                    indices = np.arange(*slice_component.with_size(target_component.local_size), dtype=pyop3.dtypes.IntType)
                else:
                    assert isinstance(slice_component, SubsetSliceComponent)
                    # evaluate the subset to get the correct indices
                    subset_axes = utils.just_one(get_shape(slice_component.array))
                    subset_loop_axes = get_loop_axes(slice_component.array)
                    if subset_loop_axes:
                        raise NotImplementedError

                    # subset_expr = CompositeDat(subset_axes, {subset_axes.leaf_path: slice_component.array})
                    # indices = materialize_composite_dat(subset_expr, target_axis.comm).buffer.data_ro
                    indices = slice_component.array.buffer.data_ro

                if isinstance(target_component.sf, pyop3.sf.StarForest):
                    sf = target_component.sf.filter(indices)
                else:
                    assert isinstance(target_component.sf, pyop3.sf.NullStarForest)
                    sf = pyop3.sf.NullStarForest(indices.size)
        else:
            sf = None

        # TODO: Add handling for the other types of slices
        component_size = None
        if target_component._size is not None:
            if isinstance(slice_component, AffineSliceComponent):
                start, stop, step = slice_component.with_size(target_component._size)
                component_size = (stop-start) // step

            elif isinstance(slice_component, RegionSliceComponent):
                region_index = target_component.region_labels.index(slice_component.region)
                component_size = target_component.regions[region_index].size

            if component_size is not None:
                component_size = replace_terminals(component_size, seen_target_exprs)

        component = pyop3.axis_tree.AxisComponent(indexed_regions, label=slice_component.label, sf=sf, size=component_size)
        components.append(component)

    axis_label = slice_.label
    axis = pyop3.axis_tree.Axis(components, label=axis_label)

    # now do target expressions
    targets = {}
    for slice_component, axis_component in zip(slice_.components, axis.components, strict=True):
        index_path = idict({slice_.label: slice_component.label})
        target_axis, target_component_label = utils.just_one(target_axes[index_path].items())
        target_component = utils.just_one(
            c for c in target_axis.components if c.label == target_component_label
        )

        linear_axis = axis.linearize(axis_component.label).regionless()

        if isinstance(slice_component, RegionSliceComponent):
            if slice_component.region in {pyop3.axis_tree.OWNED_REGION_LABEL, pyop3.axis_tree.GHOST_REGION_LABEL}:
                region_index = target_component.region_labels.index(slice_component.region)
                steps = utils.steps([r.size for r in target_component.regions], drop_last=False)
            else:
                region_index = target_component.region_labels.index(slice_component.region)
                steps = utils.steps([r.size for r in target_component.regions], drop_last=False)
            slice_expr = AxisVar(linear_axis) + steps[region_index]
        elif isinstance(slice_component, AffineSliceComponent):
            slice_expr = AxisVar(linear_axis) * slice_component.step + slice_component.start
        else:
            assert isinstance(slice_component, SubsetSliceComponent)
            # replace the index information in the subset buffer
            try:
                subset_axis_var = utils.just_one(collect_axis_vars(slice_component.array.layout))
            except ValueError:
                subset_axis_var = utils.just_one(av for av in collect_axis_vars(slice_component.array.layout) if av.axis_label == axis.label)

            if subset_axis_var.axis.label != linear_axis.label:
                replace_map = {subset_axis_var.axis.label: AxisVar(linear_axis)}
                slice_expr = replace_terminals(slice_component.array, replace_map, assert_modified=True)
            else:
                # FIXME: this isn't nice, should the labels ever match here?
                # labels match, strict=True will cause replace to fail
                slice_expr = slice_component.array
        slice_expr = replace_terminals(slice_expr, seen_target_exprs)

        targets[idict({axis.label: axis_component.label})] = [[
            pyop3.axis_tree.AxisTarget(slice_.axis, slice_component.component, slice_expr),
        ]]

    axes = axis.as_tree()
    targets = utils.freeze(targets)
    return (axes, targets)


@_index_axes_per_index.register
def _(called_map: CalledMap, *args, **kwargs):
    return called_map.axes.materialize(), called_map.axes.targets


@_index_axes_per_index.register
def _(map_: UnitCalledMap, /, *args, **kwargs):
    import pyop3
    from pyop3.expr import AxisVar

    assert map_.is_context_free

    new_targets = {idict(): []}
    assert len(map_.index.axes.targets) == 1
    match_found = False
    for index_targets in map_.index.axes.targets[idict()]:
        if len(index_targets) == 0:
            continue
        index_target = utils.just_one(index_targets)

        try:
            map_components = map_.connectivity[idict({index_target.axis: index_target.component})]
        except KeyError:
            continue

        if match_found:
            raise NotImplementedError("not sure what to do about multiple matches")
        match_found = True
        if len(map_components) != 1:
            raise NotImplementedError("suggests multiple equivalent outputs")
        else:
            map_component = utils.just_one(map_components)

        # now put the index expression from the inner index into the array expression
        axis_var = map_component.array.layout
        assert isinstance(axis_var, AxisVar)
        replace_map = {axis_var: index_target.expr}

        myexpr = pyop3.visitors.replace(map_component.array, replace_map, assert_modified=True)
        new_targets[idict()].append([pyop3.axis_tree.AxisTarget(map_component.target_axis, map_component.target_component, myexpr)])

    assert match_found
    new_targets = utils.freeze(new_targets)

    return (pyop3.axis_tree.UNIT_AXIS_TREE, new_targets)


def _make_leaf_axis_from_called_map_new(map_, map_name, output_spec, input_paths_and_exprs):
    from pyop3.expr.buffer import LinearDatBufferExpression
    from pyop3.expr.visitors import replace_terminals

    components = []
    replace_map = utils.merge_dicts(
        t.replace_map for t in input_paths_and_exprs
    )
    for map_output in output_spec:
        # NOTE: This should be done more eagerly.
        arity = map_output.arity
        if not isinstance(arity, numbers.Integral):
            assert isinstance(arity, LinearDatBufferExpression)
            # arity = arity[map_.index]
            arity = replace_terminals(map_output.arity, replace_map, assert_modified=True)
        component = pyop3.axis_tree.AxisComponent(arity, label=map_output.label)
        components.append(component)
    axis = pyop3.axis_tree.Axis(components, label=map_name)

    targets = {}
    for component, map_output in zip(components, output_spec, strict=True):
        if not isinstance(map_output, TabulatedMapComponent):
            raise NotImplementedError("Currently we assume only arrays here")

        target_axis = map_output.target_axis
        target_component = map_output.target_component
        expr = replace_terminals(map_output.array, replace_map, assert_modified=True)
        axis_target = pyop3.axis_tree.AxisTarget(target_axis, target_component, expr)
        targets[idict({axis.label: component.label})] = ((axis_target,),)
    targets = idict(targets)

    return (axis, targets)


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
        _collect_leaf_targets_per_leaf(axes, leaf_path, None, pyop3.collections.UniqueList())
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
    leaf_targets = _collect_leaf_targets_per_leaf(axes, leaf_path, None, pyop3.collections.UniqueList())
    for leaf_target in leaf_targets:
        yield utils.merge_dicts(t.path for t in leaf_target)
