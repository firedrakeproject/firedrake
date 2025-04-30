from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import numbers
import operator
from collections.abc import Iterable, Mapping
from typing import Any, Optional

import numpy as np
from immutabledict import ImmutableOrderedDict
from pyop3.array.dat import DatBufferExpression, PetscMatBufferExpression
from pyop3.buffer import AbstractArrayBuffer, PetscMatBuffer, AbstractPetscMatBuffer
from pyrsistent import pmap, PMap
from petsc4py import PETSc

from pyop3 import utils
from pyop3.array import Array, Dat, Mat, LinearDatBufferExpression, BufferExpression, NonlinearDatBufferExpression
from pyop3.axtree.tree import AxisVar, Expression, Operator, Add, Mul, AbstractAxisTree, IndexedAxisTree, AxisTree, Axis, LoopIndexVar, merge_trees2, ExpressionT, Terminal, AxisComponent, relabel_path
from pyop3.dtypes import IntType
from pyop3.utils import OrderedSet, just_one


# should inherit from _Dat
# or at least be an Expression!
@dataclasses.dataclass(frozen=True)
class CompositeDat:
    axis_tree: AxisTree
    leaf_exprs: Any
    loop_indices: Any
    dtype: np.dtype

    # def __str__(self) -> str:
    #     return f"acc({self.expr})"


# TODO: could make a postvisitor
@functools.singledispatch
def evaluate(obj: Any, /, *args, **kwargs):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@evaluate.register
def _(dat: Dat, indices):
    if dat.parent:
        raise NotImplementedError

    if not dat.axes.is_linear:
        # guess this is optional at the top level, extra kwarg?
        raise NotImplementedError
    else:
        path = dat.axes.path(dat.axes.leaf)
    offset = evaluate(dat.axes.subst_layouts()[path], indices)
    return dat.buffer.data_ro_with_halos[offset]


@evaluate.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, indices):
    offset = evaluate(expr.layout, indices)
    return expr.buffer.data_ro_with_halos[offset]


@evaluate.register
def _(expr: Add, *args, **kwargs):
    return evaluate(expr.a, *args, **kwargs) + evaluate(expr.b, *args, **kwargs)


@evaluate.register
def _(mul: Mul, *args, **kwargs):
    return evaluate(mul.a, *args, **kwargs) * evaluate(mul.b, *args, **kwargs)


@evaluate.register
def _(num: numbers.Number, *args, **kwargs):
    return num


@evaluate.register
def _(var: AxisVar, indices):
    return indices[var.axis_label]


@evaluate.register(LoopIndexVar)
def _(loop_var: LoopIndexVar):
    return OrderedSet({loop_var.index})


@functools.singledispatch
def collect_loop_index_vars(obj: Any, /) -> OrderedSet:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_loop_index_vars.register(LoopIndexVar)
def _(loop_var: LoopIndexVar):
    return OrderedSet({loop_var})


@collect_loop_index_vars.register(AxisVar)
@collect_loop_index_vars.register(numbers.Number)
def _(var):
    return OrderedSet()

@collect_loop_index_vars.register(Operator)
def _(op: Operator):
    return collect_loop_index_vars(op.a) | collect_loop_index_vars(op.b)


@collect_loop_index_vars.register(Dat)
def _(dat: Dat, /) -> OrderedSet:
    loop_indices = OrderedSet()

    if dat.parent:
        loop_indices |= collect_loop_index_vars(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_loop_index_vars(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_loop_index_vars.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /) -> OrderedSet:
    return collect_loop_index_vars(expr.layout)


@collect_loop_index_vars.register(Mat)
def _(mat: Mat, /) -> OrderedSet:
    loop_indices = OrderedSet()
    if mat.parent:
        loop_indices |= collect_loop_index_vars(mat.parent)

    for cs_axes in {mat.raxes, mat.caxes}:
        for cf_axes in cs_axes.context_map.values():
            for leaf in cf_axes.leaves:
                path = cf_axes.path(leaf)
                loop_indices |= collect_loop_index_vars(cf_axes.subst_layouts()[path])
    return loop_indices


@functools.singledispatch
def restrict_to_context(obj: Any, /, loop_context):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@restrict_to_context.register(numbers.Number)
@restrict_to_context.register(AxisVar)
@restrict_to_context.register(DatBufferExpression)
def _(var: Any, /, loop_context) -> Any:
    return var


@restrict_to_context.register(Operator)
def _(op: Operator, /, loop_context):
    return type(op)(restrict_to_context(op.a, loop_context), restrict_to_context(op.b, loop_context))


@restrict_to_context.register(Array)
def _(array: Array, /, loop_context):
    return array.with_context(loop_context)


# NOTE: bad name?? something 'shape'? 'make'?
# always return an AxisTree?

# NOTE: visited_axes is more like visited_components! Only need axis labels and component information
@functools.singledispatch
def extract_axes(obj: Any, /, visited_axes, loop_axes, cache) -> AbstractAxisTree:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@extract_axes.register(numbers.Number)
def _(var: Any, /, visited_axes, loop_axes, cache) -> AxisTree:
    try:
        return cache[var]
    except KeyError:
        return cache.setdefault(var, AxisTree())


@extract_axes.register(LoopIndexVar)
def _(loop_var: LoopIndexVar, /, visited_axes, loop_indices: tuple[LoopIndex, ...], cache) -> AxisTree:
    try:
        return cache[loop_var]
    except KeyError:
        pass

    # replace LoopIndexVars in any component sizes with AxisVars
    loop_index_replace_map = {}
    for loop_index in loop_indices:
        for axis in loop_index.iterset.nodes:
            loop_index_replace_map[(loop_index.id, axis.label)] = AxisVar(f"{axis.label}_{loop_index.id}")


    selected_axis = just_one(axis for loop_index in loop_indices for axis in loop_index.iterset.nodes if axis.label == loop_var.axis_label and loop_index.id == loop_var.loop_id
                             )

    new_components = []
    for component in axis.components:
        if isinstance(component.count, numbers.Integral):
            new_component = component
        else:
            new_count_axes = extract_axes(just_one(component.count.leaf_layouts.values()), visited_axes, loop_indices, cache)
            new_count = Dat(new_count_axes, data=component.count.buffer)
            new_component = AxisComponent(new_count, component.label)
        new_components.append(new_component)
    new_axis = Axis(new_components, f"{axis.label}_{loop_var.loop_id}")
    return cache.setdefault(loop_var, new_axis.as_tree())


@extract_axes.register(AxisVar)
def _(var: AxisVar, /, visited_axes, loop_axes, cache) -> AxisTree:
    try:
        return cache[var]
    except KeyError:
        axis = utils.single_valued(axis for axis in visited_axes.nodes if axis.label == var.axis_label)
        tree = AxisTree(axis)
        return cache.setdefault(var, tree)


@extract_axes.register(Operator)
def _(op: Operator, /, visited_axes, loop_axes, cache):
    return merge_trees2(extract_axes(op.a, visited_axes, loop_axes, cache), extract_axes(op.b, visited_axes, loop_axes, cache))


@extract_axes.register(Array)
def _(array: Array, /, visited_axes, loop_axes, cache):
    return array.axes


@extract_axes.register(LinearDatBufferExpression)
def _(expr, /, visited_axes, loop_axes, cache):
    return extract_axes(expr.layout, visited_axes, loop_axes, cache)


@extract_axes.register(CompositeDat)
def _(dat, /, visited_axes, loop_axes, cache):
    assert visited_axes == dat.visited_axes
    assert loop_axes == dat.loop_axes
    return extract_axes(dat.expr, visited_axes, loop_axes, cache)


@functools.singledispatch
def _relabel_axes(obj: Any, suffix: str) -> AbstractAxisTree:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_relabel_axes.register(AxisTree)
def _(axes: AxisTree, suffix: str) -> AxisTree:
    relabelled_node_map = _relabel_node_map(axes.node_map, suffix)
    return AxisTree(relabelled_node_map)


@_relabel_axes.register(IndexedAxisTree)
def _(axes: IndexedAxisTree, suffix: str) -> IndexedAxisTree:
    relabelled_node_map = _relabel_node_map(axes.node_map, suffix)

    # I think that I can leave unindexed the same here and just tweak the target expressions
    relabelled_targetss = tuple(
        _relabel_targets(targets, suffix)
        for targets in axes.targets
    )
    return IndexedAxisTree(relabelled_node_map, unindexed=axes.unindexed, targets=relabelled_targetss)


def _relabel_node_map(node_map: Mapping, suffix: str) -> PMap:
    relabelled_node_map = {}
    for parent, children in node_map.items():
        relabelled_children = []
        for child in children:
            if child:
                relabelled_child = child.copy(label=child.label+suffix)
                relabelled_children.append(relabelled_child)
            else:
                relabelled_children.append(None)
        relabelled_node_map[parent] = tuple(relabelled_children)
    return pmap(relabelled_node_map)


# NOTE: This only relabels the expressions. The target path is unchanged because I think that that is fine here
def _relabel_targets(targets: Mapping, suffix: str) -> PMap:
    relabelled_targets = {}
    for axis_key, (path, exprs) in targets.items():
        relabelled_exprs = {
            axis_label: relabel(expr, suffix) for axis_label, expr in exprs.items()
        }
        relabelled_targets[axis_key] = (path, relabelled_exprs)
    return pmap(relabelled_targets)


# TODO: make this a nice generic traversal
@functools.singledispatch
def replace_terminals(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@replace_terminals.register(Terminal)
def _(terminal: Terminal, /, replace_map) -> ExpressionT:
    return replace_map.get(terminal.terminal_key, terminal)


@replace_terminals.register(numbers.Number)
def _(var: ExpressionT, /, replace_map) -> ExpressionT:
    return var


# I don't like doing this.
@replace_terminals.register(Dat)
def _(dat: Dat, /, replace_map):
    return replace_terminals(dat._as_expression_dat(), replace_map)


@replace_terminals.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /, replace_map) -> LinearDatBufferExpression:
    new_layout = replace_terminals(expr.layout, replace_map)
    return LinearDatBufferExpression(expr.buffer, new_layout)


@replace_terminals.register(Operator)
def _(op: Operator, /, replace_map) -> Operator:
    return type(op)(replace_terminals(op.a, replace_map), replace_terminals(op.b, replace_map))


@functools.singledispatch
def replace(obj: Any, /, replace_map) -> ExpressionT:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@replace.register(AxisVar)
@replace.register(LoopIndexVar)
def _(var: Any, /, replace_map) -> ExpressionT:
    return replace_map.get(var, var)


@replace.register(numbers.Number)
def _(num: numbers.Number, /, replace_map) -> numbers.Number:
    return num


# I don't like doing this.
@replace.register(Dat)
def _(dat: Dat, /, replace_map):
    return replace(dat._as_expression_dat(), replace_map)


@replace.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    if expr in replace_map:
        return replace_map[expr]
    else:
        new_layout = replace(expr.layout, replace_map)
        return type(expr)(expr.buffer, new_layout)


@replace.register(CompositeDat)
def _(dat: CompositeDat, /, replace_map):
    # TODO: Can have a flag that determines the replacement order (pre/post)
    if dat in replace_map:
        return replace_map[dat]
    else:
        raise AssertionError("Not sure about this here...")
        replaced_layout = replace(dat.layout, replace_map)
        return dat.reconstruct(layout=replaced_layout)


@replace.register(Operator)
def _(op: Operator, /, replace_map) -> Operator:
    return type(op)(replace(op.a, replace_map), replace(op.b, replace_map))


@functools.singledispatch
def concretize_layouts(obj: Any, /) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@concretize_layouts.register(BufferExpression)
def _(expr: BufferExpression, /) -> BufferExpression:
    return expr


@concretize_layouts.register(numbers.Number)
def _(num: numbers.Number, /) -> numbers.Number:
    return num


@concretize_layouts.register(Dat)
def _(dat: Dat, /) -> Any:
    if dat.axes.is_linear:
        layout = just_one(dat.axes.leaf_subst_layouts.values())
        return LinearDatBufferExpression(dat.buffer, layout)
    else:
        return NonlinearDatBufferExpression(dat.buffer, dat.axes.leaf_subst_layouts)


@concretize_layouts.register(Mat)
def _(mat: Mat, /) -> Any:
    breakpoint()


@functools.singledispatch
def collect_tensor_shape(obj: Any, /) -> tuple[AxisTree, ...] | None:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_tensor_shape.register(numbers.Number)
@collect_tensor_shape.register(BufferExpression)
def _(obj: Any, /) -> None:
    return None


@collect_tensor_shape.register(Dat)
def _(dat: Dat, /) -> tuple[AxisTree]:
    return (dat.axes.materialize(),)


@collect_tensor_shape.register(Mat)
def _(mat: Mat, /) -> tuple[AxisTree,AxisTree]:
    return (mat.raxes.materialize(), mat.caxes.materialize())


@functools.singledispatch
def collect_tensor_candidate_indirections(obj: Any, /, **kwargs) -> ImmutableOrderedDict:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_tensor_candidate_indirections.register(numbers.Number)
def _(num: numbers.Number, /, **kwargs) -> ImmutableOrderedDict:
    return ImmutableOrderedDict()


@collect_tensor_candidate_indirections.register(LinearDatBufferExpression)
def _(dat_expr: LinearDatBufferExpression, /, *, axis_trees: Iterable[AxisTree], loop_indices: tuple[LoopIndex, ...], optimize: bool) -> ImmutableOrderedDict:
    if not isinstance(dat_expr.buffer, AbstractArrayBuffer):
        raise NotImplementedError("Currently we assume that Dats are based on an underlying array buffer")

    axis_tree = just_one(axis_trees)
    return ImmutableOrderedDict({
        dat_expr: collect_candidate_indirections(dat_expr.layout, axis_tree, loop_indices)
    })


@collect_tensor_candidate_indirections.register(NonlinearDatBufferExpression)
def _(dat_expr: NonlinearDatBufferExpression, /, *, axis_trees: Iterable[AxisTree], loop_indices: tuple[LoopIndex, ...], optimize: bool) -> ImmutableOrderedDict:
    if not isinstance(dat_expr.buffer, AbstractArrayBuffer):
        raise NotImplementedError("Currently we assume that Dats are based on an underlying array buffer")

    axis_tree = just_one(axis_trees)
    return ImmutableOrderedDict({
        (dat_expr, path): collect_candidate_indirections(layout, axis_tree.linearize(path), loop_indices)
        for path, layout in dat_expr.layouts.items()
    })


@collect_tensor_candidate_indirections.register(Mat)
def _(mat: Mat, loop_axes):
    # think about the buffer type... 
    if isinstance(mat.buffer, AbstractPetscMatBuffer):
        breakpoint()
    breakpoint()
    return mat.candidate_layouts(loop_axes)



# TODO: rename to concretize_array_accesses or concretize_arrays
# @functools.singledispatch
# def concretize_arrays(obj: Any, /, *args, **kwargs) -> Expression:
#     raise TypeError(f"No handler defined for {type(obj).__name__}")
#
#
# @concretize_arrays.register(Dat)
# def _(dat: Dat, /, loop_axes) -> NonlinearDatBufferExpression:
#     selected_layouts = dat.axes.leaf_subst_layouts
#     # selected_layouts = {}
#     # for leaf_path in dat.axes.leaf_paths:
#     #     possible_layouts = candidate_layouts[(dat, leaf_path)]
#     #     selected_layout, _ = min(possible_layouts, key=lambda item: item[1])
#     #     selected_layouts[leaf_path] = selected_layout
#
#     return NonlinearDatBufferExpression(dat.buffer, selected_layouts)
#
#
# @concretize_arrays.register(Mat)
# def _(mat: Mat, /, loop_axes) -> PetscMatBufferExpression:
#     from pyop3.insn_visitors import materialize_composite_dat
#
#     # TODO: Add intermediate type to assert that there is no longer a parent attr
#     assert mat.parent is None
#
#     # NOTE: default_candidate_layouts shouldn't return any cost because it doesn't matter here
#     # Actually this might not be quite true: for non-PETSc matrices we have some amount of choice
#
#     # FIXME: this is bad for temporaries because it means we needlessly tabulate
#     # layouts = mat.default_candidate_layouts(loop_axes)
#     layouts = mat.candidate_layouts(loop_axes)
#
#     # FIXME: Different treatment for buffer and petsc mats here
#     if isinstance(mat.buffer, AbstractPetscMatBuffer):
#         row_layout = materialize_composite_dat(layouts[(mat, "anything", 0)][0][0])
#         column_layout = materialize_composite_dat(layouts[(mat, "anything", 1)][0][0])
#         return PetscMatBufferExpression(mat.buffer, row_layout, column_layout)
#     else:
#         row_layouts = {}
#         for leaf_path in mat.raxes.pruned.leaf_paths:
#             possible_row_layouts = layouts[(mat, leaf_path, 0)]
#             selected_layout, _ = just_one(possible_row_layouts)
#
#             if isinstance(selected_layout, CompositeDat):
#                 selected_layout = materialize_composite_dat(selected_layout)
#
#             row_layouts[leaf_path] = selected_layout
#
#         col_layouts = {}
#         for leaf_path in mat.caxes.pruned.leaf_paths:
#             possible_col_layouts = layouts[(mat, leaf_path, 1)]
#             selected_layout, _ = just_one(possible_col_layouts)
#
#             if isinstance(selected_layout, CompositeDat):
#                 selected_layout = materialize_composite_dat(selected_layout)
#
#             col_layouts[leaf_path] = selected_layout
#
#         # merge layouts
#         layouts = {}
#         for row_path, row_layout in row_layouts.items():
#             for column_path, column_layout in col_layouts.items():
#                 relabelled_row_path = relabel_path(row_path, "0")
#                 relabelled_column_path = relabel_path(column_path, "1")
#
#                 replace_map = {var.axis_label: AxisVar(var.axis_label+"_0") for var in collect_axis_vars(row_layout)}
#                 relabelled_row_layout = replace_terminals(row_layout, replace_map)
#
#                 replace_map = {var.axis_label: AxisVar(var.axis_label+"_1") for var in collect_axis_vars(column_layout)}
#                 relabelled_column_layout = replace_terminals(column_layout, replace_map)
#
#                 layouts[ImmutableOrderedDict(relabelled_row_path|relabelled_column_path)] = relabelled_row_layout * mat.caxes.size + relabelled_column_layout
#         # TODO: Is this the right type?
#         return NonlinearDatBufferExpression(mat.buffer, layouts)


# @concretize_arrays.register(numbers.Number)
# @concretize_arrays.register(AxisVar)
# @concretize_arrays.register(BufferExpression)
# @concretize_arrays.register(LoopIndexVar)
# def _(var: Any, /, loop_axes) -> Any:
#     return var
#
#
# @concretize_arrays.register(Operator)
# def _(op: Operator, /, loop_axes) -> Operator:
#     return type(op)(concretize_arrays(op.a, loop_axes), concretize_arrays(op.b, loop_axes))


# TODO: account for non-affine accesses in arrays and selectively apply this
INDIRECTION_PENALTY_FACTOR = 5

MINIMUM_COST_TABULATION_THRESHOLD = 128
"""The minimum cost below which tabulation will not be considered.

Indirections with a cost below this are considered as fitting into cache and
so memory optimisations are ineffectual.

"""


@functools.singledispatch
def collect_candidate_indirections(obj: Any, /, *, visited_axes, loop_indices: tuple[LoopIndex, ...]) -> tuple[tuple[Any, int], ...]:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_candidate_indirections.register(numbers.Number)
@collect_candidate_indirections.register(AxisVar)
@collect_candidate_indirections.register(LoopIndexVar)
def _(var: Any, /, visited_axes, loop_axes) -> tuple:
    return ((var, 0),)


@collect_candidate_indirections.register(Operator)
def _(op: Operator, /, visited_axes, loop_axes) -> tuple:
    a_result = collect_candidate_indirections(op.a, visited_axes, loop_axes)
    b_result = collect_candidate_indirections(op.b, visited_axes, loop_axes)

    candidates = []
    for (a_expr, a_cost), (b_expr, b_cost) in itertools.product(a_result, b_result):
        candidate_expr = type(op)(a_expr, b_expr)
        candidate_cost = a_cost + b_cost
        candidates.append((candidate_expr, candidate_cost))

    # Now also include a candidate representing the packing of the expression
    # into a Dat. The cost for this is simply the size of the resulting array.
    # Only do this when the cost is large as small arrays will fit in cache
    # and not benefit from the optimisation.
    if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost in candidates):
        axes = extract_axes(op, visited_axes, loop_axes, {})
        compressed_expr = CompositeDat(axes, {visited_axes.leaf_path: op}, loop_axes, IntType)
        candidates.append((compressed_expr, axes.size))

    return tuple(candidates)


@collect_candidate_indirections.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /, visited_axes, loop_axes) -> tuple:
    candidates = []
    for layout_expr, layout_cost in collect_candidate_indirections(expr.layout, visited_axes, loop_axes):
        candidate_expr = LinearDatBufferExpression(expr.buffer, layout_expr)
        # The cost of an expression dat (i.e. the memory volume) is given by...
        # Remember that the axes here described the outer loops that exist and that
        # index expressions that do not access data (e.g. 2i+j) have a cost of zero.
        # dat[2i+j] would have a cost equal to ni*nj as those would be the outer loops
        dat_cost = extract_axes(expr.layout, visited_axes, loop_axes, cache={}).size
        # TODO: Only apply penalty for non-affine layouts
        candidate_cost = dat_cost + layout_cost * INDIRECTION_PENALTY_FACTOR
        candidates.append((candidate_expr, candidate_cost))

    if any(cost > MINIMUM_COST_TABULATION_THRESHOLD for _, cost in candidates):
        axes = extract_axes(expr, visited_axes, loop_axes, {})
        candidates.append((CompositeDat(axes, {visited_axes.leaf_path: expr}, loop_axes, IntType), axes.size))
    return tuple(candidates)


@functools.singledispatch
def compute_indirection_cost(obj: Any, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@compute_indirection_cost.register(AxisVar)
@compute_indirection_cost.register(LoopIndexVar)
@compute_indirection_cost.register(numbers.Number)
def _(var: Any, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    return 0


@compute_indirection_cost.register(Operator)
def _(op: Operator, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    if seen_exprs_mut is not None:
        if op in seen_exprs_mut:
            return 0
        else:
            seen_exprs_mut.add(op)

    return (
        compute_indirection_cost(op.a, visited_axes, loop_axes, seen_exprs_mut, cache)
        + compute_indirection_cost(op.b, visited_axes, loop_axes, seen_exprs_mut, cache)
    )


@compute_indirection_cost.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    if seen_exprs_mut is not None:
        if expr in seen_exprs_mut:
            return 0
        else:
            seen_exprs_mut.add(expr)

    # The cost of a buffer expression (i.e. the memory volume) is given by...
    # Remember that the axes here described the outer loops that exist and that
    # index expressions that do not access data (e.g. 2i+j) have a cost of zero.
    # dat[2i+j] would have a cost equal to ni*nj as those would be the outer loops
    # TODO: Add penalty for non-affine layouts
    layout_cost = compute_indirection_cost(expr.layout, visited_axes, loop_axes, seen_exprs_mut, cache)
    dat_cost = extract_axes(expr.layout, visited_axes, loop_axes, cache=cache).size
    return dat_cost + layout_cost * INDIRECTION_PENALTY_FACTOR


@compute_indirection_cost.register(CompositeDat)
def _(dat: CompositeDat, /, visited_axes, loop_axes, seen_exprs_mut, cache) -> int:
    if seen_exprs_mut is not None:
        if dat in seen_exprs_mut:
            return 0
        else:
            seen_exprs_mut.add(dat)

    return extract_axes(dat.expr, visited_axes, loop_axes, cache).size


@functools.singledispatch
def concretize_materialized_tensor_indirections(obj: Any, /, *args, **kwargs) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@concretize_materialized_tensor_indirections.register(numbers.Number)
def _(num: numbers.Number, /, *args, **kwargs) -> numbers.Number:
    return num


@concretize_materialized_tensor_indirections.register(LinearDatBufferExpression)
def _(buffer_expr: LinearDatBufferExpression, layouts, key):
    layout = layouts[key + (buffer_expr,)]
    return LinearDatBufferExpression(buffer_expr.buffer, layout)


@concretize_materialized_tensor_indirections.register(NonlinearDatBufferExpression)
def _(buffer_expr: NonlinearDatBufferExpression, layouts, key):
    new_layouts = {
        leaf_path: layouts[key + ((buffer_expr, leaf_path),)]
        for leaf_path in buffer_expr.layouts.keys()
    }
    return NonlinearDatBufferExpression(buffer_expr.buffer, new_layouts)


@concretize_materialized_tensor_indirections.register(Mat)
# @_compress_array_indirection_maps.register(Sparsity)
def _(mat, layouts, outer_key) -> PetscMatBufferExpression:
    # If we have a temporary then things are easy and we do not need to substitute anything
    # (at least for now)
    if strictly_all(isinstance(ax, AxisTree) for ax in {mat.raxes, mat.caxes}):
        return PetscMatBufferExpression(mat.buffer, mat.raxes.layouts, mat.caxes.layouts, parent=mat.parent)

    def collect(axes, newlayouts, counter):
        for leaf_path in axes.leaf_paths:
            chosen_layout = layouts[outer_key + ((mat, leaf_path, counter),)]
            # try:
            #     chosen_layout = layouts[outer_key + ((mat, leaf_path, counter),)]
            # except KeyError:
            #     # zero-sized axis, no layout needed
            #     chosen_layout = -1
            newlayouts[leaf_path] = chosen_layout

    row_layouts = {}
    col_layouts = {}
    collect(mat.raxes.pruned, row_layouts, 0)
    collect(mat.caxes.pruned, col_layouts, 1)

    # will go away when we can assert using types
    assert mat.parent is None
    return PetscMatBufferExpression(mat.buffer, row_layouts, col_layouts)

# @functools.singledispatch
# def materialize(obj: Any, /, *args, **kwargs) -> ExpressionT:
#     raise TypeError
#
#
# @materialize.register(AxisVar)
# @materialize.register(LoopIndexVar)
# @materialize.register(numbers.Number)
# def _(var: Any, /, *args, **kwargs):
#     return var
#
#
# @materialize.register(Operator)
# def _(op: Operator, /, *args, **kwargs) -> Operator:
#     return type(op)(materialize(op.a, *args, **kwargs), materialize(op.b, *args, **kwargs))
#
#
# @materialize.register(CompositeDat)
# def _(dat: CompositeDat, /, visited_axes, loop_axes) -> _ExpressionDat:
#     axes = extract_axes(dat, visited_axes, loop_axes)
#
#     # dtype correct?
#     result = Dat(axes, dtype=IntType)
#
#     # replace LoopIndexVars in the expression with AxisVars
#     loop_index_replace_map = {}
#     for loop_id, iterset in loop_axes.items():
#         for axis in iterset.nodes:
#             loop_index_replace_map[(loop_id, axis.label)] = AxisVar(f"{axis.label}_{loop_id}")
#     expr = replace_terminals(dat.expr, loop_index_replace_map)
#
#     result.assign(expr, eager=True)
#
#     # now put the loop indices back
#     inv_map = {axis_var.axis_label: LoopIndexVar(loop_id, axis_label) for (loop_id, axis_label), axis_var in loop_index_replace_map.items()}
#     layout = just_one(result.axes.leaf_subst_layouts.values())
#     newlayout = replace_terminals(layout, inv_map)
#
#     return _ExpressionDat(result, newlayout)
@functools.singledispatch
def collect_axis_vars(obj: Any, /) -> OrderedSet:
    from pyop3.itree.tree import LoopIndexVar

    if isinstance(obj, Operator):
        return collect_axis_vars(obj.a) | collect_axis_vars(obj.b)

    raise TypeError(f"No handler defined for {type(obj).__name__}")


@collect_axis_vars.register(numbers.Number)
@collect_axis_vars.register(LoopIndexVar)
def _(var):
    return OrderedSet()

@collect_axis_vars.register(AxisVar)
def _(var):
    return OrderedSet([var])


@collect_axis_vars.register(Dat)
def _(dat: Dat, /) -> OrderedSet:
    loop_indices = OrderedSet()

    if dat.parent:
        loop_indices |= collect_axis_vars(dat.parent)

    for leaf in dat.axes.leaves:
        path = dat.axes.path(leaf)
        loop_indices |= collect_axis_vars(dat.axes.subst_layouts()[path])
    return loop_indices


@collect_axis_vars.register(LinearDatBufferExpression)
def _(dat: LinearDatBufferExpression, /) -> OrderedSet:
    return collect_axis_vars(dat.layout)


@collect_axis_vars.register(NonlinearDatBufferExpression)
def _(dat: NonlinearDatBufferExpression, /) -> OrderedSet:
    result = OrderedSet()
    for layout_expr in dat.layouts.values():
        result |= collect_axis_vars(layout_expr)
    return result
