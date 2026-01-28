import collections
import functools
import itertools
from typing import Any

import numpy as np
import pyop3 as op3
import finat
import ufl
from immutabledict import immutabledict as idict

from firedrake import utils
from firedrake.cofunction import Cofunction
from firedrake.function import CoordinatelessFunction, Function
from firedrake.functionspaceimpl import WithGeometry
from firedrake.matrix import Matrix
from firedrake.mesh import IterationSpec


@functools.singledispatch
def pack(tensor: Any, loop_info: IterationSpec, **kwargs) -> op3.Tensor:
    """Prepare a tensor for use inside a pyop3 loop."""
    raise TypeError(f"No handler defined for {utils.pretty_type(tensor)}")


@pack.register(Function)
@pack.register(Cofunction)
@pack.register(CoordinatelessFunction)
def _(func, loop_info: IterationSpec, **kwargs):
    return pack(func.dat, func.function_space(), loop_info, **kwargs)


@pack.register(Matrix)
def _(matrix: Matrix, loop_info, **kwargs):
    return pack(matrix.M, *matrix.ufl_function_spaces(), loop_info, **kwargs)


@pack.register(op3.Dat)
def _(
    dat: op3.Dat,
    space: WithGeometry,
    loop_info: IterationSpec,
    *,
    nodes: bool = False,
    permutation: collections.abc.Iterable | None = None,
):
    mesh = space.mesh()

    # This is tricky. Consider the case where you have a mixed space with hexes and
    # each space needs a different (non-permutation) transform. That means that we
    # have to generate code like:
    #
    # t0 = dat[:, closure(cell)]
    # t1 = transform0(t0[0])  # (field 0)
    # t2 = transform1(t0[1])  # (field 1)
    # t3[0] = t1
    # t3[1] = t2
    packed_dats = np.empty(len(space), dtype=object)
    for i, (index, subspace) in enumerate(iter_space(space)):
        packed_dats[i] = _pack_dat_nonmixed(dat[index], subspace, loop_info)

    if packed_dats.size == 1:
        return packed_dats.item()
    else:
        return op3.AggregateDat(packed_dats)


def _pack_dat_nonmixed(
    dat: op3.Dat,
    space: WithGeometry,
    loop_info: IterationSpec,
    *,
    nodes: bool = False,
    permutation: collections.abc.Iterable | None = None,
):
    mesh = space.mesh()

    if not nodes:
        map_ = space.entity_node_map(loop_info)
        cell_index = map_.index
        packed_dat = dat[map_]
        # bit of a hack, find the depth of the axis labelled 'closure', this relies
        # on the fact that the tree is always linear at the top
        if isinstance(packed_dat.axes, op3.AxisForest):  # bit of a hack
            axes = packed_dat.axes.trees[0]
        else:
            axes = packed_dat.axes
        depth = [axis.label for axis in axes.axes].index("closure")
    else:
        raise NotImplementedError

    return transform_packed_cell_closure_dat(packed_dat, space, cell_index, depth=depth, permutation=permutation)


@pack.register(op3.Mat)
def _(
    mat: op3.Mat,
    row_space: WithGeometry,
    column_space: WithGeometry,
    loop_info: IterationSpec,
    *,
    nodes: bool = False,
):
    if mat.buffer.mat_type == "python":
        mat_context = mat.buffer.mat.getPythonContext()
        if isinstance(mat_context, op3.RowDatPythonMatContext):
            space = row_space
        else:
            assert isinstance(mat_context, op3.ColumnDatPythonMatContext)
            space = column_space
        dat = mat_context.dat
        return pack(dat, space, loop_info, nodes=nodes)

    packed_mats = np.empty((len(row_space), len(column_space)), dtype=object)
    for ir, (row_index, row_subspace) in enumerate(iter_space(row_space)):
        for ic, (column_index, column_subspace) in enumerate(iter_space(column_space)):
            packed_mats[ir, ic] = _pack_mat_nonmixed(
                mat[row_index, column_index], row_subspace, column_subspace, loop_info
            )

    if packed_mats.size == 1:
        return packed_mats.item()
    else:
        return op3.AggregateMat(packed_mats)


def _pack_mat_nonmixed(
    mat: op3.Mat,
    row_space: WithGeometry,
    column_space: WithGeometry,
    loop_info: IterationSpec,
    *,
    nodes: bool = False,
):
    row_map = row_space.entity_node_map(loop_info)
    column_map = column_space.entity_node_map(loop_info)
    packed_mat = mat[row_map, column_map]

    depths = []
    for axes in [packed_mat.row_axes, packed_mat.column_axes]:
        if isinstance(axes, op3.AxisForest):  # bit of a hack
            axes = axes.trees[0]
        depth = [axis.label for axis in axes.axes].index("closure")
        depths.append(depth)
    row_depth, column_depth = depths

    return transform_packed_cell_closure_mat(
        packed_mat,
        row_space,
        column_space,
        row_map.index,
        column_map.index,
        row_depth=row_depth,
        column_depth=column_depth,
    )


def transform_packed_cell_closure_dat(
    packed_dat: op3.Dat,
    space,
    cell_index: op3.LoopIndex,
    *,
    depth: int = 0,
    permutation=None,
) -> op3.Dat:
    # Do this before the DoF transformations because this occurs at the level of entities, not nodes
    # TODO: In current Firedrake we apply this universally when 'entity_permutations' is
    # defined. This makes no sense for simplex and quad meshes because they are already
    # oriented. In effect we just arbitrarily permute the DoFs in the cell-node map for
    # no reason. This orientation work should really only be necessary for hexes but I'm
    # leaving as is for now because we otherwise get small inconsistencies between the
    # old and new 'cell_node_list's which I want to avoid.
    packed_dat = _orient_dofs(packed_dat, space, cell_index, depth=depth)

    # FIXME: This is awful!
    if _needs_static_permutation(space.finat_element) or permutation is not None:
        nodal_axis_tree, nodal_axis = _packed_nodal_axes(packed_dat.axes, space, depth)
        packed_dat = packed_dat.reshape(nodal_axis_tree)

        if _needs_static_permutation(space.finat_element):
            dof_perm_slice = _static_node_permutation_slice(nodal_axis, space, depth)
            packed_dat = packed_dat[dof_perm_slice]

        if permutation is not None:
            # needed because we relabel here... else the labels dont match
            nodal_axis = packed_dat.axes.axes[depth]
            perm_dat = op3.Dat(nodal_axis, data=permutation, prefix="perm", buffer_kwargs={"constant": True})
            perm_slice = op3.Slice(
                nodal_axis.label,
                [op3.Subset(None, perm_dat)],
            )
            packed_dat = packed_dat[perm_slice]

    return packed_dat


def transform_packed_cell_closure_mat(
    packed_mat: op3.Mat,
    row_space: WithGeometry,
    column_space: WithGeometry,
    row_cell_index: op3.Index,
    column_cell_index: op3.Index,
    *,
    row_depth: int = 0,
    column_depth: int = 0,
) -> op3.Mat:
    row_element = row_space.finat_element
    column_element = column_space.finat_element

    # Do this before the DoF transformations because this occurs at the level of entities, not nodes
    packed_mat = _orient_dofs(
        packed_mat,
        row_space,
        column_space,
        row_cell_index,
        column_cell_index,
        row_depth=row_depth,
        column_depth=column_depth,
    )

    if _needs_static_permutation(row_space.finat_element) or _needs_static_permutation(column_space.finat_element):
        rnodal_axis_tree, rnodal_axis = _packed_nodal_axes(packed_mat.row_axes, row_space, row_depth)
        cnodal_axis_tree, cnodal_axis = _packed_nodal_axes(packed_mat.column_axes, column_space, column_depth)
        packed_mat = packed_mat.reshape(rnodal_axis_tree, cnodal_axis_tree)

        row_dof_perm_slice = _static_node_permutation_slice(rnodal_axis, row_space, row_depth)
        column_dof_perm_slice = _static_node_permutation_slice(cnodal_axis, column_space, column_depth)
        packed_mat = packed_mat[row_dof_perm_slice, column_dof_perm_slice]

    return packed_mat


def _make_closure_map_tree(space: WithGeometry, loop_info: IterationSpec) -> op3.IndexTree:
    if len(space) == 1:
        return space.entity_node_map(loop_info)

    # mixed, need a closure per subspace and a full slice over the top
    # TODO: This is full slice, need nice API for that
    space_axis = space.plex_axes.root
    space_slice = op3.Slice(
        space_axis.name,
        [
            op3.AffineSliceComponent(space_index, label=space_index)
            for space_index in space_axis.component_labels
        ],
        label=space_axis.name,
    )
    index_tree = op3.IndexTree(space_slice)
    for leaf_path, subspace in zip(index_tree.leaf_paths, space, strict=True):
        index_tree = index_tree.add_subtree(
            leaf_path, _make_closure_map_tree(subspace, loop_info)
        )
    return index_tree


@functools.singledispatch
def _orient_dofs(packed_tensor: op3.Tensor, *args, **kwargs) -> op3.Tensor:
    raise TypeError(f"No handler defined for '{utils.pretty_type(packed_tensor)}'")


@_orient_dofs.register(op3.Dat)
def _(packed_dat: op3.Dat, space: WithGeometry, cell_index: op3.Index, *, depth: int) -> op3.Dat:
    """

    As an example, consider the edge DoFs of a Q3 function space in 2D. The
    DoFs have two possible permutations depending on the cell orientation.

    We realise this by taking the initial indexing:

        t0[i_edge, i_dof] = dat[map[i_cell, i_edge], i_dof]

    where 'i_cell' is the current cell (outer loop), 'i_edge' (<4) is the edge index,
    and 'i_dof' (<2) is the DoF index.

    To permute the DoFs we have to transform this expression to:

        t0[i_edge, i_dof] = dat[map[i_cell, i_edge], perm[ort[i_cell, i_edge], i_dof]]

    This can be achieved using indexing, but it is much easier to apply the
    transformation

        i_dof -> perm[ort[i_cell, i_edge], i_dof]

    """
    try:
        space.finat_element.entity_permutations  # noqa: F401
    except NotImplementedError:
        return packed_dat
    else:
        if space.mesh().dimension > 0:  # i.e. not a VoM
            permuted_axis_tree = _orient_axis_tree(packed_dat.axes, space, cell_index, depth=depth)
        else:
            permuted_axis_tree = packed_dat.axes
        return packed_dat.with_axes(permuted_axis_tree)


@_orient_dofs.register(op3.Mat)
def _(packed_mat: op3.Mat, row_space: WithGeometry, column_space: WithGeometry, row_cell_index: op3.Index, column_cell_index: op3.Index, *, row_depth: int, column_depth: int) -> op3.Mat:
    try:
        row_space.finat_element.entity_permutations  # noqa: F401
    except NotImplementedError:
        permuted_row_axes = packed_mat.row_axes
    else:
        if row_space.mesh().dimension > 0:  # i.e. not a VoM
            permuted_row_axes = _orient_axis_tree(packed_mat.row_axes, row_space, row_cell_index, depth=row_depth)
        else:
            permuted_row_axes = packed_mat.row_axes
    try:
        column_space.finat_element.entity_permutations  # noqa: F401
    except NotImplementedError:
        permuted_column_axes = packed_mat.column_axes
    else:
        if column_space.mesh().dimension > 0:  # i.e. not a VoM
            permuted_column_axes = _orient_axis_tree(packed_mat.column_axes, column_space, column_cell_index, depth=column_depth)
        else:
            permuted_column_axes  = packed_mat.column_axes
    return packed_mat.with_axes(permuted_row_axes, permuted_column_axes)


def _orient_axis_tree(axes, space: WithGeometry, cell_index: op3.Index, *, depth: int) -> op3.IndexedAxisTree:
    # discard nodal information
    if isinstance(axes, op3.AxisForest):
        axes = axes.trees[0]

    outer_axes = []
    outer_path = idict()
    for _ in range(depth):
        outer_axis = axes.node_map[outer_path]
        assert len(outer_axis.components) == 1
        outer_axes.append(outer_axis)
        outer_path = outer_path | {outer_axis.label: outer_axis.component.label}

    new_targets = {
        path: [list(targets) for targets in targetss]
        for (path, targetss) in axes.targets.items()
    }
    point_axis = axes.node_map[outer_path]
    for dim_axis_component in point_axis.components:
        dim_label = dim_axis_component.label

        dof_axis_label = f"dof{dim_label}"
        # dof_axis = utils.single_valued(axis for axis in space.plex_axes.axes if axis.label == dof_axis_label)
        dof_axis = utils.single_valued(axis for axis in axes.axes if axis.label == f"dof{dim_label}")
        if dof_axis.size == 0:
            continue

        # First create an buffer expression for the permutations that looks like:
        #
        #     'perm[i_which, i_dof]'
        # TODO: For some cases can avoid this permutation as it's just identity
        perm_expr = _entity_permutation_buffer_expr(space, dim_axis_component.label, axes)

        # Now replace 'i_which' with 'ort[i0, i1]'
        orientation_expr = op3.as_linear_buffer_expression(space.mesh().entity_orientations_dat[cell_index][(slice(None),)*depth+(op3.as_slice(dim_label),)])
        selector_axis_var = utils.just_one(axis_var for axis_var in op3.collect_axis_vars(perm_expr) if axis_var.axis.label == "which")
        perm_expr = op3.replace(perm_expr, {selector_axis_var: orientation_expr}, assert_modified=True)

        # This gives us the expression 'perm[ort[i0, i1], i2]' that we can
        # now plug into 'packed_dat'

        path = outer_path | idict({point_axis.label: dim_axis_component.label}) | {dof_axis_label: "XXX"}
        before = utils.just_one(new_targets[path][0])  # hack to get the right one...
        assert before.axis == "dof"
        new_targets[path] = [[before.__record_init__(
            # expr=op3.replace_terminals(before.expr, {dof_axis.label: perm_expr}, assert_modified=True)
            expr=op3.replace_terminals(before.expr, {dof_axis.label: perm_expr})
        )]]

    new_targets = utils.freeze(new_targets)

    return axes.__record_init__(_targets=new_targets)


# @op3.cache.serial_cache(hashkey=lambda space, dim: (space.finat_element, dim))
def _entity_permutation_buffer_expr(space: WithGeometry, dim_label, axes) -> tuple[op3.LinearDatBufferExpression, ...]:
    perms = _prepare_entity_permutations(space.finat_element, dim_label)
    perms_array = np.concatenate(perms)
    perms_buffer = op3.ArrayBuffer(perms_array, constant=True, rank_equal=True)

    # Create an buffer expression for the permutations that looks like: 'perm[i_which, i_dof]'
    perm_selector_axis = op3.Axis(len(perms), "which")
    dof_axis = utils.single_valued(axis for axis in axes.axes if axis.label == f"dof{dim_label}")
    perm_dat_axis_tree = op3.AxisTree.from_iterable([perm_selector_axis, dof_axis])
    perm_dat = op3.Dat(perm_dat_axis_tree, buffer=perms_buffer, prefix="perm")
    return op3.as_linear_buffer_expression(perm_dat)


@op3.cache.serial_cache()
def _prepare_entity_permutations(element, dim_label):
    if not isinstance(element, finat.TensorProductElement):
        myvar = element.entity_permutations[dim_label]
        return list(utils.single_valued(myvar.values()).values())

    finat_element = element
    base_dim_label = dim_label
    nrepeats = 1
    while isinstance(finat_element, finat.TensorProductElement):
        finat_element, interval_element = finat_element.factors
        base_dim_label, vert_or_edge = base_dim_label[:-1], base_dim_label[-1]

        if vert_or_edge == 1:
            # the extruded edge, can have repeats (not so for vertices)
            ndofs_on_edge = len(interval_element.entity_dofs()[1][0])
            nrepeats *= ndofs_on_edge
    base_dim_label = utils.just_one(base_dim_label)
    perms = utils.single_valued(finat_element.entity_permutations[base_dim_label].values())

    # turn something like [0, 1], [1, 0] into [0, 1, 2, 3, 4, 5], [3, 4, 5, 0, 1, 2]
    new_perms = []
    for perm in map(np.asarray, perms.values()):
        new_perm = []
        for p in perm:
            for i in range(nrepeats):
                new_perm.append(p*nrepeats+i)
        new_perms.append(new_perm)

    return new_perms



@op3.cache.serial_cache()
def _flatten_entity_dofs(element) -> np.ndarray:
    """Flatten FInAT element ``entity_dofs`` into an array."""
    entity_dofs = element.entity_dofs()

    # now flatten
    flat_entity_dofs = []
    for dim in sorted(entity_dofs.keys()):
        num_entities = len(entity_dofs[dim])
        for entity_num in range(num_entities):
            dofs = entity_dofs[dim][entity_num]
            flat_entity_dofs.extend(dofs)
    flat_entity_dofs = np.asarray(flat_entity_dofs, dtype=utils.IntType)
    assert utils.has_unique_entries(flat_entity_dofs)
    return utils.readonly(flat_entity_dofs)


def _static_node_permutation_slice(nodal_axis, space: WithGeometry, depth) -> tuple[op3.AxisTree, tuple]:
    permutation = _node_permutation_from_element(space.finat_element)
    dof_perm_dat = op3.Dat(nodal_axis, data=permutation, prefix="perm", buffer_kwargs={"constant": True})
    dof_perm_slice = op3.Slice(
        nodal_axis.label,
        [op3.Subset(None, dof_perm_dat)],
    )
    return (*[slice(None)]*depth, dof_perm_slice)


def _packed_nodal_axes(packed_axes: op3.AxisTree, space, depth):
    # involved way to get num_nodes
    permutation = _node_permutation_from_element(space.finat_element)

    # TODO: Could be 'AxisTree.linear_to_depth()' or similar
    outer_axes = []
    outer_path = idict()
    for _ in range(depth):
        outer_axis = packed_axes.node_map[outer_path]
        assert len(outer_axis.components) == 1
        outer_axes.append(outer_axis)
        outer_path = outer_path | {outer_axis.label: outer_axis.component.label}

    nodal_axis = op3.Axis(permutation.size)
    nodal_axis_tree = op3.AxisTree.from_iterable([*outer_axes, nodal_axis, *space.shape])
    return nodal_axis_tree, nodal_axis


@op3.cache.serial_cache()
def _node_permutation_from_element(element) -> np.ndarray:
    return utils.readonly(utils.invert(_flatten_entity_dofs(element)))


@op3.cache.serial_cache()
def _needs_static_permutation(element) -> bool:
    perm = _node_permutation_from_element(element)
    return any(perm != np.arange(perm.size, dtype=perm.dtype))


def _requires_orientation(space: WithGeometry) -> bool:
    return space.finat_element.fiat_equivalent.dual.entity_permutations is not None


def iter_space(space: WithGeometry):
    """Index-friendly iterator for function spaces."""
    if len(space) == 1:
        yield (Ellipsis, space)
    else:
        yield from enumerate(space)
