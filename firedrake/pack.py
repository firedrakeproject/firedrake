import collections
import functools
import warnings
from typing import Any

import numpy as np
import pyop3 as op3
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

    if len(space) > 1:
        # do a loop
        raise NotImplementedError
        # This is tricky. Consider the case where you have a mixed space with hexes and
        # each space needs a different (non-permutation) transform. That means that we
        # have to generate code like:
        #
        # t0 = dat[:, closure(cell)]
        # t1 = transform0(t0[0])  # (field 0)
        # t2 = transform1(t0[1])  # (field 1)
        # t3[0] = t1
        # t3[1] = t2
        #
        # I think that the easiest approach here is to index the full thing before passing it
        # down. We can then combine everything at the top-level

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
    # NOTE: This is now a special case of the fuse stuff below
    # if space.mesh().ufl_cell() == ufl.hexahedron:
    #     packed_dat = _orient_dofs(packed_dat, space, cell_index, depth=depth)

    transform_in_kernel, transform_out_kernel = fuse_orientations([space])

    if packed_dat.dtype == utils.IntType:
        warnings.warn("Int Type dats cannot be transformed using fuse transforms")
    elif transform_in_kernel and transform_out_kernel:
        orientations = space.mesh().entity_orientations_dat

        mat_work_array = op3.Dat.null(op3.AxisTree.from_iterable([packed_dat.size, packed_dat.size]), dtype=utils.ScalarType, prefix="trans")
        vec_work_array = op3.Dat.null(op3.AxisTree.from_iterable([packed_dat.size]), dtype=utils.ScalarType, prefix="trans")

        def transform_in(untransformed, transformed):
            return (
                transform_in_kernel(orientations[cell_index], mat_work_array, untransformed, transformed),
            )

        def transform_out(transformed, untransformed):
            return (
                transform_out_kernel(orientations[cell_index], mat_work_array, transformed, untransformed),
            )

        transform = op3.OutOfPlaceCallableTensorTransform(transform_in, transform_out, packed_dat.transform)
        packed_dat = packed_dat.__record_init__(transform=transform)

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

    transform_in_kernel, transform_out_kernel = fuse_orientations([row_space, column_space])
    if packed_mat.dtype == utils.IntType:
        warnings.warn("Int Type mats cannot be transformed using fuse transforms")
    elif transform_in_kernel and transform_out_kernel:
        orientations = row_space.mesh().entity_orientations_dat
        orientations_c = column_space.mesh().entity_orientations_dat

        mat_work_array_row = op3.Dat.null(op3.AxisTree.from_iterable([packed_mat.nrows, packed_mat.nrows]), dtype=utils.ScalarType, prefix="trans")
        mat_work_array_col = op3.Dat.null(op3.AxisTree.from_iterable([packed_mat.ncols, packed_mat.ncols]), dtype=utils.ScalarType, prefix="trans")

        def transform_in(untransformed, transformed):
            return (
                transform_in_kernel(orientations[row_cell_index], mat_work_array_row, mat_work_array_col, untransformed, transformed),
            )

        def transform_out(transformed, untransformed):
            return (
                transform_out_kernel(orientations[row_cell_index], mat_work_array_row, mat_work_array_col, transformed, untransformed),
            )

        transform = op3.OutOfPlaceCallableTensorTransform(transform_in, transform_out, packed_mat.transform)
        packed_mat = packed_mat.__record_init__(transform=transform)


    # Do this before the DoF transformations because this occurs at the level of entities, not nodes
    if utils.strictly_all(
        space.mesh().ufl_cell() == ufl.hexahedron for space in [row_space, column_space]
    ):
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
        row_nodal_axis_tree, row_dof_perm_slice = _static_node_permutation_slice(packed_mat.row_axes, row_space, row_depth)
        column_nodal_axis_tree, column_dof_perm_slice = _static_node_permutation_slice(packed_mat.column_axes, column_space, column_depth)
        packed_mat = packed_mat.reshape(row_nodal_axis_tree, column_nodal_axis_tree)[row_dof_perm_slice, column_dof_perm_slice]

    return packed_mat


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
    permuted_axis_tree = _orient_axis_tree(packed_dat.axes, space, cell_index, depth=depth)
    return packed_dat.with_axes(permuted_axis_tree)


@_orient_dofs.register(op3.Mat)
def _(packed_mat: op3.Mat, row_space: WithGeometry, column_space: WithGeometry, row_cell_index: op3.Index, column_cell_index: op3.Index, *, row_depth: int, column_depth: int) -> op3.Mat:
    permuted_row_axes = _orient_axis_tree(packed_mat.row_axes, row_space, row_cell_index, depth=row_depth)
    permuted_column_axes = _orient_axis_tree(packed_mat.column_axes, column_space, column_cell_index, depth=column_depth)
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
        k: (path, dict(exprs))
        for k, (path, exprs) in axes.targets[0].items()
    }
    point_axis = axes.node_map[outer_path]
    for dim_axis_component in point_axis.components:
        dim_label = dim_axis_component.label

        dof_axis_label = f"dof{dim_label}"
        dof_axis = utils.single_valued(axis for axis in space.plex_axes.axes if axis.label == dof_axis_label)
        if dof_axis.size == 0:
            continue

        # First create an buffer expression for the permutations that looks like:
        #
        #     'perm[i_which, i_dof]'
        # TODO: For some cases can avoid this permutation as it's just identity
        perm_expr = _entity_permutation_buffer_expr(space, dim_axis_component.label)

        # Now replace 'i_which' with 'ort[i0, i1]'
        orientation_expr = op3.as_linear_buffer_expression(space.mesh().entity_orientations_dat[cell_index][(slice(None),)*depth+(dim_label,)])
        selector_axis_var = utils.just_one(axis_var for axis_var in op3.collect_axis_vars(perm_expr) if axis_var.axis_label == "which")
        perm_expr = op3.replace(perm_expr, {selector_axis_var: orientation_expr})

        # This gives us the expression 'perm[ort[i0, i1], i2]' that we can
        # now plug into 'packed_dat'

        path = outer_path | idict({point_axis.label: dim_axis_component.label}) | {dof_axis_label: "XXX"}
        before = new_targets[path][1]["dof"]
        new_targets[path][1]["dof"] = op3.replace(new_targets[path][1]["dof"], {op3.AxisVar(dof_axis): perm_expr})
        assert new_targets[path][1]["dof"] != before

    # TODO: respect the fact that targets can contain multiple entries
    return axes.__record_init__(targets=(new_targets,))


@op3.cache.serial_cache(hashkey=lambda space, dim: (space.finat_element, dim))
def _entity_permutation_buffer_expr(space: WithGeometry, dim_label) -> tuple[op3.LinearDatBufferExpression, ...]:
    perms = utils.single_valued(space.finat_element.entity_permutations[dim_label].values())
    # TODO: can optimise the dtype here to be as small as possible
    perms_array = np.concatenate(list(perms.values()))
    perms_buffer = op3.ArrayBuffer(perms_array, constant=True, rank_equal=True)

    # Create an buffer expression for the permutations that looks like: 'perm[i_which, i_dof]'
    perm_selector_axis = op3.Axis(len(perms), "which")
    dof_axis = utils.single_valued(axis for axis in space.plex_axes.axes if axis.label == f"dof{dim_label}")
    perm_dat_axis_tree = op3.AxisTree.from_iterable([perm_selector_axis, dof_axis])
    perm_dat = op3.Dat(perm_dat_axis_tree, buffer=perms_buffer, prefix="perm")
    return op3.as_linear_buffer_expression(perm_dat)


@op3.cache.serial_cache()
def _flatten_entity_dofs(element) -> np.ndarray:
    """Flatten FInAT element ``entity_dofs`` into an array."""
    entity_dofs = element.entity_dofs()
    flat_entity_dofs = []
    for dim in sorted(entity_dofs.keys()):
        num_entities = len(entity_dofs[dim])
        for entity_num in range(num_entities):
            dofs = entity_dofs[dim][entity_num]
            flat_entity_dofs.extend(dofs)
    flat_entity_dofs = np.asarray(flat_entity_dofs, dtype=utils.IntType)
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


def construct_switch_statement(self, mats: dict, n: int, idx: int, args: list, var_list: list[str]) -> str:
    string = []
    string += f"a{idx} = iden; \n "
    string += "\nswitch (dim) { \n"

    var_list += ["iden"]
    args += [loopy.TemporaryVariable("iden", initializer=np.identity(n), dtype=utils.ScalarType, read_only=True, address_space=loopy.AddressSpace(1))]

    closure_sizes = self._mesh._closure_sizes[self._mesh.dimension]
    closure_size_acc = 0
    indent = 0
    for dim in range(len(closure_sizes)):
        string += f"case {dim}:\n "
        indent += 1
        string += indent*"\t" + "o_val = o[i + closure_size_acc]; \n "
        #string += indent*"\t" + "o_val = 0; \n "
        string += [indent*"\t" + "switch (i) { \n"]
        indent += 1
        for i in range(closure_sizes[dim]):
            string += indent*"\t" + f"case {i}:\n "
            indent += 1
            string += indent*"\t" + "switch (o_val) { \n"
            indent += 1
            for val in sorted(mats[dim][i].keys()):
                string += indent*"\t" + f"case {val}:\n "
                indent += 1
                matname = f"mat{dim}_{i}_{val}"
                string += indent*"\t" + f"a{idx} = {matname};\n"
                string += indent*"\t" + "break;\n"
                var_list += [matname]
                mat = np.array(mats[dim][i][val], dtype=utils.ScalarType)
                args += [loopy.TemporaryVariable(matname, initializer=mat, dtype=utils.ScalarType, read_only=True, address_space=loopy.AddressSpace(1))]
                indent -= 1
            indent -= 1
            string += indent*"\t" + "default: break;}break;\n"
            indent -= 1
        string += indent*"\t" + "default: break; }\n"
        closure_size_acc += dim
        # string += indent*"\t" + "printf(\"res (before) %f, %f, %f\\n\", res[0], res[1], res[2]); \n"
        # string += indent*"\t" + "printf(\"b (before) %f, %f, %f\\n\", b[0], b[1], b[2]); \n"
        #string += indent*"\t" + "printf(\"d: %d, dim: %d, i: %d, o_val: %d \\n\", d, dim, i, o_val); \n"
        #for i in range(n):
        #    string += indent*"\t" + f"printf(\"a row {i}: {" ".join('%f' for i in range(n))}\\n\", {", ".join(f"a[{i*n + j}]" for j in range(n))}); \n"
        #string += indent*"\t" + f"printf(\"a row 1: {" ".join('%f' for i in range(n))}\\n\", {", ".join(f"a[{n + i}]" for i in range(n))}); \n"
        #string += indent*"\t" + f"printf(\"a row 2: {" ".join('%f' for i in range(n))}\\n\", {", ".join(f"a[{2*n + i}]" for i in range(n))}); \n"
        #string += indent*"\t" + "printf(\"a rows ...\\n\"); \n"
        # string += indent*"\t" + "printf(\"\\n\");\n"
        string += indent*"\t" + "\n"
        string += indent*"\t" + "break;\n"
        indent -= 1

    string += "default: break; }\n"
    return string, args, var_list

def get_utility_kernels(ns: tuple[int]) -> tuple:
    if len(ns) == 1:
        row_idx = "j"
        col_idx = ""
        iter_idx = "i"
        all_elems = ":"
        all_idxs = f"{{[{row_idx}]:0 <= j < {ns[0]}}}",
    elif len(ns) == 2:
        row_idx = "i"
        col_idx = "j"
        iter_idx = "k"
        all_elems = ":,:"
        all_idxs = f"{{[i,j]:0 <= i < {ns[0]} and 0 <= j < {ns[1]}}}",
    else:
        raise NotImplementedError("Fuse orientations cannot handle tensors")
    a_idx = ",".join([i for i in [row_idx, iter_idx] if i != ""])
    res_idx = ",".join([i for i in [row_idx, col_idx] if i != ""])
    b_idx = ",".join([i for i in [iter_idx, col_idx] if i != ""])
    all_idx = ",".join([i for i in [row_idx, iter_idx, col_idx] if i != ""])
    matmuls = []
    if len(ns) == 1:
        # computes res = Ab
        matmuls += [loopy.make_function(
          f"{{[{all_idx}]:0 <= {all_idx} < {ns[0]}}}",
          f"""
              res[{res_idx}] =  res[{res_idx}] + a[{a_idx}]*b[{b_idx}]
          """, name=f"matmul0", target=loopy.CWithGNULibcTarget())]
    else:
        # computes res = A^T B
        matmuls += [loopy.make_function(
          f"{{[i,j,k]:0 <= i,k < {ns[0]} and 0 <= j < {ns[1]}}}",
          f"""
              res[i,j] =  res[i,j] + a[k,i]*b[k,j]
          """, name=f"matmul0", target=loopy.CWithGNULibcTarget())]
        # computes res = BA
        matmuls += [loopy.make_function(
          f"{{[i,j,k]:0 <= i < {ns[0]} and 0 <= j,k < {ns[1]}}}",
          f"""
              res[i,j] =  res[i,j] + b[i,k]*a[k,j]
          """, name=f"matmul1", target=loopy.CWithGNULibcTarget())]
        #for n1, n2, idx1, idx2 in zip(ns, reversed(ns), [a_idx, b_idx], [col_idx, row_idx]):
       #  
       #     matmuls += [loopy.make_function(
       #     f"{{[{idx1+","+idx2}]:0 <= {idx1} < {n1} and 0 <= {idx2} < {n2}}}",
       #     f"""
       #        res[{res_idx}] =  res[{res_idx}] + a[{idx2 + "," + iter_idx}]*b[{iter_idx +}]
       #     """, name="matmul", target=loopy.CWithGNULibcTarget())]

    inc_args = [loopy.GlobalArg("b", dtype=utils.ScalarType, shape=ns, is_input=True, is_output=True),
                loopy.GlobalArg("res", dtype=utils.ScalarType, shape=ns, is_input=True)]
    inc_knl = loopy.make_function(
        all_idxs,
        [f"b[{res_idx}] = b[{res_idx}] + res[{res_idx}]"],
        kernel_data=inc_args,
        name="inc", target=loopy.CWithGNULibcTarget()
    )
    set_args = [loopy.GlobalArg("b", dtype=utils.ScalarType, shape=ns, is_input=True, is_output=True),
                loopy.GlobalArg("res", dtype=utils.ScalarType, shape=ns, is_input=True)]
    set_knl = loopy.make_function(
        all_idxs,
        [f"b[{res_idx}] = res[{res_idx}]"],
        kernel_data=set_args,
        name="set", target=loopy.CWithGNULibcTarget()
    )
    zero_knl = loopy.make_function(
        all_idxs,
        [f"res[{res_idx}] = 0"],
        [loopy.GlobalArg("res", shape=ns, dtype=int, is_input=True, is_output=True)],
        target=loopy.CWithGNULibcTarget(),
        name="zero",
    )
        #     child_knls[0] = loopy.make_function(
        #         f"{{[i, j, k]:0<=i, k < {shapes[0]} and 0<= j < {shapes[1]}}}",
        #         """
        #             res[k, j] =  res[k, j] + a[k, i]*b[i, j]
        #         """, name=f"matmul{0}", target=loopy.CWithGNULibcTarget())
        #     child_knls[1] = loopy.make_function(
        #         f"{{[i, j, k]:0<=k < {shapes[0]} and 0<=  j, i < {shapes[1]}}}",
        #         """
        #             res[k, j] =  res[k, j] + a[k, i]*b[i, j]
        #         """, name=f"matmul{1}", target=loopy.CWithGNULibcTarget())
    return matmuls + [set_knl, zero_knl, inc_knl], all_elems


def fuse_orientations(spaces: list[WithGeometry]):
    fuse_defined_spaces = [hasattr(space.ufl_element(), "triple") for space in spaces]
    fuse_matrix_spaces = [hasattr(spaces[i].ufl_element().triple, "matrices") if fuse_defined_spaces[i] else False for i in range(len(spaces))]
    fuse_needs_matrices = [spaces[i].ufl_element().triple.apply_matrices if fuse_defined_spaces[i] else False for i in range(len(spaces))]
    
    if not all(fuse_defined_spaces):
        return None, None

    # TODO case where only one has matrices? or do all fuse elements have them, check
    if all(fuse_defined_spaces) and all(fuse_matrix_spaces) and any(fuse_needs_matrices):
        print("NEW FUSE")
        mesh = spaces[0]._mesh
        mats = []
        reversed_mats = []
        ns = tuple()
        for space in spaces:
            fs = space
            mats += [space.ufl_element().triple.matrices]
            reversed_mats += [space.ufl_element().triple.reversed_matrices]
            t_dim = space.ufl_element().cell._tdim
            os = mats[-1][t_dim][0]
            ns += (os[next(iter(os.keys()))].shape[0],)
        closures_dict = mesh._closure_sizes[fs._mesh.dimension]
        closures = [closures_dict[c] for c in sorted(closures_dict.keys())]

        utilities, all_elems = get_utility_kernels(ns) 
        args = [loopy.ValueArg("d", dtype=utils.IntType),
                loopy.ValueArg("closure_size_acc", dtype=utils.IntType),
                loopy.ValueArg("o_val", dtype=utils.IntType),
                loopy.GlobalArg("o", dtype=utils.IntType, shape=(sum(closures)), is_input=True)] + [loopy.GlobalArg(f"a{i}", dtype=utils.ScalarType, shape=(ns[i], ns[i]), is_input=True, is_output=False) for i in range(len(ns))] + [loopy.GlobalArg("b", dtype=utils.ScalarType, shape=ns, is_input=True, is_output=True),
                          loopy.GlobalArg("res", dtype=utils.ScalarType, shape=ns, is_input=True, is_output=True)]

        a_list = ",".join([f"a{i}[:,:]" for i in range(len(ns))])
        var_list = ["o", "d", "i", "o_val", "dim"]

        def switch(space, mats, n, i, args, var_list, all_elems, name, reverse=False):
            dim_arg = [loopy.ValueArg("dim", dtype=utils.IntType)]
            switch_string, args, var_list = construct_switch_statement(fs, mats, n, i, args, var_list)
            transform_insn = loopy.CInstruction(tuple(), "".join(switch_string), assignees=(f"a{i}", "o_val"), read_variables=frozenset(var_list), id="assign", depends_on="zero")
            #arg_order = f"a{i}, b" if not reverse else f"b, a"
            matmul_insn = f"res[{all_elems}] = matmul{i}(a{i}, b, res) {{id=matmul, dep=*, dep=assign}}"
            print_insn = loopy.CInstruction(tuple(),
                         "printf(\"res: %f, %f, %f\\n\", res[0], res[1], res[2]);",
                          assignees=(), read_variables=frozenset([]), within_inames=frozenset(["i"]), id="print", depends_on="matmul")
            return loopy.make_function(
                "{[i]:0<= i < d}",
                [f"res[{all_elems}] = zero(res) {{id=zero, inames=i}}",
                 transform_insn, matmul_insn,
                 f"b[{all_elems}] = set(b[{all_elems}], res[{all_elems}]) {{id=set, dep=matmul, inames=i}}"
                 ],
                name=name + "_switch_on_o",
                kernel_data=dim_arg + args,
                target=loopy.CWithGNULibcTarget())
        in_switches = [switch(spaces[i], mats[i], ns[i],i, args, var_list, all_elems, name="in"+str(i)) for i in range(len(spaces))]
        out_switches = [switch(spaces[i], reversed_mats[i], ns[i], i, args, var_list, all_elems, name="out"+str(i)) for i in range(len(spaces))]

        closure_arg = [loopy.TemporaryVariable("closure_sizes", initializer=np.array(closures, dtype=np.int32), dtype=utils.IntType, read_only=True, address_space=loopy.AddressSpace(1))]
        printres_insn = loopy.CInstruction(tuple(), "printf(\"replaces res: %f\\n\", res[0]);", assignees=(), read_variables=frozenset(["res"]), depends_on="replace")

        def loop_dims(direction, all_elems):
            num_switch = len(all_elems.split(","))
            labelling = [ f"{{id=switch{chr(i+65)} " + (",dep=*}" if i == 0 else f",dep=switch{chr(i+64)},dep=*}}") for i in range(num_switch)]
            switches = [f"""
                         b[{all_elems}], res[{all_elems}] = {direction + str(i)}_switch_on_o(dim, d, closure_size_acc, o_val, o[:], {a_list}, b[{all_elems}], res[{all_elems}]) {labelling[i]}"""
                        for i in range(num_switch)]
            return loopy.make_function(
                f"{{[dim]:{0} <= dim <= {mesh.dimension - 1}}}",
                ["d = closure_sizes[dim] {id=closure}"] + switches +
                [f"closure_size_acc = closure_size_acc + d {{id=replace, dep=switch{chr(65 + num_switch-1)}}}"],
                name=f"{direction}_loop_over_dims",
                kernel_data=closure_arg + args,
                target=loopy.CWithGNULibcTarget())

        print_insn = loopy.CInstruction(tuple(),
                     f"""printf(\"initial b: {" ".join('%f' for i in range(ns[0]))}\\n\", {', '.join(f"b[{j}]" for j in range(ns[0]))});
                         printf(\"initial res: {" ".join('%f' for i in range(ns[0]))}\\n\", {', '.join(f"res[{j}]" for j in range(ns[0]))});
                         printf(\"initial temp: {" ".join('%f' for i in range(ns[0]))}\\n\", {', '.join(f"temp[{j}]" for j in range(ns[0]))});
                         printf(\"o: {" ".join('%d' for i in range(sum(closures)))}\\n\", {', '.join(f"o[{j}]" for j in range(sum(closures)))});""", assignees=(), read_variables=frozenset([]), id="print")
        # ---- VERSION 1 ----
        # no manual increment - overwrites values
        print_insn1 = loopy.CInstruction(tuple(),
                      f"""printf(\"final res: {" ".join('%f' for i in range(ns[0]))}\\n\", {', '.join(f"res[{j}]" for j in range(ns[0]))});
                          printf(\"final temp: {" ".join('%f' for i in range(ns[0]))}\\n\", {', '.join(f"temp[{j}]" for j in range(ns[0]))});"""
                      , assignees=(), read_variables=frozenset(["res"]), depends_on="replace")

        
        def overall(direction, all_elems):
            tempArg = [loopy.TemporaryVariable("temp", initializer=np.zeros(ns), dtype=utils.ScalarType, read_only=True, address_space=loopy.AddressSpace(1))]
            return loopy.make_kernel(
            "{:}",
            [print_insn, 
             f"temp[{all_elems}] = set(temp[{all_elems}], res[{all_elems}]) {{id=store, dep=print}}",
            f"b[{all_elems}], res[{all_elems}] = {direction}_loop_over_dims(0,0,0,o[:], {a_list}, b[{all_elems}], res[{all_elems}]) {{dep=store, id=loop}}",
             f"res[{all_elems}] = set(res[{all_elems}], b[{all_elems}]) {{id=replace, dep=loop}}",
             f"b[{all_elems}] = zero(b[{all_elems}]) {{dep=replace, id=zerob}}",
             f"temp[{all_elems}] = zero(temp[{all_elems}]) {{dep=replace, id=zero_temp}}",
             print_insn1],
            name=f"{direction}_transform",
            kernel_data=args[3:] + tempArg,
            target=loopy.CWithGNULibcTarget())
        # ---- VERSION 2 ----
        # manual increment - gets wrong numbers
        #print_insn1 = loopy.CInstruction(tuple(),
        #              f"""printf(\"final res: {" ".join('%f' for i in range(ns[0]))}\\n\", {', '.join(f"res[{j}]" for j in range(ns[0]))});
        #                  printf(\"final temp: {" ".join('%f' for i in range(ns[0]))}\\n\", {', '.join(f"temp[{j}]" for j in range(ns[0]))});"""
        #              , assignees=(), read_variables=frozenset(["res"]), depends_on="replace")

        #
        #def overall(direction, all_elems):
        #    #tempArg = [loopy.GlobalArg("temp", dtype=utils.ScalarType, shape=ns, is_input=True, is_output=False)]
        #    tempArg = [loopy.TemporaryVariable("temp", initializer=np.zeros(ns), dtype=utils.ScalarType, read_only=True, address_space=loopy.AddressSpace(1))]
        #    return loopy.make_kernel(
        #    "{:}",
        #    [print_insn, 
        #     f"temp[{all_elems}] = set(temp[{all_elems}], res[{all_elems}]) {{id=store, dep=print}}",
        #    f"b[{all_elems}], res[{all_elems}] = {direction}_loop_over_dims(0,0,0,o[:], {a_list}, b[{all_elems}], res[{all_elems}]) {{dep=store, id=loop}}",
        #     f"b[{all_elems}] = inc(b[{all_elems}], temp[{all_elems}]) {{id=inc, dep=loop}}",
        #     f"res[{all_elems}] = set(res[{all_elems}], b[{all_elems}]) {{id=replace, dep=inc}}",
        #     f"b[{all_elems}] = zero(b[{all_elems}]) {{dep=replace, id=zerob}}",
        #     f"temp[{all_elems}] = zero(temp[{all_elems}]) {{dep=replace, id=zero_temp}}",
        #     print_insn1],
        #    name=f"{direction}_transform",
        #    kernel_data=args[3:] + tempArg,
        #    target=loopy.CWithGNULibcTarget())
        # ----------
        in_knl = loopy.merge([overall("in", all_elems), loop_dims("in", all_elems)] + in_switches + utilities)
        out_knl = loopy.merge([overall("out", all_elems), loop_dims("out", all_elems)] + out_switches + utilities)
        #print(in_knl) 
        #breakpoint()

        # b is modified in the transform functions but the result is written to res and therefore is not needed further.
        transform_in = op3.Function(in_knl, [op3.READ] + [op3.WRITE for n in ns] + [op3.READ, op3.RW])

        transform_out = op3.Function(out_knl, [op3.READ] + [op3.WRITE for n in ns] + [op3.READ, op3.RW])
        return transform_in, transform_out
    elif fuse_defined_spaces and sum(fuse_matrix_spaces) == 0:
        return None, None
    elif fuse_defined_spaces and sum(fuse_defined_spaces) != sum(fuse_matrix_spaces):
        raise ValueError("If a fuse space is used, all spaces must be fuse spaces")
    else:
        return None, None
