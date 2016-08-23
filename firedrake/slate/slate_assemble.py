"""This module contains an assembly algorithm for creating
usable data objects from compiled SLATE expressions returned
from the HFC.

This module is primarily use for testing.

Written by: Thomas Gibson (t.gibson15@imperial.ac.uk)
"""

import firedrake

from hfc import compile_slate_expression
from slate_assertions import expecting_slate_expr
from slate import *


__all__ = ['slate_assemble']


def slate_assemble(expr, bcs=None, nest=False):
    """Assemble the SLATE expression `expr` and return a Firedrake object
    representing the result. This will be a :class:`float` for rank-0
    tensors, a :class:`.Function` for rank-1 tensors and a :class:`.Matrix`
    for rank-2 tensors. The result will be returned as a `tensor` of
    :class:`firedrake.Function` for rank-0 and rank-1 SLATE expressions and
    :class:`firedrake.matrix.Matrix` for rank-2 SLATE expressions.

    :arg expr: A SLATE object to assemble.
    :arg bcs: A tuple of :class:`.DirichletBC`\s to be applied.
    """

    if not isinstance(expr, Tensor):
        expecting_slate_expr(expr)

    arguments = expr.arguments()
    rank = len(arguments)

    # If the expression is a rank-2 tensor: matrix
    if rank == 2:
        test_function, trial_function = arguments
        fs_names = (test_function.function_space().name,
                    trial_function.function_space().name)
        maps = tuple((test_function.cell_node_map(), trial_function.cell_node_map()))
        sparsity = firedrake.op2.Sparsity((test_function.function_space().dof_dset,
                                           trial_function.function_space().dof_dset),
                                          maps,
                                          "%s_%s_sparsity" % fs_names,
                                          nest=nest)
        tensor = firedrake.matrix.Matrix(expr, bcs, sparsity)
        tensor_arg = tensor._M(firedrake.op2.INC, (test_function.cell_node_map(bcs)[firedrake.op2.i[0]], trial_function.cell_node_map(bcs)[firedrake.op2.i[0]]), flatten=True)

    # If the expression is a rank-1 tensor: vector
    elif rank == 1:
        test_function = arguments[0]
        tensor = firedrake.Function(test_function.function_space())
        tensor_arg = tensor.dat(firedrake.op2.INC, test_function.cell_node_map()[firedrake.op2.i[0]],
                                flatten=True)

    # if the expression is a rank-0 tensor: scalar
    elif rank == 0:
        tensor = firedrake.op2.Global(1, [0.0])
        tensor_arg = tensor(firedrake.op2.INC)
    else:
        raise NotImplementedError("Not implemented for rank-%d tensors.", rank)

    # Retrieve information from the slate form compiler
    coords, coefficients, need_cell_facets, kernel = compile_slate_expression(expr, testing=True)
    mesh = coords.function_space().mesh()
    args = [kernel, mesh.cell_set, tensor_arg, coords.dat(firedrake.op2.READ,
                                                          coords.cell_node_map(),
                                                          flatten=True)]
    # Append the coefficients associated with the slate expression
    for c in coefficients:
        args.append(c.dat(firedrake.op2.READ, c.cell_node_map(), flatten=True))

    # Append the cell-to-facet mapping for facet integrals (if needed)
    if need_cell_facets:
        args.append(mesh.cell_to_facet_map(firedrake.op2.READ))

    # Performs the cell-loop and generates the global tensor
    firedrake.op2.par_loop(*args)

    # Apply boundary conditions afterwards and assemble
    if bcs is not None and rank == 2:
        for bc in bcs:
            tensor._M.set_local_diagonal_entries(bc.nodes)
    if rank == 2:
        tensor._M.assemble()
    return tensor
