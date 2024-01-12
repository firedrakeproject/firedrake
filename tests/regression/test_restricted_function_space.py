import pytest
import numpy as np
from firedrake import *


@pytest.mark.parametrize("j", [1, 2, 5])
def test_restricted_function_space_1_1_square(j):
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", j)
    x, y = SpatialCoordinate(mesh)
    u = TrialFunction(V)
    v = TestFunction(V)
    original_form = u * v * dx

    bc = DirichletBC(V, 0, 2)
    V_res = RestrictedFunctionSpace(V, name="Restricted", bcs=[bc])

    u2 = TrialFunction(V_res)
    v2 = TestFunction(V_res)
    restricted_form = u2 * v2 * dx
    restricted_fs_matrix = assemble(restricted_form) 
    normal_fs_matrix = assemble(original_form, bcs=[bc])

    identity = np.identity(normal_fs_matrix.M.nrows)
    delete_rows = []
    for i in range(normal_fs_matrix.M.nrows):
        row = normal_fs_matrix.M.values[i, :]
        comparison = row == identity[i, :]
        if comparison.all():
            delete_rows.append(i)

    # Remove all rows/columns associated with boundaries (i.e. identity)
    normal_fs_matrix_reduced = np.delete(normal_fs_matrix.M.values, delete_rows,
                                         axis=0)
    normal_fs_matrix_reduced = np.delete(normal_fs_matrix_reduced, delete_rows,
                                         axis=1)

    assert (restricted_fs_matrix.M.nrows == np.shape(normal_fs_matrix_reduced)[0])
    assert (restricted_fs_matrix.M.ncols == np.shape(normal_fs_matrix_reduced)[1])
    assert (np.array_equal(normal_fs_matrix_reduced, restricted_fs_matrix.M.values))
