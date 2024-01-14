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


@pytest.mark.parametrize("j", [1, 2, 5])
def test_restricted_function_space_j_j_square(j):
    mesh = UnitSquareMesh(j, j)
    V = FunctionSpace(mesh, "CG", 3)  # this fails for CG1
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


def test_poisson_homogeneous_bcs():

    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 2)   # fails for CG1
    f = Function(V)
    x, y = SpatialCoordinate(mesh)

    u = TrialFunction(V)
    v = TestFunction(V)

    f.interpolate(16 * pi**2 * (y-1)**2 * y**2 - 2 * (y-1)**2 - 8 * (y-1)*y 
                  - 2*y**2)*sin(4*pi*x)

    original_form = inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, 0, 1)
    V_res = RestrictedFunctionSpace(V, name="Restricted", bcs=[bc])
    f2 = Function(V_res)

    f2.interpolate(16 * pi**2 * (y-1)**2 * y**2 - 2 * (y-1)**2 - 8 * (y-1)*y
                   - 2*y**2)*sin(4*pi*x)

    u2 = TrialFunction(V_res)
    v2 = TestFunction(V_res)
    restricted_form = inner(grad(u2), grad(v2)) * dx

    L = inner(f, v) * dx
    L_res = inner(f2, v2) * dx

    u = Function(V)
    u2 = Function(V_res)
    solve(original_form == L, u, bcs=[bc])
    solve(restricted_form == L_res, u2)

    # might run into problems if other non-boundary nodes evaluate at 0?
    u_data_remove_zeros = u.dat.data[u.dat.data != 0]  # correspond to boundary -> different ordering for RFS / FS
    u2_data_remove_zeros = u2.dat.data[u2.dat.data != 0]

    assert (np.all(np.isclose(u_data_remove_zeros, u2_data_remove_zeros, 1e-16)))
