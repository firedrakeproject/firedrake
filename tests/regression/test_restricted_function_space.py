import pytest
import numpy as np
from firedrake import *


def compare_function_space_assembly(mesh, function_space, restricted_function_space, bcs, res_bcs=[]):
    x, y = SpatialCoordinate(mesh)
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    original_form = u * v * dx

    u2 = TrialFunction(restricted_function_space)
    v2 = TestFunction(restricted_function_space)
    restricted_form = u2 * v2 * dx

    normal_fs_matrix = assemble(original_form, bcs=bcs)
    restricted_fs_matrix = assemble(restricted_form, bcs=res_bcs)

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
def test_restricted_function_space_1_1_square(j):
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", j)
    bc = DirichletBC(V, 0, 2)
    V_res = RestrictedFunctionSpace(V, name="Restricted", boundary_set=[2])
    res_bc = DirichletBC(V_res, 0, 2)
    compare_function_space_assembly(mesh, V, V_res, [bc], [res_bc])


@pytest.mark.parametrize("j", [1, 2, 5])
def test_restricted_function_space_j_j_square(j):
    mesh = UnitSquareMesh(j, j)
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0, 2)
    V_res = RestrictedFunctionSpace(V, name="Restricted", boundary_set=[2])

    compare_function_space_assembly(mesh, V, V_res, [bc])


def test_poisson_homogeneous_bcs():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)

    u = TrialFunction(V)
    v = TestFunction(V)

    f.interpolate(16 * pi**2 * (y-1)**2 * y**2 - 2 * (y-1)**2 - 8 * (y-1)*y
                  - 2*y**2)*sin(4*pi*x)

    original_form = inner(grad(u), grad(v)) * dx

    bc = DirichletBC(V, 0, 1)
    V_res = RestrictedFunctionSpace(V, name="Restricted", boundary_set=[1])
    f2 = Function(V_res)

    f2.interpolate(16 * pi**2 * (y-1)**2 * y**2 - 2 * (y-1)**2 - 8 * (y-1)*y
                   - 2*y**2)*sin(4*pi*x)

    u2 = TrialFunction(V_res)
    v2 = TestFunction(V_res)
    restricted_form = inner(grad(u2), grad(v2)) * dx

    L = inner(f, v) * dx
    L_res = inner(f2, v2) * dx

    u = Function(V)
    bc2 = DirichletBC(V_res, Constant(0), 1)
    u2 = Function(V_res)

    solve(original_form == L, u, bcs=[bc])
    solve(restricted_form == L_res, u2, bcs=[bc2])

    u_data_remove_zeros = u.dat.data[u.dat.data != 0]  # correspond to boundary
    u2_data_remove_zeros = u2.dat.data[u2.dat.data != 0]

    assert (np.all(np.isclose(u_data_remove_zeros, u2_data_remove_zeros, 1e-16)))


def test_poisson_inhomogeneous_bcs():
    mesh = UnitSquareMesh(1, 2)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    V_res = RestrictedFunctionSpace(V, boundary_set=[1, 2])
    bc = DirichletBC(V_res, 0, 1)
    bc2 = DirichletBC(V_res, 1, 2)
    u2 = TrialFunction(V_res)
    v2 = TestFunction(V_res)
    restricted_form = inner(grad(u2), grad(v2)) * dx
    u = Function(V_res)
    rhs = Constant(0) * v2 * dx

    solve(restricted_form == rhs, u, bcs=[bc, bc2])

    assert errornorm(SpatialCoordinate(mesh)[0], u) < 1.e-12


@pytest.mark.parametrize("j", ["2", "sin(x) * y", "x**2"])
def test_poisson_inhomogeneous_bcs_2(j):
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 2)

    x, y = SpatialCoordinate(mesh)

    bc3 = DirichletBC(V, 0, 1)
    bc4 = DirichletBC(V, Function(V).interpolate(eval(j)), 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    V_res = RestrictedFunctionSpace(V, name="Restricted", boundary_set=[1, 2])

    bc = DirichletBC(V_res, 0, 1)
    bc2 = DirichletBC(V_res, Function(V_res).interpolate(eval(j)), 2)

    u2 = TrialFunction(V_res)
    v2 = TestFunction(V_res)

    original_form = inner(grad(u), grad(v)) * dx
    restricted_form = inner(grad(u2), grad(v2)) * dx

    u = Function(V)
    u2 = Function(V_res)

    rhs = Constant(0) * v * dx
    rhs2 = Constant(0) * v2 * dx

    solve(original_form == rhs, u, bcs=[bc3, bc4])
    solve(restricted_form == rhs2, u2, bcs=[bc, bc2])

    assert np.linalg.norm(np.sort(u2.dat.data) - np.sort(u.dat.data)) < 1.e-12


@pytest.mark.parametrize("j", [1, 2, 5])
def test_restricted_function_space_coord_change(j):
    mesh = UnitSquareMesh(1, 2)
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, "CG", 2)
    new_mesh = Mesh(Function(V).interpolate(as_vector([x, y])))
    new_V = FunctionSpace(new_mesh, "CG", j)
    bc = DirichletBC(new_V, 0, 1)
    new_V_restricted = RestrictedFunctionSpace(new_V, name="Restricted", boundary_set=[1])

    compare_function_space_assembly(new_mesh, new_V, new_V_restricted, [bc])
