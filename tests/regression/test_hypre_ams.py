import pytest
import numpy
from firedrake import *


@pytest.mark.skipcomplex(reason="Hypre doesn't support complex mode")
def test_homogeneous_field_linear():
    mesh = UnitCubeMesh(5, 5, 5)
    V = FunctionSpace(mesh, "N1curl", 1)
    V0 = VectorFunctionSpace(mesh, "DG", 0)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(u), curl(v))*dx
    L = inner(Constant((0., 0., 0.)), v)*dx

    x, y, z = SpatialCoordinate(mesh)
    B0 = 1
    constant_field = as_vector([-0.5*B0*(y - 0.5), 0.5*B0*(x - 0.5), 0])

    bc = DirichletBC(V, constant_field, (1, 2, 3, 4))

    params = {'snes_type': 'ksponly',
              'ksp_type': 'cg',
              'ksp_max_it': '30',
              'ksp_rtol': '1e-15',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HypreAMS',
              'pc_hypre_ams_zero_beta_poisson': True
              }

    A = Function(V)
    solve(a == L, A, bc, solver_parameters=params)
    B = project(curl(A), V0)
    assert numpy.allclose(B.dat.data_ro, numpy.array((0., 0., 1.)), atol=1e-6)


@pytest.mark.skipcomplex(reason="Hypre doesn't support complex mode")
def test_homogeneous_field_matfree():
    mesh = UnitCubeMesh(5, 5, 5)
    V = FunctionSpace(mesh, "N1curl", 1)
    V0 = VectorFunctionSpace(mesh, "DG", 0)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(u), curl(v))*dx
    L = inner(Constant((0., 0., 0.)), v)*dx

    x, y, z = SpatialCoordinate(mesh)
    B0 = 1
    constant_field = as_vector([-0.5*B0*(y - 0.5), 0.5*B0*(x - 0.5), 0])

    bc = DirichletBC(V, constant_field, (1, 2, 3, 4))

    params = {'snes_type': 'ksponly',
              'mat_type': 'matfree',
              'ksp_type': 'cg',
              'ksp_max_it': '30',
              'ksp_rtol': '1e-15',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.AssembledPC',
              'assembled_pc_type': 'python',
              'assembled_pc_python_type': 'firedrake.HypreAMS',
              'assembled_pc_hypre_ams_zero_beta_poisson': True
              }

    A = Function(V)
    solve(a == L, A, bc, solver_parameters=params)
    B = project(curl(A), V0)
    assert numpy.allclose(B.dat.data_ro, numpy.array((0., 0., 1.)), atol=1e-6)


@pytest.mark.skipcomplex(reason="Hypre doesn't support complex mode")
def test_homogeneous_field_nonlinear():
    mesh = UnitCubeMesh(5, 5, 5)
    V = FunctionSpace(mesh, "N1curl", 1)
    V0 = VectorFunctionSpace(mesh, "DG", 0)

    u = Function(V)
    v = TestFunction(V)

    a = inner(curl(u), curl(v))*dx
    L = inner(Constant((0., 0., 0.)), v)*dx

    x, y, z = SpatialCoordinate(mesh)
    B0 = 1
    constant_field = as_vector([-0.5*B0*(y - 0.5), 0.5*B0*(x - 0.5), 0])

    bc = DirichletBC(V, constant_field, (1, 2, 3, 4))

    params = {'snes_type': 'ksponly',
              'ksp_type': 'cg',
              'ksp_itmax': '30',
              'ksp_rtol': '1e-15',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HypreAMS',
              'pc_hypre_ams_zero_beta_poisson': True
              }

    solve(a - L == 0, u, bc, solver_parameters=params)
    B = project(curl(u), V0)
    assert numpy.allclose(B.dat.data_ro, numpy.array((0., 0., 1.)), atol=1e-6)


@pytest.mark.skipcomplex(reason="Hypre doesn't support complex mode")
def test_homogeneous_field_linear_convergence():
    N = 4
    mesh = UnitCubeMesh(2**N, 2**N, 2**N)
    V = FunctionSpace(mesh, "N1curl", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(u), curl(v))*dx
    L = inner(Constant((0., 0., 0.)), v)*dx

    x, y, z = SpatialCoordinate(mesh)
    B0 = 1
    constant_field = as_vector([-0.5*B0*(y - 0.5), 0.5*B0*(x - 0.5), 0])

    bc = DirichletBC(V, constant_field, (1, 2, 3, 4))

    params = {'snes_type': 'ksponly',
              'ksp_type': 'cg',
              'ksp_max_it': '30',
              'ksp_rtol': '1e-8',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HypreAMS',
              'pc_hypre_ams_zero_beta_poisson': True
              }

    A = Function(V)
    problem = LinearVariationalProblem(a, L, A, bcs=bc)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    assert solver.snes.ksp.getIterationNumber() < 10
