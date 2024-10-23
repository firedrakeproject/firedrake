import pytest
import numpy
from firedrake import *
from firedrake.petsc import get_external_packages


if "hypre" not in get_external_packages():
    pytest.skip("hypre not installed with PETSc", allow_module_level=True)


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
              'ksp_rtol': '2e-15',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HypreAMS',
              'pc_hypre_ams_zero_beta_poisson': True
              }

    A = Function(V)
    solve(a == L, A, bc, solver_parameters=params)
    B = project(curl(A), V0)
    assert numpy.allclose(B.dat.data_ro, numpy.array((0., 0., 1.)), atol=1e-6)


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
              'ksp_rtol': '2e-15',
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
              'ksp_rtol': '2e-15',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HypreAMS',
              'pc_hypre_ams_zero_beta_poisson': True
              }

    solve(a - L == 0, u, bc, solver_parameters=params)
    B = project(curl(u), V0)
    assert numpy.allclose(B.dat.data_ro, numpy.array((0., 0., 1.)), atol=1e-6)


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
    A = Function(V)
    problem = LinearVariationalProblem(a, L, A, bcs=bc)

    # test hypre options
    for cycle_type in (1, 13):
        expected = 9 if cycle_type == 1 else 6
        params = {'snes_type': 'ksponly',
                  'ksp_type': 'cg',
                  'ksp_max_it': '30',
                  'ksp_rtol': '1e-8',
                  'pc_type': 'python',
                  'pc_python_type': 'firedrake.HypreAMS',
                  'pc_hypre_ams_zero_beta_poisson': True,
                  'hypre_ams_pc_hypre_ams_cycle_type': cycle_type,
                  }

        A.assign(0)
        solver = LinearVariationalSolver(problem, solver_parameters=params)
        solver.solve()
        assert solver.snes.ksp.getIterationNumber() == expected
