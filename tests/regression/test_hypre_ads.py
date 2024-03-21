import pytest
from firedrake import *
from firedrake.petsc import get_external_packages


if "hypre" not in get_external_packages():
    pytest.skip("hypre not installed with PETSc", allow_module_level=True)


def test_homogeneous_field_linear():
    mesh = UnitCubeMesh(10, 10, 10)
    V = FunctionSpace(mesh, "RT", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(div(u), div(v))*dx + inner(u, v)*dx
    L = inner(Constant((1, 0.5, 4)), v)*dx

    bc = DirichletBC(V, Constant((1, 0.5, 4)), (1, 2, 3, 4))

    params = {'snes_type': 'ksponly',
              'ksp_type': 'cg',
              'ksp_max_it': '30',
              'ksp_rtol': '1e-15',
              'ksp_atol': '1e-15',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HypreADS',
              }

    u = Function(V)
    solve(a == L, u, bc, solver_parameters=params)
    assert (errornorm(Constant((1, 0.5, 4)), u, 'L2') < 1e-10)


def test_homogeneous_field_matfree():
    mesh = UnitCubeMesh(10, 10, 10)
    V = FunctionSpace(mesh, "RT", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(div(u), div(v))*dx + inner(u, v)*dx
    L = inner(Constant((1, 0.5, 4)), v)*dx

    bc = DirichletBC(V, Constant((1, 0.5, 4)), (1, 2, 3, 4))

    params = {'snes_type': 'ksponly',
              'mat_type': 'matfree',
              'ksp_type': 'cg',
              'ksp_max_it': '30',
              'ksp_rtol': '1e-15',
              'ksp_atol': '1e-15',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.AssembledPC',
              'assembled_pc_type': 'python',
              'assembled_pc_python_type': 'firedrake.HypreADS',
              }

    u = Function(V)
    solve(a == L, u, bc, solver_parameters=params)
    assert (errornorm(Constant((1, 0.5, 4)), u, 'L2') < 1e-10)


def test_homogeneous_field_nonlinear():
    mesh = UnitCubeMesh(10, 10, 10)
    V = FunctionSpace(mesh, "RT", 1)

    u = Function(V)
    v = TestFunction(V)

    F = inner(div(u), div(v))*dx + inner(u, v)*dx - inner(Constant((1, 0.5, 4)), v)*dx

    bc = DirichletBC(V, Constant((1, 0.5, 4)), (1, 2, 3, 4))

    params = {'snes_type': 'ksponly',
              'ksp_type': 'cg',
              'ksp_itmax': '30',
              'ksp_rtol': '1e-15',
              'ksp_atol': '1e-15',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HypreADS',
              }

    solve(F == 0, u, bc, solver_parameters=params)
    assert (errornorm(Constant((1, 0.5, 4)), u, 'L2') < 1e-10)


def test_homogeneous_field_linear_convergence():
    mesh = UnitCubeMesh(10, 10, 10)
    V = FunctionSpace(mesh, "RT", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(div(u), div(v))*dx + inner(u, v)*dx
    L = inner(Constant((1, 0.5, 4)), v)*dx

    bc = DirichletBC(V, Constant((1, 0.5, 4)), (1, 2, 3, 4))

    A = Function(V)
    problem = LinearVariationalProblem(a, L, A, bcs=bc)

    # test hypre options
    for cycle_type in (1, 13):
        expected = 7 if cycle_type == 1 else 9
        params = {'snes_type': 'ksponly',
                  'ksp_type': 'cg',
                  'ksp_max_it': '30',
                  'ksp_rtol': '1e-8',
                  'pc_type': 'python',
                  'pc_python_type': 'firedrake.HypreADS',
                  'hypre_ads_pc_hypre_ads_cycle_type': cycle_type,
                  }

        A.assign(0)
        solver = LinearVariationalSolver(problem, solver_parameters=params)
        solver.solve()
        assert solver.snes.ksp.getIterationNumber() == expected
