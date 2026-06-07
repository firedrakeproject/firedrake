from firedrake import *
import pytest


@pytest.mark.skipcomplex
def test_bratu():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 3)
    x = SpatialCoordinate(mesh)[0]

    u = Function(V)
    guess = Function(V).interpolate(6*x*(1-x))
    v = TestFunction(V)

    lmbda = Constant(2)

    F = - inner(grad(u), grad(v))*dx + lmbda*inner(exp(u), v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs)

    sp = {"snes_type": "python",
          "snes_python_type": "firedrake.DeflatedSNES",
          "deflated_snes_type": "newtonls",
          "deflated_snes_monitor": None,
          "deflated_snes_linesearch_type": "basic",
          "deflated_ksp_type": "preonly",
          "deflated_pc_type": "lu"}

    deflation = Deflation(op=lambda x, y: inner(x-y, x-y)*dx)
    appctx = {"deflation": deflation}

    # Find the first solution
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, appctx=appctx)
    u.assign(guess)
    solver.solve()

    # The first solution has now been deflated.
    # Find the second solution
    u.assign(guess)
    solver.solve()

    (first, second) = deflation.roots
    assert norm(first - second) > 1


@pytest.mark.skipcomplex
def test_bratu_energy():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 3)

    u = Function(V)
    v = TestFunction(V)
    sol1 = Function(V)
    sol2 = Function(V)

    lmbda = Constant(2)

    E = 0.5 * inner(grad(u), grad(u))*dx + exp(lmbda*u)*dx
    F = inner(grad(u), grad(v))*dx + lmbda*inner(exp(lmbda*u), v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")
    sp = {"snes_monitor": "::ascii_info_detail",
          "snes_type": "newtontr",
          "ksp_monitor": None,
          "ksp_type": "cg",
          "pc_type": "none"}
    problem = NonlinearVariationalProblem(F, u, bcs, objective=E)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()
    sol1.assign(u)

    u.assign(0)
    sp = {"snes_monitor": "::ascii_info_detail"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()
    sol2.assign(u)

    assert norm(sol1 - sol2) < 1.e-8
