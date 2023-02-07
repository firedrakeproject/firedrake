from firedrake import *


def test_solve_rhs():
    mesh = UnitSquareMesh(8, 8, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 1)
    u1 = Function(V)
    u2 = Function(V)
    v = TestFunction(V)
    f = Constant(1)

    F1 = inner(grad(u1), grad(v))*dx - inner(f, v)*dx
    F2 = inner(grad(u2), grad(v))*dx
    rhs = assemble(inner(f, v)*dx)

    bcs = DirichletBC(V, 1, "on_boundary")

    solve(F1 == 0, u1, bcs)

    problem = NonlinearVariationalProblem(F2, u2, bcs)
    solver = NonlinearVariationalSolver(problem)
    solver.solve(rhs=rhs)

    u1.assign(u1 - u2)
    assert norm(u1) < 1.0e-10
