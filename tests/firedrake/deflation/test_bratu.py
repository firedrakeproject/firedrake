from firedrake import *

def test_bratu():
    mesh = UnitIntervalMesh(400)
    V = FunctionSpace(mesh, "CG", 2)

    u = Function(V)
    v = TestFunction(V)

    lmbda = Constant(2)

    F = - inner(grad(u), grad(v))*dx + lmbda*inner(exp(u), v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs)

    sp = {"snes_type": "python",
          "snes_python_type": "firedrake.DeflatedSNES",
          "snes_view": None,
          "deflated_snes_type": "newtonls",
          "deflated_snes_monitor": None,
          "deflated_snes_linesearch_type": "l2",
          "deflated_ksp_type": "preonly",
          "deflated_pc_type": "lu"}

    deflation = None # Deflation([])
    appctx = {"deflation": deflation}
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, appctx=appctx)
    solver.solve()

if __name__ == "__main__":
    test_bratu()
