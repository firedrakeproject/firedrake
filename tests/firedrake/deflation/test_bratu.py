from firedrake import *


def test_bratu(output=False):
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 3)
    x = SpatialCoordinate(mesh)[0]

    u = Function(V)
    u.interpolate(6*x*(1-x))
    v = TestFunction(V)

    lmbda = Constant(2)

    F = - inner(grad(u), grad(v))*dx + lmbda*inner(exp(u), v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs)

    sp = {"snes_type": "python",
          "snes_python_type": "firedrake.DeflatedSNES",
          "deflated_snes_type": "newtonls",
          "deflated_snes_monitor": None,
          "deflated_snes_linesearch_type": "l2",
          "deflated_ksp_type": "preonly",
          "deflated_pc_type": "lu"}

    deflation = Deflation(op=lambda x, y: inner(x-y, x-y)*dx)
    appctx = {"deflation": deflation}

    # Find the first solution
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, appctx=appctx)
    solver.solve()

    # Now deflate the first solution found, restore initial guess, and
    # find second solution
    first = Function(u)
    deflation.append(first)
    u.interpolate(6*x*(1-x))
    solver.solve()

    second = Function(u)

    print(f"Norm of difference: {norm(first - second)}")
    assert norm(first - second) > 1

    if output:
        return (first, second)


if __name__ == "__main__":
    (first, second) = test_bratu(output=True)

    import matplotlib.pyplot as plt
    ax = plt.gca()
    plot(first, linestyle='-', edgecolor='tab:blue', axes=ax)
    plot(second, linestyle='--', edgecolor='tab:red', axes=ax)
    plt.show()
