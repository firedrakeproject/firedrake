from firedrake import *


def test_vi():
    mesh = IntervalMesh(1, 5)
    R = FunctionSpace(mesh, "DG", 0)

    # Energy function, specified by points to interpolate
    pts = [(0, 0),
           (1, 1),
           (2, 0.5),
           (3, 2),
           (4, 1.5),
           (5, 5)]

    def J(x):
        y = 0
        for (i, pt) in enumerate(pts):
            prod = 1
            for (j, ptj) in enumerate(pts):
                if j != i:
                    # Symbolically build Lagrange interpolating polynomial
                    prod *= (x - Constant(ptj[0])) / (Constant(pt[0]) - Constant(ptj[0]))

            y += pt[1] * prod

        return y

    guess = Constant(1)
    u = Function(R)
    F = derivative(J(u)*dx, u)

    sp = {"snes_type": "python",
          "snes_python_type": "firedrake.DeflatedSNES",
          "deflated_snes_type": "vinewtonrsls",
          "deflated_snes_monitor": None,
          "deflated_snes_linesearch_type": "basic",
          "deflated_ksp_type": "preonly",
          "deflated_pc_type": "lu"}

    problem = NonlinearVariationalProblem(F, u)
    deflation = Deflation(op=lambda x, y: inner(x-y, x-y)*dx)
    appctx = {"deflation": deflation}
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, appctx=appctx)
    lb = Function(R).interpolate(Constant(0))
    ub = Function(R).interpolate(Constant(100))

    values = []
    # Find the solutions and deflate
    for i in range(5):
        u.interpolate(guess)
        try:
            solver.solve(bounds=(lb, ub))
        except ConvergenceError:
            break
        soln = Function(u)
        values.append(soln.at((0.5,)))
        print(f"Found solution: {values[-1]}")
        deflation.append(soln)

    assert len(values) == 5
    return (mesh, J, values)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    (mesh, J, values) = test_vi()

    W = FunctionSpace(mesh, "DG", 10)
    x = SpatialCoordinate(mesh)[0]
    J_ = Function(W, name="Energy").interpolate(J(x))

    ax = plt.gca()
    plot(J_, axes=ax)
    for value in values:
        plt.plot([value], [J_(value)], '*k')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$J(x)$")
    plt.title("Energy landscape and critical points")
    plt.show()
