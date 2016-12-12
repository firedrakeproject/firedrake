from firedrake import *
import numpy as np
import pytest


def test_callbacks():
    mesh = UnitIntervalMesh(10)

    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V).assign(1)
    temp = Function(V)
    alpha = Constant(0.0)
    beta = Constant(0.0)

    F = alpha*u*v*dx - beta*f*v*dx  # we will override alpha and beta later

    def update_alpha(current_solution):
        alpha.assign(1.0)

    def update_beta(current_solution):
        with temp.dat.vec as foo:
            current_solution.copy(foo)

        # introduce current-solution-dependent behaviour
        if temp.dat.data[0] == 0.0:
            beta.assign(1.5)  # this branch is hit at the first iteration
        else:
            beta.assign(2.0)

    problem = NonlinearVariationalProblem(F, u)

    solver = NonlinearVariationalSolver(problem,
                                        pre_jacobian_callback=update_alpha,
                                        pre_function_callback=update_beta)

    solver.solve()

    assert np.allclose(u.dat.data, 2.0)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
