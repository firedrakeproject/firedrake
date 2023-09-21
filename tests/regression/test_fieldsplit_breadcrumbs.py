import numpy
from firedrake import *


def test_fieldsplit_breadcrumbs():
    mesh = UnitSquareMesh(10, 10)
    V_u = VectorFunctionSpace(mesh, 'DG', 1)
    V_eta = FunctionSpace(mesh, 'DG', 1)
    V = MixedFunctionSpace([V_u, V_eta])

    test = TestFunction(V)
    test_u, test_eta = TestFunctions(V)

    u_source = Function(V_u, name='source')
    solution = Function(V, name='solution')

    F = - inner(solution, test) * dx + inner(u_source, test_u) * dx

    solver_parameters = {
        'ksp_type': 'gmres',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'multiplicative',
        'snes_type': 'newtonls'
    }
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=solver_parameters)

    x, y = SpatialCoordinate(mesh)
    g = Function(V_u, name='aux')
    g.interpolate(as_vector((x, y)))

    for i in range(3):
        u_source.assign(g*i)
        solver.solve()

        u, eta = solution.subfunctions
        assert numpy.allclose(u.dat.data_ro, u_source.dat.data_ro)
        assert numpy.allclose(eta.dat.data_ro, 0)
