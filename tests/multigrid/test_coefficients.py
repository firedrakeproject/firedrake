import numpy
from firedrake import *


def test_coefficients_retransferred():
    mesh = UnitSquareMesh(1, 1)

    mh = MeshHierarchy(mesh, 2)

    mesh = mh[-1]

    V = FunctionSpace(mesh, "P", 1)

    u = TrialFunction(V)

    v = TestFunction(V)

    Vdg = FunctionSpace(mesh, "DG", 0)

    c = Function(Vdg)

    c.assign(1)

    a = c*u*v*dx

    L = v*dx

    uh = Function(V, name="solution")
    problem = LinearVariationalProblem(a, L, uh)

    problem = LinearVariationalProblem(a, L, uh, constant_jacobian=False)

    solver = LinearVariationalSolver(problem, solver_parameters={"pc_type": "mg",
                                                                 "ksp_type": "preonly",
                                                                 # Coarse grid correction is exact
                                                                 "mg_levels_pc_type": "none",
                                                                 "mg_levels_ksp_max_it": 0})

    solver.solve()
    assert numpy.allclose(uh.dat.data_ro, 1)
    assert numpy.allclose(solver._ctx._coarse.J.coefficients()[0].dat.data_ro, 1)

    c.assign(10)

    uh.assign(0)
    solver.solve()
    assert numpy.allclose(uh.dat.data_ro, 0.1)
    assert numpy.allclose(solver._ctx._coarse.J.coefficients()[0].dat.data_ro, 10)
