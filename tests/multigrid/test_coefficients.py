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


def test_mixed_coefficients_retransferred():
    mesh = UnitSquareMesh(1, 1)
    mh = MeshHierarchy(mesh, 2)
    mesh = mh[-1]
    V = FunctionSpace(mesh, "DG", 0)
    W = V*V

    u, p = TrialFunctions(W)

    v, q = TestFunctions(W)

    x = Function(W)
    c, g = x.split()
    c.assign(1)
    g.assign(3)
    c, g = split(x)
    a = c*u*v*dx + g*p*q*dx

    L = v*dx + q*dx

    uh = Function(W, name="solution")
    problem = LinearVariationalProblem(a, L, uh)

    problem = LinearVariationalProblem(a, L, uh, constant_jacobian=False)

    solver = LinearVariationalSolver(problem, solver_parameters={
        "pc_type": "fieldsplit",
        "ksp_type": "preonly",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_0": {
            "ksp_type": "preonly",
            "pc_type": "mg",
            "mg_levels_ksp_max_it": 0,
            "mg_levels_pc_type": "none",
        }
    }, options_prefix="")

    solver.solve()

    a, b = uh.split()
    assert numpy.allclose(a.dat.data_ro, 1)
    assert numpy.allclose(b.dat.data_ro, 1/3)

    c, g = x.split()
    c.assign(10)

    g.assign(1/3)
    uh.assign(0)
    solver.solve()
    assert numpy.allclose(a.dat.data_ro, 0.1)
    assert numpy.allclose(b.dat.data_ro, 3)


def test_mixed_custom_transfer_subblock():
    mesh = UnitSquareMesh(1, 1)
    mh = MeshHierarchy(mesh, 2)
    mesh = mh[-1]
    V = FunctionSpace(mesh, "DG", 0)
    W = V*V

    u, p = TrialFunctions(W)

    v, q = TestFunctions(W)

    x = Function(W)
    c, g = x.split()
    c.assign(1)
    g.assign(3)
    c, g = split(x)
    a = c*u*v*dx + g*p*q*dx

    L = v*dx + q*dx

    def myinject(fine, coarse):
        coarse.assign(100)

    transfer = dmhooks.transfer_operators(W.sub(0), inject=myinject)
    uh = Function(W, name="solution")
    problem = LinearVariationalProblem(a, L, uh)

    problem = LinearVariationalProblem(a, L, uh, constant_jacobian=False)

    solver = LinearVariationalSolver(problem, solver_parameters={
        "pc_type": "fieldsplit",
        "ksp_type": "preonly",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_0": {
            "ksp_type": "preonly",
            "pc_type": "mg",
            "mg_levels_ksp_max_it": 0,
            "mg_levels_pc_type": "none",
        }
    }, options_prefix="")

    solver.set_transfer_operators(transfer)
    solver.solve()

    a, b = uh.split()
    assert numpy.allclose(a.dat.data_ro, 1/100)
    assert numpy.allclose(b.dat.data_ro, 1/3)

    c, g = x.split()
    c.assign(10)

    g.assign(1/3)
    uh.assign(0)
    solver.solve()
    assert numpy.allclose(a.dat.data_ro, 1/100)
    assert numpy.allclose(b.dat.data_ro, 3)
