import pytest
import numpy
from functools import partial
from itertools import count
from firedrake import *
from firedrake.mg.ufl_utils import coarsen


def test_repeated_custom_transfer():
    mesh = UnitIntervalMesh(2)
    mh = MeshHierarchy(mesh, 1)
    mesh = mh[-1]
    count = 0

    def myprolong(coarse, fine):
        nonlocal count
        prolong(coarse, fine)
        count += 1

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx
    L = v*dx

    uh = Function(V)
    options = {"ksp_type": "preonly",
               "pc_type": "mg"}

    with dmhooks.transfer_operators(V, prolong=myprolong):
        solve(a == L, uh, solver_parameters=options)

    assert count == 1

    uh.assign(0)

    solve(a == L, uh, solver_parameters=options)

    assert count == 1


def test_multiple_custom_transfer_split():
    mesh = UnitIntervalMesh(2)
    mh = MeshHierarchy(mesh, 2)
    mesh = mh[-1]
    count_V = 0
    count_Q = 0

    def prolong_V(coarse, fine):
        nonlocal count_V
        prolong(coarse, fine)
        count_V += 1

    def prolong_Q(fine, coarse):
        nonlocal count_Q
        prolong(fine, coarse)
        count_Q -= 1

    V = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)

    W = V*Q
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a = u*v*dx + p*q*dx
    L = v*dx

    options = {"ksp_type": "preonly",
               "pc_type": "fieldsplit",
               "fieldsplit_pc_type": "mg",
               "pc_fieldsplit_type": "additive",
               "fieldsplit_ksp_type": "preonly",
               "mat_type": "aij"}

    wh = Function(W)
    problem = LinearVariationalProblem(a, L, wh)
    solver = LinearVariationalSolver(problem, solver_parameters=options)
    solver.set_transfer_operators(dmhooks.transfer_operators(V, prolong=prolong_V),
                                  dmhooks.transfer_operators(Q, prolong=prolong_Q))

    solver.solve()

    assert count_V == 2
    assert count_Q == -2


def test_multiple_custom_transfer_monolithc():
    mesh = UnitIntervalMesh(2)
    mh = MeshHierarchy(mesh, 2)
    mesh = mh[-1]
    count_V = 0
    count_Q = 0

    def prolong_V(coarse, fine):
        nonlocal count_V
        prolong(coarse, fine)
        count_V += 1

    def prolong_Q(fine, coarse):
        nonlocal count_Q
        prolong(fine, coarse)
        count_Q -= 1

    V = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)

    W = V*Q
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a = u*v*dx + p*q*dx
    L = v*dx

    options = {"ksp_type": "preonly",
               "pc_type": "mg",
               "mat_type": "aij"}

    wh = Function(W)
    problem = LinearVariationalProblem(a, L, wh)
    solver = LinearVariationalSolver(problem, solver_parameters=options)
    solver.set_transfer_operators(dmhooks.transfer_operators(V, prolong=prolong_V),
                                  dmhooks.transfer_operators(Q, prolong=prolong_Q))

    solver.solve()

    assert count_V == 2
    assert count_Q == -2


def test_custom_transfer_setting():
    mesh = UnitIntervalMesh(2)
    mh = MeshHierarchy(mesh, 1)
    mesh = mh[-1]
    count = 0

    def myprolong(coarse, fine):
        nonlocal count
        prolong(coarse, fine)
        count += 1

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx
    L = v*dx

    uh = Function(V)
    options = {"ksp_type": "preonly",
               "pc_type": "mg"}

    problem = LinearVariationalProblem(a, L, uh)
    solver = LinearVariationalSolver(problem, solver_parameters=options)
    solver.set_transfer_operators(dmhooks.transfer_operators(V, prolong=myprolong))

    solver.solve()

    with pytest.raises(RuntimeError):
        solver.set_transfer_operators(dmhooks.transfer_operators(V, prolong=myprolong))

    assert count == 1


def test_transfers_transferring():
    mesh = UnitIntervalMesh(2)
    mh = MeshHierarchy(mesh, 1)

    mesh = mh[-1]

    Vf = FunctionSpace(mesh, "P", 1)
    Qf = FunctionSpace(mesh, "DP", 0)
    Wf = Vf*Qf

    Vcount = count()
    Qcount = count()
    Wcount = count()

    def myinject(f, c, counter=None):
        next(counter)
        inject(f, c)

    injectQ = partial(myinject, counter=Qcount)
    injectV = partial(myinject, counter=Vcount)
    injectW = partial(myinject, counter=Wcount)

    f = Function(Wf)
    f.assign(1)

    with dmhooks.transfer_operators(Wf, inject=injectW):
        c = coarsen(f, coarsen)
        assert next(Wcount) == 1
        for d in c.split():
            assert numpy.allclose(d.dat.data_ro, 1)

    f.sub(0).assign(2)
    f.sub(1).assign(3)
    with dmhooks.transfer_operators(Wf.sub(0), inject=injectV), dmhooks.transfer_operators(Wf.sub(1), inject=injectQ):
        c = coarsen(f, coarsen)
        assert next(Vcount) == 1
        assert next(Qcount) == 1

        assert numpy.allclose(c.sub(0).dat.data_ro, 2)
        assert numpy.allclose(c.sub(1).dat.data_ro, 3)
