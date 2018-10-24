import pytest
from firedrake import *


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


@pytest.mark.xfail(reason="Transfer operators on subspaces not carried through for monolithic multigrid")
def test_multiple_custom_transfer():
    mesh = UnitIntervalMesh(2)
    mh = MeshHierarchy(mesh, 1)
    mesh = mh[-1]
    count_V = 0
    count_Q = 0

    def prolong_V(coarse, fine):
        nonlocal count_V
        prolong(coarse, fine)
        count_V += 1

    def prolong_Q(coarse, fine):
        nonlocal count_Q
        prolong(coarse, fine)
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

    assert count_V == 1
    assert count_Q == -1


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
