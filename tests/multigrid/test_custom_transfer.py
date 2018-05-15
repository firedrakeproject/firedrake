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
