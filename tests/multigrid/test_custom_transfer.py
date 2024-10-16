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

    a = inner(u, v)*dx
    L = conj(v)*dx

    uh = Function(V)
    options = {"ksp_type": "preonly",
               "pc_type": "mg"}

    transfer = TransferManager(native_transfers={V.ufl_element(): (myprolong, restrict, inject)})
    problem = LinearVariationalProblem(a, L, uh)
    solver = LinearVariationalSolver(problem, solver_parameters=options)
    solver.set_transfer_manager(transfer)
    solver.solve()

    assert count == 1

    uh.assign(0)

    solve(a == L, uh, solver_parameters=options)

    assert count == 1


optcount = 0


class CountingTransferManager(TransferManager):
    def prolong(self, *args, **kwargs):
        global optcount
        TransferManager.prolong(self, *args, **kwargs)
        optcount += 1


def test_repeated_custom_transfer_options():
    mesh = UnitIntervalMesh(2)
    mh = MeshHierarchy(mesh, 1)
    mesh = mh[-1]

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v)*dx
    L = conj(v)*dx

    uh = Function(V)
    options = {"ksp_type": "preonly",
               "pc_type": "mg",
               "mg_transfer_manager": __name__ + ".CountingTransferManager"}

    problem = LinearVariationalProblem(a, L, uh)
    solver = LinearVariationalSolver(problem, solver_parameters=options)
    solver.solve()

    global optcount
    assert optcount == 1

    uh.assign(0)

    solve(a == L, uh, solver_parameters=options)

    assert optcount == 2


@pytest.mark.skipcomplexnoslate
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

    a = inner(u, v)*dx + inner(p, q)*dx
    L = conj(v)*dx

    options = {"ksp_type": "preonly",
               "pc_type": "fieldsplit",
               "fieldsplit_pc_type": "mg",
               "pc_fieldsplit_type": "additive",
               "fieldsplit_ksp_type": "preonly",
               "mat_type": "aij"}

    wh = Function(W)
    transfer = TransferManager(native_transfers={V.ufl_element(): (prolong_V, restrict, inject),
                                                 Q.ufl_element(): (prolong_Q, restrict, inject)})
    problem = LinearVariationalProblem(a, L, wh)
    solver = LinearVariationalSolver(problem, solver_parameters=options)
    solver.set_transfer_manager(transfer)

    solver.solve()

    assert count_V == 2
    assert count_Q == -2


@pytest.mark.skipcomplexnoslate
def test_multiple_custom_transfer_monolithic():
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

    a = inner(u, v)*dx + inner(p, q)*dx
    L = conj(v)*dx

    options = {"ksp_type": "preonly",
               "pc_type": "mg",
               "mat_type": "aij"}

    wh = Function(W)
    transfer = TransferManager(native_transfers={V.ufl_element(): (prolong_V, restrict, inject),
                                                 Q.ufl_element(): (prolong_Q, restrict, inject)})
    problem = LinearVariationalProblem(a, L, wh)
    solver = LinearVariationalSolver(problem, solver_parameters=options)
    solver.set_transfer_manager(transfer)

    solver.solve()
    assert count_V == 2
    assert count_Q == -2


@pytest.mark.parametrize("mode", ("full", pytest.param("partial", marks=pytest.mark.skipcomplexnoslate)))
def test_custom_transfer_setting(mode):
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

    a = inner(u, v)*dx
    L = conj(v)*dx

    uh = Function(V)
    options = {"ksp_type": "preonly",
               "pc_type": "mg"}

    if mode == "partial":
        transfer_ops = (myprolong, None, None)
    else:
        transfer_ops = (myprolong, restrict, inject)

    transfer = TransferManager(native_transfers={V.ufl_element(): transfer_ops})
    problem = LinearVariationalProblem(a, L, uh)
    solver = LinearVariationalSolver(problem, solver_parameters=options)
    solver.set_transfer_manager(transfer)

    solver.solve()

    with pytest.raises(ValueError):
        solver.set_transfer_manager(transfer)

    assert count == 1
