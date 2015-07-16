from firedrake import *
import pytest


@pytest.fixture(scope="module")
def W():
    m = UnitSquareMesh(4, 4)

    V = FunctionSpace(m, 'CG', 1, name="V")
    P = VectorFunctionSpace(m, 'CG', 2, name="P")
    Q = FunctionSpace(m, 'CG', 2, name="Q")
    R = VectorFunctionSpace(m, 'CG', 1, name="R")

    return V*P*Q*R


@pytest.fixture(scope="module")
def A(W):
    u = TrialFunction(W)
    v = TestFunction(W)

    a = inner(u, v)*dx

    return assemble(a, nest=False)


@pytest.fixture(scope="module")
def expect():
    return Constant((1, 2, 3, 4, 5, 6))


@pytest.fixture(scope="module")
def b(W, expect):
    v = TestFunction(W)

    L = inner(expect, v)*dx

    return assemble(L)


@pytest.mark.parametrize("parameters",
                         [{"ksp_type": "preonly",
                           "pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "fieldsplit_V_pc_type": "lu",
                           "fieldsplit_P_pc_type": "lu",
                           "fieldsplit_Q_pc_type": "lu",
                           "fieldsplit_R_pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "pc_fieldsplit_0_fields": "0, 3",
                           "pc_fieldsplit_1_fields": "1, 2",
                           "fieldsplit_pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "pc_fieldsplit_0_fields": "0, 2",
                           "pc_fieldsplit_1_fields": "1",
                           "pc_fieldsplit_2_fields": "3",
                           "fieldsplit_0_pc_type": "lu",
                           "fieldsplit_P_pc_type": "lu",
                           "fieldsplit_R_pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "pc_fieldsplit_0_fields": "0, 2, 3",
                           "pc_fieldsplit_1_fields": "1",
                           "fieldsplit_0_pc_type": "lu",
                           "fieldsplit_P_pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "pc_fieldsplit_0_fields": "0, 2, 3",
                           "pc_fieldsplit_1_fields": "1",
                           "fieldsplit_0_pc_type": "fieldsplit",
                           "fieldsplit_0_pc_fieldsplit_type": "additive",
                           "fieldsplit_0_fieldsplit_V_pc_type": "lu",
                           "fieldsplit_0_fieldsplit_Q_pc_type": "lu",
                           "fieldsplit_0_fieldsplit_R_pc_type": "lu",
                           "fieldsplit_P_pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "pc_fieldsplit_0_fields": "0, 2, 3",
                           "pc_fieldsplit_1_fields": "1",
                           "fieldsplit_0_pc_type": "fieldsplit",
                           "fieldsplit_0_pc_fieldsplit_type": "additive",
                           "fieldsplit_0_pc_fieldsplit_0_fields": "1",
                           "fieldsplit_0_pc_fieldsplit_1_fields": "0, 2",
                           "fieldsplit_0_fieldsplit_Q_pc_type": "lu",
                           "fieldsplit_0_fieldsplit_1_pc_type": "lu",
                           "fieldsplit_P_pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "pc_fieldsplit_0_fields": "0, 2, 3",
                           "pc_fieldsplit_1_fields": "1",
                           "fieldsplit_0_pc_type": "fieldsplit",
                           "fieldsplit_0_pc_fieldsplit_type": "additive",
                           "fieldsplit_0_pc_fieldsplit_0_fields": "1",
                           "fieldsplit_0_pc_fieldsplit_1_fields": "0, 2",
                           "fieldsplit_0_fieldsplit_Q_pc_type": "lu",
                           "fieldsplit_0_fieldsplit_1_pc_type": "fieldsplit",
                           "fieldsplit_0_fieldsplit_1_pc_fieldsplit_type": "additive",
                           "fieldsplit_0_fieldsplit_1_fieldsplit_V_pc_type": "lu",
                           "fieldsplit_0_fieldsplit_1_fieldsplit_R_pc_type": "lu",
                           "fieldsplit_P_pc_type": "lu"}],
                         ids=["LU",
                              "FS 4 blocks + LU",
                              "FS 2 blocks [(0, 3), (1, 2)] + LU",
                              "FS 3 blocks [(0, 2), (1, ), (3, )] + LU",
                              "FS 2 blocks [(0, 2, 3), (1, )] + LU",
                              "FS 2 blocks [(0, 2, 3), (1, )] + block 1 LU, block 0 FS 3 blocks + LU",
                              "FS 2 blocks [(0, 2, 3), (1, )] + block 1 LU, block 0 FS 2 blocks [(1, ), (0, 2)] + LU",
                              "FS 2 blocks [(0, 2, 3), (1, )] + block 1 LU, block 0 FS 2 blocks [(1, ), (0, 2)] + block 0 LU, block 1 FS 2 blocks + LU"])
def test_nested_fieldsplit_solve(W, A, b, expect, parameters):
    solver = LinearSolver(A, solver_parameters=parameters)

    f = Function(W)

    solver.solve(f, b)

    f -= expect
    assert norm(f) < 1e-11


@pytest.mark.parallel(nprocs=3)
def test_nested_fieldsplit_solve_parallel(W, A, b, expect):
    parameters = {"ksp_type": "preonly",
                  "pc_type": "fieldsplit",
                  "pc_fieldsplit_type": "additive",
                  "pc_fieldsplit_0_fields": "0, 3",
                  "pc_fieldsplit_1_fields": "1, 2",
                  "fieldsplit_ksp_type": "cg",
                  "fieldsplit_ksp_rtol": 1e-12,
                  "fieldsplit_pc_type": "bjacobi",
                  "fieldsplit_sub_pc_type": "lu"}
    solver = LinearSolver(A, solver_parameters=parameters)
    f = Function(W)

    solver.solve(f, b)

    f -= expect
    assert norm(f) < 1e-11


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
