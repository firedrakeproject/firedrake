from firedrake import *
import pytest


@pytest.fixture
def W():
    m = UnitSquareMesh(4, 4)

    V = FunctionSpace(m, 'CG', 1, name="V")
    P = VectorFunctionSpace(m, 'CG', 2, name="P")
    Q = FunctionSpace(m, 'CG', 2, name="Q")
    R = VectorFunctionSpace(m, 'CG', 1, name="R")

    return V*P*Q*R


@pytest.fixture
def A(W):
    u = TrialFunction(W)
    v = TestFunction(W)

    a = inner(u, v)*dx

    return assemble(a, mat_type="aij")


@pytest.fixture
def expect(W):
    f = Function(W)
    f.sub(0).assign(Constant(1))
    f.sub(1).assign(Constant((2, 3)))
    f.sub(2).assign(Constant(4))
    f.sub(3).assign(Constant((5, 6)))
    return f


@pytest.fixture
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


def test_matrix_types(W):
    a = inner(TrialFunction(W), TestFunction(W))*dx

    with pytest.raises(ValueError):
        assemble(a, mat_type="baij")

    A = assemble(a)
    assert A.M.handle.getType() == "seq" + parameters["default_matrix_type"]

    A = assemble(a, mat_type="aij")

    assert A.M.handle.getType() == "seqaij"

    A = assemble(a, mat_type="nest")

    assert A.M.handle.getType() == "nest"

    assert A.M[1, 1].handle.getType() == "seq" + parameters["default_sub_matrix_type"]

    A = assemble(a, mat_type="nest", sub_mat_type="aij")

    assert A.M[1, 1].handle.getType() == "seqaij"

    A = assemble(a, mat_type="nest", sub_mat_type="baij")

    assert A.M[1, 1].handle.getType() == "seqbaij"
