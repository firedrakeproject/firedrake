from firedrake import *
from firedrake import matrix
import pytest


@pytest.fixture
def V():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    return V


@pytest.fixture
def test(V):
    return TestFunction(V)


@pytest.fixture
def trial(V):
    return TrialFunction(V)


@pytest.fixture
def a(test, trial):
    return inner(trial, test) * dx


@pytest.fixture(params=["nest", "aij", "matfree"])
def mat_type(request):
    return request.param


def test_assemble_returns_matrix(a):
    A = assemble(a)

    assert isinstance(A, matrix.Matrix)


def test_solve_with_assembled_matrix(a):
    (v, u) = a.arguments()
    V = v.function_space()
    x, = SpatialCoordinate(V.mesh())
    f = Function(V).interpolate(x)

    A = AssembledMatrix((v, u), bcs=(), petscmat=assemble(a).petscmat)
    L = inner(f, v) * dx

    solution = Function(V)
    solve(A == L, solution)

    assert norm(assemble(f - solution)) < 1e-15


@pytest.fixture
def matrices():
    mesh = UnitSquareMesh(4, 4)
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)

    M1 = assemble(inner(TrialFunction(V1), TestFunction(V1)) * dx)
    M1_2 = assemble(inner(2 * TrialFunction(V1), TestFunction(V1)) * dx)
    I1 = assemble(interpolate(TrialFunction(V1), V1))  # Identity on V1
    I12 = assemble(interpolate(TrialFunction(V1), V2))  # Interp from V1 to V2
    return M1, M1_2, I1, I12


def test_matrix_matmul(matrices):
    M1, M1_2, I1, _ = matrices

    # Test matrix-matrix multiplication
    matmul_symb = M1 @ I1
    res1 = assemble(matmul_symb)
    assert isinstance(matmul_symb, ufl.Action)
    assert isinstance(res1, matrix.AssembledMatrix)
    assert np.allclose(res1.petscmat[:, :], M1.petscmat[:, :])

    # Incompatible matmul
    with pytest.raises(TypeError, match="Incompatible function spaces in Action"):
        M1 @ M1_2


def test_matrix_addition(matrices):
    M1, M1_2, I1, _ = matrices

    V1 = M1.arguments()[0].function_space()

    # Test matrix addition
    matadd_symb = I1 + I1
    res2 = assemble(matadd_symb)
    assert isinstance(matadd_symb, ufl.FormSum)
    assert isinstance(res2, matrix.AssembledMatrix)
    assert np.allclose(res2.petscmat[:, :], 2 * np.eye(V1.dim()))

    # Matrix addition incorrect args
    with pytest.raises(ValueError, match="Arguments in matrix addition must match."):
        I1 + M1

    # Different matrices, correct arguments
    matadd_symb3 = M1 + M1_2
    res_add3 = assemble(matadd_symb3)
    assert isinstance(matadd_symb3, ufl.FormSum)
    assert isinstance(res_add3, matrix.AssembledMatrix)
    assert np.allclose(res_add3.petscmat[:, :], 3 * M1.petscmat[:, :])


def test_matrix_subtraction(matrices):
    M1, M1_2, I1, _ = matrices

    matsub = M1 - M1_2
    res_sub = assemble(matsub)
    assert isinstance(matsub, ufl.FormSum)
    assert isinstance(res_sub, matrix.AssembledMatrix)
    assert np.allclose(res_sub.petscmat[:, :], -1 * M1.petscmat[:, :])

    matsub2 = M1_2 - M1
    res_sub2 = assemble(matsub2)
    assert isinstance(matsub2, ufl.FormSum)
    assert isinstance(res_sub2, matrix.AssembledMatrix)
    assert np.allclose(res_sub2.petscmat[:, :], M1.petscmat[:, :])

    with pytest.raises(ValueError, match="Arguments in matrix subtraction must match."):
        I1 - M1


def test_matrix_scalar_multiplication(matrices):
    M1, _, _, _ = matrices

    # Test left scalar multiplication
    matscal_left = 2.5 * M1
    res7 = assemble(matscal_left)
    assert isinstance(matscal_left, ufl.FormSum)
    assert isinstance(res7, matrix.AssembledMatrix)
    assert np.allclose(res7.petscmat[:, :], 2.5 * M1.petscmat[:, :])

    # Test right scalar multiplication
    matscal_right = M1 * 3.0
    res8 = assemble(matscal_right)
    assert isinstance(matscal_right, ufl.FormSum)
    assert isinstance(res8, matrix.AssembledMatrix)
    assert np.allclose(res8.petscmat[:, :], 3.0 * M1.petscmat[:, :])

    # Test with Constant
    c = Constant(4.0)
    matscal_const = c * M1
    res_const = assemble(matscal_const)
    assert isinstance(matscal_const, ufl.FormSum)
    assert isinstance(res_const, matrix.AssembledMatrix)
    assert np.allclose(res_const.petscmat[:, :], 4.0 * M1.petscmat[:, :])


def test_matrix_scalar_division(matrices):
    M1, _, _, _ = matrices

    # Test scalar division
    matdiv = M1 / 2.0
    res9 = assemble(matdiv)
    assert isinstance(matdiv, ufl.FormSum)
    assert isinstance(res9, matrix.AssembledMatrix)
    assert np.allclose(res9.petscmat[:, :], 0.5 * M1.petscmat[:, :])

    # Test division by Constant
    c = Constant(4.0)
    matdiv_const = M1 / c
    res_const = assemble(matdiv_const)
    assert isinstance(matdiv_const, ufl.FormSum)
    assert isinstance(res_const, matrix.AssembledMatrix)
    assert np.allclose(res_const.petscmat[:, :], 0.25 * M1.petscmat[:, :])


def test_matrix_negation(matrices):
    M1, _, _, _ = matrices

    matneg_symb = -M1
    res_neg = assemble(matneg_symb)
    assert isinstance(matneg_symb, ufl.FormSum)
    assert isinstance(res_neg, matrix.AssembledMatrix)
    assert np.allclose(res_neg.petscmat[:, :], -1 * M1.petscmat[:, :])

    matneg2_symb = -matneg_symb
    res_neg2 = assemble(matneg2_symb)
    isinstance(matneg2_symb, ufl.FormSum)
    isinstance(res_neg2, matrix.AssembledMatrix)
    assert np.allclose(res_neg2.petscmat[:, :], M1.petscmat[:, :])


def test_matrix_vector_product(matrices):
    M1, _, I1, I12 = matrices

    V1 = M1.arguments()[0].function_space()

    f = Function(V1).assign(1.0)
    matvec = I1 @ f
    assert isinstance(matvec, ufl.Action)
    res4 = assemble(matvec)
    assert isinstance(res4, Function)
    assert np.allclose(res4.dat.data[:], f.dat.data[:])

    # test vector-matrix product
    x, y = SpatialCoordinate(V1.mesh())
    f = Function(V1).interpolate(x + y)
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for @: 'Function' and 'AssembledMatrix'"):
        f @ I12


def test_cofunction_matrix_product(matrices):
    M1, _, I1, I12 = matrices

    V1 = M1.arguments()[0].function_space()
    V2 = I12.arguments()[0].function_space().dual()

    f = assemble(conj(TestFunction(V2)) * dx)  # Cofunction in V2*
    vecmat = f @ I12  # adjoint interpolation from V2^* to V1^*
    assert isinstance(vecmat, ufl.Action)
    res5 = assemble(vecmat)
    assert isinstance(res5, Cofunction)
    res5_comp = assemble(conj(TestFunction(V1)) * dx)
    assert np.allclose(res5.dat.data_ro[:], res5_comp.dat.data_ro[:])

    I12_adj = assemble(adjoint(I12))
    vecmat = I12_adj @ f
    assert isinstance(vecmat, ufl.Action)
    res6 = assemble(vecmat)
    assert isinstance(res6, Cofunction)
    res6_comp = assemble(conj(TestFunction(V1)) * dx)
    assert np.allclose(res6.dat.data_ro[:], res6_comp.dat.data_ro[:])
