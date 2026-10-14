import pytest
from firedrake import *
from firedrake.formmanipulation import ExtractSubBlock
import numpy as np


def test_assemble_split_derivative():
    """Assemble the derivative of a form with a zero block."""
    mesh = UnitSquareMesh(1, 1)
    V1 = FunctionSpace(mesh, "BDM", 1, name="V")
    V2 = FunctionSpace(mesh, "DG", 0, name="P")
    W = V1 * V2

    x = Function(W)
    u, p = split(x)
    v, q = TestFunctions(W)

    F = (inner(u, v) + inner(p, v[1])) * dx

    assert assemble(derivative(F, x))


def test_function_split_raises():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "DG", 0)

    W = V*V

    f = Function(W)

    u, p = f.subfunctions

    phi = u*dx + p*dx

    with pytest.raises(ValueError):
        derivative(phi, f)


@pytest.mark.skipif(utils.complex_mode, reason="u**2 not complex Gateaux differentiable.")
def test_split_function_derivative():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "DG", 0)

    W = V*V

    f = Function(W)

    u, p = f.subfunctions

    f.assign(1)

    phi = u**2*dx + p*dx

    actual = assemble(derivative(phi, u))
    expect = assemble(2*conj(TestFunction(V))*dx)

    assert np.allclose(actual.dat.data_ro, expect.dat.data_ro)

    actual = assemble(derivative(derivative(phi, u), u))
    expect = assemble(2 * inner(TrialFunction(V), TestFunction(V)) * dx)

    assert np.allclose(actual.M.values, expect.M.values)


@pytest.mark.skipif(utils.complex_mode, reason="inner(grad(u), grad(u)) not complex Gateaux differentiable.")
def test_assemble_split_mixed_derivative():
    """Assemble the derivative of a form wrt part of mixed function."""
    mesh = UnitSquareMesh(1, 1)
    V1 = FunctionSpace(mesh, "P", 1)
    W = V1 * V1

    x = Function(W)
    u, p = split(x)
    v, q = TestFunctions(W)

    I = 0.5*inner(grad(u), grad(u))*dx
    F = derivative(I, u, v)
    J = derivative(F, u, TrialFunctions(W)[0])

    x.sub(0).interpolate(SpatialCoordinate(mesh)[0])

    actual = assemble(F)
    expect = assemble(inner(grad(u), grad(v))*dx)

    assert np.allclose(actual.dat.data_ro, expect.dat.data_ro)

    actual = assemble(J, mat_type="aij")
    u_trial, _ = TrialFunctions(W)
    expect = assemble(inner(grad(u_trial), grad(v))*dx, mat_type="aij")

    assert np.allclose(actual.M.values, expect.M.values)


def test_split_coordinate_derivative():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "P", 1)
    Q = FunctionSpace(mesh, "DP", 0)
    W = V*Q
    v = TestFunction(W)
    w = Function(W)
    x = SpatialCoordinate(mesh)
    J = derivative(inner(v, w)*dx, x)
    splitter = ExtractSubBlock()

    J00 = splitter.split(J, (0, 0))
    expect = derivative(inner(as_vector([TestFunction(V), 0]), w)*dx, x)

    assert J00.signature() == expect.signature()


def test_split_coefficient_not_argument():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "P", 1)
    Q = FunctionSpace(mesh, "DP", 0)
    W = V*Q
    w = Function(W)
    wr = Function(W)
    J = derivative(derivative(inner(grad(w), grad(w))*dx, w), w, wr)
    splitter = ExtractSubBlock()

    J00 = splitter.split(J, (0, 0))

    expect = derivative(derivative(inner(grad(w), grad(w))*dx,
                                   w,
                                   as_vector([TestFunction(V), 0])),
                        w, wr)
    assert J00.signature() == expect.signature()


@pytest.mark.parallel([1, 3])
def test_split_adjoint_action():
    V = FunctionSpace(UnitSquareMesh(4, 4), "CG", 1)
    W = V * V
    u, _ = TrialFunctions(W)
    c = Cofunction(V.dual()).assign(1)
    F = action(adjoint(interpolate(u, V)), c)
    splitter = ExtractSubBlock()

    F0 = splitter.split(F, (0,))
    assert F0.arguments() == (TestFunction(V),)
    actual = assemble(F0)
    expected = assemble(interpolate(TestFunction(V), c))
    assert np.allclose(actual.dat.data_ro, expected.dat.data_ro)

    F1 = splitter.split(F, (1,))
    assert F1 == 0
    assert F1.arguments() == (TestFunction(V),)


@pytest.mark.parallel([1, 3])
def test_split_action_composition():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 2)
    W = V * Q
    u, p = TrialFunctions(W)
    I = interpolate(as_vector((u + p, 2*u + 3*p)), W)
    mass = inner(TrialFunction(W), TestFunction(W))*dx
    A = action(adjoint(I), action(mass, I))
    matrix = assemble(A, mat_type="aij").petscmat
    ises = W.dof_dset.field_ises
    splitter = ExtractSubBlock()

    for i, j in np.ndindex(2, 2):
        block = splitter.split(A, (i, j))
        assert block.arguments() == (TestFunction(W[i].collapse()), TrialFunction(W[j].collapse()))
        actual = assemble(block, mat_type="aij").petscmat
        expected = matrix.createSubMatrix(ises[i], ises[j])
        actual.axpy(-1, expected, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        assert actual.norm(PETSc.NormType.NORM_FROBENIUS) < 1e-12
