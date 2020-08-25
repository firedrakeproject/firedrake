import pytest
from firedrake import *
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

    u, p = f.split()

    phi = u*dx + p*dx

    with pytest.raises(ValueError):
        derivative(phi, f)


@pytest.mark.skipif(utils.complex_mode, reason="u**2 not complex Gateaux differentiable.")
def test_split_function_derivative():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "DG", 0)

    W = V*V

    f = Function(W)

    u, p = f.split()

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
