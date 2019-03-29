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

    F = (inner(u, v) + v[1]*p)*dx

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


def test_split_function_derivative():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "DG", 0)

    W = V*V

    f = Function(W)

    u, p = f.split()

    f.assign(1)

    phi = u**2*dx + p*dx

    actual = assemble(derivative(phi, u))
    expect = assemble(2*TestFunction(V)*dx)

    assert np.allclose(actual.dat.data_ro, expect.dat.data_ro)

    actual = assemble(derivative(derivative(phi, u), u))
    expect = assemble(2*TestFunction(V)*TrialFunction(V)*dx)

    assert np.allclose(actual.M.values, expect.M.values)
