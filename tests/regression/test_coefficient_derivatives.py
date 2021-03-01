from firedrake import *
import numpy as np
import pytest


@pytest.mark.skipif(utils.complex_mode, reason="Not complex differentiable")
def test_coefficient_derivatives():
    m = UnitSquareMesh(3, 3)
    x = SpatialCoordinate(m)
    V = FunctionSpace(m, "CG", 1)
    f = Function(V)
    g = Function(V)

    f.interpolate(1 + x[0] + x[1])
    g.assign(f + 1)

    # Derivative of g wrt to f is 1.
    cd = {g: 1}
    phi = (f + g**2)*dx
    v = TestFunction(V)
    manual = (1 + 2*g)*v*dx

    wrong = derivative(phi, f)
    correct = derivative(phi, f, coefficient_derivatives=cd)

    assert np.allclose(assemble(wrong).dat.data_ro, assemble(v*dx).dat.data_ro)

    assert np.allclose(assemble(manual).dat.data_ro, assemble(correct).dat.data_ro)
