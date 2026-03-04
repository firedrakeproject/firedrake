from firedrake import *
import numpy as np
import scipy
import pytest


@pytest.mark.skipif(utils.complex_mode, reason="Complex bessel functions are not implemented.")
def test_hypergeometric_function():
    sp_hyp2f1 = scipy.special.hyp2f1
    a = 1
    b = 1
    c = 1
    mesh = UnitDiskMesh(3)
    V = FunctionSpace(mesh, "CG", 1)

    x, y = SpatialCoordinate(mesh)

    expr = x / 10
    uexact = hyp2f1(a, b, c, expr)
    assert np.allclose(assemble(Function(V).interpolate(uexact)).dat.data,
                       sp_hyp2f1(a, b, c, assemble(Function(V).interpolate(expr)).dat.data))

    expr = sqrt(x**2+y**2) / 10
    uexact = hyp2f1(a, b, c, expr)
    assert np.allclose(assemble(Function(V).interpolate(uexact)).dat.data,
                       sp_hyp2f1(a, b, c, assemble(Function(V).interpolate(expr)).dat.data))

    expr = sqrt(x*x + y*y) / 10
    uexact = hyp2f1(a, b, c, expr)
    assert np.allclose(assemble(Function(V).interpolate(uexact)).dat.data,
                       sp_hyp2f1(a, b, c, assemble(Function(V).interpolate(expr)).dat.data))
