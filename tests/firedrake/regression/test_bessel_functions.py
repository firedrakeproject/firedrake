from firedrake import *
from scipy.special import jn
import numpy as np
import pytest


@pytest.mark.skipif(utils.complex_mode, reason="Complex bessel functions are not implemented.")
def test_bessel_functions():

    mesh = UnitDiskMesh(3)
    V = FunctionSpace(mesh, "CG", 1)

    x, y = SpatialCoordinate(mesh)

    expr = sqrt(x**2+y**2)
    uexact = bessel_J(1, expr)
    assert np.allclose(assemble(Function(V).interpolate(uexact)).dat.data,
                       jn(1, assemble(Function(V).interpolate(expr)).dat.data))

    expr = sqrt(x*x + y*y)
    uexact = bessel_J(1, expr)
    assert np.allclose(assemble(Function(V).interpolate(uexact)).dat.data,
                       jn(1, assemble(Function(V).interpolate(expr)).dat.data))

    expr = x
    uexact = bessel_J(1, expr)
    assert np.allclose(assemble(Function(V).interpolate(uexact)).dat.data,
                       jn(1, assemble(Function(V).interpolate(expr)).dat.data))
