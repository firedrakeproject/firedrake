from firedrake import *
from scipy.special import hyp2f1 as sp_hyp2f1
import numpy as np
import pytest


def test_hypergeometric_function():
    mesh = UnitDiskMesh(3)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    w = Function(V)
    x, y = SpatialCoordinate(mesh)

    expressions = [((1/3, 1/2, 1), Constant(0.9999) * x),
                   ((1/2, 1/2, 1), Constant(0.9999) * sqrt(x**2+y**2)),
                   ((-1/2, 1/2, 1), Constant(0.4999) * sqrt(x*x + y*y))]

    for (a, b, c), expr in expressions:
        u.interpolate(hyp2f1(a, b, c, expr))
        result = u.dat.data
        w.interpolate(expr)
        expect = sp_hyp2f1(a, b, c, w.dat.data)
        assert np.allclose(result, expect)
