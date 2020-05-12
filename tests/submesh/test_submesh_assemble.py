# Simple Poisson equation
# =========================

import numpy as np
import math
import pytest

from firedrake import *
from firedrake.cython import dmplex
from firedrake.petsc import PETSc
from pyop2.datatypes import IntType
import ufl


def test_submesh_poisson_cell_error2():

    msh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    msh.init()

    msh.markSubdomain("half_domain", 222, "cell", None, geometric_expr = lambda x: x[0] > 0.9999)

    submsh = SubMesh(msh, "half_domain", 222, "cell")

    V0 = FunctionSpace(msh, "CG", 1)
    V1 = FunctionSpace(submsh, "CG", 1)

    W = V0 * V1

    w = Function(W)
    u0, u1 = TrialFunctions(W)
    v0, v1 = TestFunctions(W)

    f0 = Function(V0)
    x0, y0 = SpatialCoordinate(msh)
    f0.interpolate(-8.0 * pi * pi * cos(x0 * pi * 2) * cos(y0 * pi * 2))

    dx0 = Measure("cell", domain=msh)
    dx1 = Measure("cell", domain=submsh)


    a = inner(u0, v0) * dx0 + inner(u0 - u1, v1) * dx1
    L = inner(f0, v0) * dx0

    mat = assemble(a)
    print(mat.M[1][0].values)

    m00 = np.array([[2./9. , 1./9. , 1./36., 1./18., 1./18., 1./36.],
                    [1./9. , 2./9. , 1./18., 1./36., 1./36., 1./18.],
                    [1./36., 1./18., 1./9. , 1./18., 0.    , 0.    ],
                    [1./18., 1./36., 1./18., 1./9. , 0.    , 0.    ],
                    [1./18., 1./36., 0.    , 0.    , 1./9. , 1./18.],
                    [1./36., 1./18., 0.    , 0.    , 1./18., 1./9. ]])

    m01 = np.array([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])

    m10 = np.array([[1./18., 1./9. , 1./18., 1./36., 0., 0.],
                    [1./9. , 1./18., 1./36., 1./18., 0., 0.],
                    [1./18., 1./36., 1./18., 1./9. , 0., 0.],
                    [1./36., 1./18., 1./9. , 1./18., 0., 0.]])

    m11 = np.array([[-1./9. , -1./18., -1./36., -1./18.],
                    [-1./18., -1./9. , -1./18., -1./36.],
                    [-1./36., -1./18., -1./9. , -1./18.],
                    [-1./18., -1./36., -1./18., -1./9. ]])

    assert(np.allclose(mat.M[0][0].values, m00))
    assert(np.allclose(mat.M[0][1].values, m01))
    assert(np.allclose(mat.M[1][0].values, m10))
    assert(np.allclose(mat.M[1][1].values, m11))
