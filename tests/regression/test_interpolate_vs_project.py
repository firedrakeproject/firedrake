import numpy as np
import pytest
from firedrake import *
from firedrake.__future__ import *


@pytest.fixture(params=["square", "cube"], scope="module")
def mesh(request):
    if request.param == "square":
        return SquareMesh(2, 2, 2)
    elif request.param == "cube":
        return CubeMesh(2, 2, 2, 2)


@pytest.fixture(params=[("CG", 2, FunctionSpace),
                        ("CG", 2, VectorFunctionSpace),
                        ("CG", 2, TensorFunctionSpace),
                        ("N1curl", 2, FunctionSpace),
                        ("N2curl", 2, FunctionSpace),
                        ("N1div", 2, FunctionSpace),
                        ("N2div", 2, FunctionSpace),
                        ("BDM", 2, VectorFunctionSpace),
                        ("N1curl", 2, VectorFunctionSpace),
                        ("N1div", 2, VectorFunctionSpace),
                        ("Regge", 1, FunctionSpace)],
                ids=lambda x: "%s(%s%s)" % (x[2].__name__, x[0], x[1]))
def V(request, mesh):
    space, degree, typ = request.param
    return typ(mesh, space, degree)


def test_interpolate_vs_project(V):
    mesh = V.mesh()
    dim = mesh.geometric_dimension()
    if dim == 2:
        x, y = SpatialCoordinate(mesh)
    elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)

    shape = V.value_shape
    if dim == 2:
        if len(shape) == 0:
            expression = x + y
        elif len(shape) == 1:
            expression = as_vector([x, y])
        elif len(shape) == 2:
            expression = as_tensor(([x, y], [x, y]))
    elif dim == 3:
        if len(shape) == 0:
            expression = x + y + z
        elif len(shape) == 1:
            expression = as_vector([x, y, z])
        elif len(shape) == 2:
            expression = as_tensor(([x, y, z], [x, y, z], [x, y, z]))

    f = assemble(interpolate(expression, V))
    expect = project(expression, V)
    assert np.allclose(f.dat.data, expect.dat.data, atol=1e-06)
