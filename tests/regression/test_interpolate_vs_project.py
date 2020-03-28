import numpy as np
import pytest
from firedrake import *

'''
format for funcspaces = (space, exp_type, functionspace_type)
choices for exp_type: "scalar", "vector", "tensor"
choices for functionspac_type:
 "0" -> FunctionSpace
 "1" -> VectorFunctionSpace
 "2" -> TensorFunctionSpace
'''

funcspaces = [
    ("CG", "scalar", "0"),
    ("CG", "vector", "1"),
    ("CG", "tensor", "2"),
    ("N1curl", "vector", "0"),
    ("N2curl", "vector", "0"),
    ("N1div", "vector", "0"),
    ("N2div", "vector", "0"),
    ("BDM", "tensor", "1"),
    ("Regge", "tensor", "0"),
]


@pytest.mark.parametrize("space , exp_type, functionspace_type", funcspaces)
@pytest.mark.parametrize("dim", [2, 3])
def test_interpolate_vs_project(space, exp_type, functionspace_type, dim):
    if dim == 2:
        mesh = SquareMesh(2, 2, 2)
        x, y = SpatialCoordinate(mesh)
    elif dim == 3:
        mesh = CubeMesh(2, 2, 2, 2)
        x, y, z = SpatialCoordinate(mesh)

    if functionspace_type == "0":
        V = FunctionSpace(mesh, space, 2)
    elif functionspace_type == "1":
        V = VectorFunctionSpace(mesh, space, 1)
    elif functionspace_type == "2":
        V = TensorFunctionSpace(mesh, space, 1)

    if dim == 2:
        if exp_type == "scalar":
            expression = x + y
        elif exp_type == "vector":
            expression = as_vector([x, y])
        elif exp_type == "tensor":
            expression = as_tensor(([x, y], [x, y]))
    elif dim == 3:
        if exp_type == "scalar":
            expression = x + y + z
        elif exp_type == "vector":
            expression = as_vector([x, y, z])
        elif exp_type == "tensor":
            expression = as_tensor(([x, y, z], [x, y, z], [x, y, z]))

    f = interpolate(expression, V)
    expect = project(expression, V)
    assert np.allclose(f.dat.data, expect.dat.data, atol=1e-06)
