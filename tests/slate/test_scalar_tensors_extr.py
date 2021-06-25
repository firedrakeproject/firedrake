import numpy as np
from math import ceil

from firedrake import *


def test_constant_one_tensor():
    mesh = ExtrudedMesh(UnitIntervalMesh(5), 5)
    one = Constant(1, domain=mesh)
    assert np.allclose(assemble(Tensor(one * dx)), 1.0)


def test_mass_matrix_variable_layers_extrusion():
    # construct variable layer mesh with height increasing from H1 to H2
    L = 50
    H1 = 2.
    H2 = 42.
    dx_ = 5.0
    nx = round(L/dx_)
    dy_ = 2.0
    tiny_dy = 0.01

    # create mesh
    mesh1d = IntervalMesh(nx, L)
    layers = []
    cell = 0
    xr = 0
    for i in range(nx):
        xr += dx_  # x of rhs of column (assumed to be the higher one)
        height = H1 + xr/L * (H2-H1)
        ncells = ceil(height/dy_)
        layers.append([0, ncells])
        cell += ncells

    mesh = ExtrudedMesh(mesh1d, layers, layer_height=dy_)
    # move top nodes to create continuous, piecewise linear top boundary
    # with height increasing from H1 to H2
    x = mesh.coordinates.dat.data_ro[:, 0]
    y = mesh.coordinates.dat.data_ro[:, 1]
    # left top nodes is moved up from H1 to H1+tiny_dy, to avoid zero edge on boundary
    height = np.maximum(H1 + x/L * (H2-H1), H1+tiny_dy)
    mesh.coordinates.dat.data[:, 1] = np.minimum(height, y)
    V = FunctionSpace(mesh, "DG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)

    A1 = assemble(Tensor(v*u*dx)).M.values
    A2 = assemble(v*u*dx).M.values
    A3 = assemble(Tensor(v*u*dx).inv).M.values

    # check A1==A2
    assert np.allclose(A1, A2, rtol=1e-12)

    # check A2*A3==Identity
    assert np.allclose(np.matmul(A2, A3), np.eye(*A2.shape), rtol=1e-12)
