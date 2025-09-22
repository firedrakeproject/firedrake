from firedrake import *
import numpy as np
from math import ceil, sqrt


def test_variable_layers_exterior_integrals(b1=0):
    # setup 2d vert. slice domain of length L
    # flat bottom, and sloping top with
    # height H1 on the left smaller than height H2 on the right
    L = 100
    H1 = 2.
    H2 = 42.

    dx = 5.0
    nx = round(L/dx)
    dy = 2.0
    mesh1d = IntervalMesh(nx, L)
    layers = []
    cell = 0
    xr = 0
    for i in range(nx):
        xr += dx  # x of rhs of column (assumed to be the higher one)
        height = H1 + xr/L * (H2-H1)
        ncells = ceil(height/dy)
        layers.append([0, ncells])
        cell += ncells

    mesh = ExtrudedMesh(mesh1d, layers, layer_height=dy)
    x = mesh.coordinates.dat.data_ro[:, 0]
    y = mesh.coordinates.dat.data_ro[:, 1]
    mesh.coordinates.dat.data[:, 1] = np.minimum(H1 + x/L * (H2-H1), y)

    # check for correct lenghts of four sides:
    np.testing.assert_allclose(assemble(Constant(1.0)*ds_b(domain=mesh)), L)
    np.testing.assert_allclose(assemble(Constant(1.0)*ds_t(domain=mesh)), sqrt(L**2+(H2-H1)**2))
    np.testing.assert_allclose(assemble(Constant(1.0)*ds_v(1, domain=mesh)), H1)
    np.testing.assert_allclose(assemble(Constant(1.0)*ds_v(2, domain=mesh)), H2)
