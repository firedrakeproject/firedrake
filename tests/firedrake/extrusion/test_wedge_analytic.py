from firedrake import *
import pytest
import numpy as np


@pytest.fixture(scope='module')
def u_v():
    m = ExtrudedMesh(UnitTriangleMesh(), layers=1, layer_height=1)
    V = FunctionSpace(m, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    return u, v


def test_analytic_laplacian(u_v):
    u, v = u_v

    a = inner(grad(u), grad(v))*dx

    vals = assemble(a).M.values

    # Computed via sage:
    # sage: x, y, z = var('x', 'y', 'z')
    # sage: phi0 = (1 - x - y)*(1 - z)
    # sage: phi1 = (1 - x - y)*z
    # sage: phi2 = x*(1-z)
    # sage: phi3 = x*z
    # sage: phi4 = y*(1-z)
    # sage: phi5 = y*z
    # sage: phi = [phi0, phi1, phi2, phi3, phi4, phi5]
    # sage: def grad(f):
    # ...       return [f.derivative(x), f.derivative(y), f.derivative(z)]
    # sage: def dot(a, b):
    # ...       return sum(a_ * b_ for a_, b_ in zip(a, b))
    # sage: def laplace(phi_i, phi_j):
    # ...       return integral(integral(integral(dot(grad(phi_i), grad(phi_j)), x, 0, 1 - y), y, 0, 1), z, 0, 1)
    # sage: [[w_laplace(phi_i, phi_j) for phi_i in phi] for phi_j in phi]
    analytic = np.asarray([[5/12, 1/12, -1/8, -1/8, -1/8, -1/8],
                           [1/12, 5/12, -1/8, -1/8, -1/8, -1/8],
                           [-1/8, -1/8, 1/4, 0, 1/24, -1/24],
                           [-1/8, -1/8, 0, 1/4, -1/24, 1/24],
                           [-1/8, -1/8, 1/24, -1/24, 1/4, 0],
                           [-1/8, -1/8, -1/24, 1/24, 0, 1/4]])

    assert np.allclose(sorted(np.linalg.eigvals(vals)),
                       sorted(np.linalg.eigvals(analytic)))


def test_analytic_mass(u_v):
    u, v = u_v

    a = inner(u, v)*dx

    vals = assemble(a).M.values

    # sage: def w_mass(phi_i, phi_j):
    # ...       return integral(integral(integral(phi_i*phi_j, x, 0, 1 - y), y, 0, 1), z, 0, 1)
    # sage: [[w_mass(phi_i, phi_j) for phi_i in phi] for phi_j in phi]
    analytic = np.asarray([[1/36, 1/72, 1/72, 1/144, 1/72, 1/144],
                           [1/72, 1/36, 1/144, 1/72, 1/144, 1/72],
                           [1/72, 1/144, 1/36, 1/72, 1/72, 1/144],
                           [1/144, 1/72, 1/72, 1/36, 1/144, 1/72],
                           [1/72, 1/144, 1/72, 1/144, 1/36, 1/72],
                           [1/144, 1/72, 1/144, 1/72, 1/72, 1/36]])

    assert np.allclose(sorted(np.linalg.eigvals(vals)),
                       sorted(np.linalg.eigvals(analytic)))
