from firedrake import *
import pytest
import numpy as np


@pytest.fixture
def mesh():
    return UnitSquareMesh(2, 2)


def test_lump_scheme(mesh):
    # [test_lump_scheme 1]
    degree = 3
    V = FunctionSpace(mesh, "KMV", degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v) * dx(scheme="KMV", degree=degree)
    # [test_lump_scheme 2]

    A = assemble(a)
    d = assemble(inner(1, v)*dx)
    with d.dat.vec as dvec:
        Dmat = PETSc.Mat().createDiagonal(dvec)

    Bmat = A.petscmat.copy()
    Bmat.axpy(-1, Dmat)
    assert np.allclose(Bmat.norm(PETSc.NormType.FROBENIUS), 0)


@pytest.mark.parametrize("quad_scheme", (None, "default", "canonical"))
def test_quadrature_space(mesh, quad_scheme):
    quad_degree = 4
    # [test_quadrature_space 1]
    Q = FunctionSpace(mesh, "Quadrature", degree=quad_degree, quad_scheme=quad_scheme)
    # [test_quadrature_space 2]

    f = sum(SpatialCoordinate(mesh)) ** 2
    q = Function(Q).interpolate(f)

    assert norm(q - f) < 1E-12
