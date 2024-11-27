"""
This demo verifies that the various FEEC operators can reproduce the
Betti numbers of the 2D annulus.

It also verifies that the various FEEC operators with strong Dirichlet
boundary conditions can reproduce the Betti numbers of the 2D annulus,
obtained from Poincare duality, which says that the dimension of the
kth cohomology group with Dirichlet boundary conditions is equal to
the dimension of the (n-k)th cohomology group without boundary
conditions.
"""
from os.path import abspath, dirname, join
import numpy.linalg as linalg
import numpy
from firedrake import *
import pytest

cwd = abspath(dirname(__file__))


@pytest.fixture
def mesh():
    return Mesh(join(cwd, "..", "meshes", "annulus.msh"))


@pytest.mark.parametrize(('space'),
                         [(("CG", 1), ("RT", 1), ("DG", 0)),
                          (("CG", 2), ("RT", 2), ("DG", 1)),
                          (("CG", 3), ("RT", 3), ("DG", 2)),
                          (("CG", 2), ("BDM", 1), ("DG", 0)),
                          (("CG", 3), ("BDM", 2), ("DG", 1)),
                          (("CG", 4), ("BDM", 3), ("DG", 2)),
                          (("CG", 2, "B", 3), ("BDFM", 2), ("DG", 1))])
def test_betti0(space, mesh):
    """
    Verify that the 0-form Hodge Laplacian with strong Dirichlet
    boundary conditions has kernel of dimension equal to the 2nd Betti
    number of the annulus mesh, i.e. 0.
    """
    V0tag, V1tag, V2tag = space

    if len(V0tag) == 2:
        V0 = FunctionSpace(mesh, V0tag[0], V0tag[1])
    else:
        V0a = FiniteElement(V0tag[0], "triangle", V0tag[1])
        V0b = FiniteElement(V0tag[2], "triangle", V0tag[3])
        V0 = FunctionSpace(mesh, V0a + V0b)
    # V0 Hodge Laplacian
    u = TrialFunction(V0)
    v = TestFunction(V0)

    L = assemble(inner(nabla_grad(u), nabla_grad(v))*dx)

    bc0 = DirichletBC(V0, 0, 9)
    L0 = assemble(inner(nabla_grad(u), nabla_grad(v))*dx, bcs=[bc0])

    u, s, v = linalg.svd(L.M.values)
    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 1

    u, s, v = linalg.svd(L0.M.values)
    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 0


@pytest.mark.parametrize(('space'),
                         [(("CG", 1), ("RT", 1), ("DG", 0)),
                          (("CG", 2), ("RT", 2), ("DG", 1)),
                          (("CG", 3), ("RT", 3), ("DG", 2)),
                          (("CG", 2), ("BDM", 1), ("DG", 0)),
                          (("CG", 3), ("BDM", 2), ("DG", 1)),
                          (("CG", 4), ("BDM", 3), ("DG", 2)),
                          (("CG", 2, "B", 3), ("BDFM", 2), ("DG", 1))])
def test_betti1(space, mesh):
    """
    Verify that the 1-form Hodge Laplacian with strong Dirichlet
    boundary conditions has kernel of dimension equal to the 1st Betti
    number of the annulus mesh, i.e. 1.
    """
    V0tag, V1tag, V2tag = space

    if len(V0tag) == 2:
        V0 = FunctionSpace(mesh, V0tag[0], V0tag[1])
    else:
        V0a = FiniteElement(V0tag[0], "triangle", V0tag[1])
        V0b = FiniteElement(V0tag[2], "triangle", V0tag[3])
        V0 = FunctionSpace(mesh, V0a + V0b)

    V1 = FunctionSpace(mesh, V1tag[0], V1tag[1])

    W = V0*V1
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    L = assemble((inner(sigma, tau) - inner(u, rot(tau)) + inner(rot(sigma), v)
                  + inner(div(u), div(v))) * dx)

    bc0 = DirichletBC(W.sub(0), 0, 9)
    bc1 = DirichletBC(W.sub(1), 0, 9)
    L0 = assemble((inner(sigma, tau) - inner(u, rot(tau)) + inner(rot(sigma), v)
                   + inner(div(u), div(v))) * dx, bcs=[bc0, bc1])

    dV0 = V0.dof_count
    dV1 = V1.dof_count

    A = numpy.zeros((dV0+dV1, dV0+dV1), dtype=utils.ScalarType)
    A[:dV0, :dV0] = L.M[0, 0].values
    A[:dV0, dV0:dV0+dV1] = L.M[0, 1].values
    A[dV0:dV0+dV1, :dV0] = L.M[1, 0].values
    A[dV0:dV0+dV1, dV0:dV0+dV1] = L.M[1, 1].values

    u, s, v = linalg.svd(A)

    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 1

    dV0 = V0.dof_count
    dV1 = V1.dof_count

    A0 = numpy.zeros((dV0+dV1, dV0+dV1), dtype=utils.ScalarType)
    A0[:dV0, :dV0] = L0.M[0, 0].values
    A0[:dV0, dV0:dV0+dV1] = L0.M[0, 1].values
    A0[dV0:dV0+dV1, :dV0] = L0.M[1, 0].values
    A0[dV0:dV0+dV1, dV0:dV0+dV1] = L0.M[1, 1].values

    u, s, v = linalg.svd(A0)

    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 1


@pytest.mark.parametrize(('space'),
                         [(("CG", 1), ("RT", 1), ("DG", 0)),
                          (("CG", 2), ("RT", 2), ("DG", 1)),
                          (("CG", 3), ("RT", 3), ("DG", 2)),
                          (("CG", 2), ("BDM", 1), ("DG", 0)),
                          (("CG", 3), ("BDM", 2), ("DG", 1)),
                          (("CG", 4), ("BDM", 3), ("DG", 2)),
                          (("CG", 2, "B", 3), ("BDFM", 2), ("DG", 1))])
def test_betti2(space, mesh):
    """
    Verify that the 2-form Hodge Laplacian with strong Dirichlet
    boundary conditions has kernel of dimension equal to the 2nd Betti
    number of the annulus mesh, i.e. 1.
    """
    V0tag, V1tag, V2tag = space

    V1 = FunctionSpace(mesh, V1tag[0], V1tag[1])

    V2 = FunctionSpace(mesh, V2tag[0], V2tag[1])

    W = V1*V2

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    L = assemble((inner(sigma, tau) - inner(u, div(tau)) + inner(div(sigma), v))*dx)

    bc1 = DirichletBC(W.sub(0), 0, 9)
    L0 = assemble((inner(sigma, tau) - inner(u, div(tau)) + inner(div(sigma), v))*dx, bcs=[bc1])

    dV1 = V1.dof_count
    dV2 = V2.dof_count

    A = numpy.zeros((dV1+dV2, dV1+dV2), dtype=utils.ScalarType)
    A[:dV1, :dV1] = L.M[0, 0].values
    A[:dV1, dV1:dV1+dV2] = L.M[0, 1].values
    A[dV1:dV1+dV2, :dV1] = L.M[1, 0].values
    A[dV1:dV1+dV2, dV1:dV1+dV2] = L.M[1, 1].values

    u, s, v = linalg.svd(A)

    nharmonic = sum(s < 1.0e-5)
    print(nharmonic, V1tag[0])
    assert nharmonic == 0

    A0 = numpy.zeros((dV1+dV2, dV1+dV2), dtype=utils.ScalarType)
    A0[:dV1, :dV1] = L0.M[0, 0].values
    A0[:dV1, dV1:dV1+dV2] = L0.M[0, 1].values
    A0[dV1:dV1+dV2, :dV1] = L0.M[1, 0].values
    A0[dV1:dV1+dV2, dV1:dV1+dV2] = L0.M[1, 1].values

    u, s, v = linalg.svd(A0)

    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 1
