"""
This demo verifies that the various FEEC operators can reproduce the
Betti numbers of the extruded interval, for which the topology is
rather boring, it could become more interesting if we had periodic.

It also verifies that the various FEEC operators with strong Dirichlet
boundary conditions can reproduce the Betti numbers of the extruded
interval, obtained from Poincare duality, which says that the
dimension of the kth cohomology group with Dirichlet boundary
conditions is equal to the dimension of the (n-k)th cohomology group
without boundary conditions.
"""
import numpy.linalg as linalg
import numpy
from firedrake import *
import pytest
from common import *


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((("CG", 1), ("DG", 0)),
                           (("CG", 1), ("DG", 0)))])
def test_betti0(horiz_complex, vert_complex):
    """
    Verify that the 0-form Hodge Laplacian has kernel of dimension
    equal to the 0th Betti number of the extruded mesh, i.e. 1.  Also
    verify that the 0-form Hodge Laplacian with Dirichlet boundary
    conditions has kernel of dimension equal to the 2nd Betti number
    of the extruded mesh, i.e. 0.
    """
    U0, U1 = horiz_complex
    V0, V1 = vert_complex

    m = UnitIntervalMesh(5)
    mesh = ExtrudedMesh(m, layers=5, layer_height=0.25)
    U0 = FiniteElement(U0[0], "interval", U0[1])
    V0 = FiniteElement(V0[0], "interval", V0[1])

    W0_elt = OuterProductElement(U0, V0)
    W0 = FunctionSpace(mesh, W0_elt)

    u = TrialFunction(W0)
    v = TestFunction(W0)

    L = assemble(inner(grad(u), grad(v))*dx)
    uvecs, s, vvecs = linalg.svd(L.M.values)
    nharmonic = sum(s < 1.0e-5)
    assert(nharmonic == 1)

    bc0 = DirichletBC(W0, 0., [1, 2])
    L = assemble(inner(grad(u), grad(v))*dx, bcs=[bc0])
    uvecs, s, vvecs = linalg.svd(L.M.values)
    nharmonic = sum(s < 1.0e-5)
    assert(nharmonic == 0)


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((("CG", 1), ("DG", 0)),
                           (("CG", 1), ("DG", 0)))])
def test_betti1(horiz_complex, vert_complex):
    """
    Verify that the 1-form Hodge Laplacian has kernel of dimension
    equal to the 1st Betti number of the extruded mesh, i.e. 0.  Also
    verify that the 1-form Hodge Laplacian with Dirichlet boundary
    conditions has kernel of dimension equal to the 2nd Betti number
    of the extruded mesh, i.e. 0.
    """
    U0, U1 = horiz_complex
    V0, V1 = vert_complex

    m = UnitIntervalMesh(5)
    mesh = ExtrudedMesh(m, layers=5, layer_height=0.25)
    U0 = FiniteElement(U0[0], "interval", U0[1])
    U1 = FiniteElement(U1[0], "interval", U1[1])
    V0 = FiniteElement(V0[0], "interval", V0[1])
    V1 = FiniteElement(V1[0], "interval", V1[1])

    W0_elt = OuterProductElement(U0, V0)

    W1_a = HDiv(OuterProductElement(U1, V0))
    W1_b = HDiv(OuterProductElement(U0, V1))
    W1_elt = W1_a + W1_b

    W0 = FunctionSpace(mesh, W0_elt)
    W1 = FunctionSpace(mesh, W1_elt)

    W = W0*W1
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    L = assemble((sigma*tau - inner(rot(tau), u) + inner(rot(sigma), v) +
                  div(u)*div(v))*dx)

    dW0 = W0.dof_count
    dW1 = W1.dof_count

    A = numpy.zeros((dW0+dW1, dW0+dW1))
    A[:dW0, :dW0] = L.M[0, 0].values
    A[:dW0, dW0:dW0+dW1] = L.M[0, 1].values
    A[dW0:dW0+dW1, :dW0] = L.M[1, 0].values
    A[dW0:dW0+dW1, dW0:dW0+dW1] = L.M[1, 1].values

    uvecs, s, vvecs = linalg.svd(A)

    nharmonic = sum(s < 1.0e-5)
    assert(nharmonic == 0)

    bc0 = [DirichletBC(W.sub(0), 0., x) for x in [1, 2, "top", "bottom"]]
    bc1 = [DirichletBC(W.sub(1), Expression(("0.", "0.")), x)
           for x in [1, 2, "top", "bottom"]]
    L0 = assemble((sigma*tau - inner(rot(tau), u) + inner(rot(sigma), v) +
                   div(u)*div(v))*dx, bcs=(bc0 + bc1))

    A0 = numpy.zeros((dW0+dW1, dW0+dW1))
    A0[:dW0, :dW0] = L0.M[0, 0].values
    A0[:dW0, dW0:dW0+dW1] = L0.M[0, 1].values
    A0[dW0:dW0+dW1, :dW0] = L0.M[1, 0].values
    A0[dW0:dW0+dW1, dW0:dW0+dW1] = L0.M[1, 1].values

    u, s, v = linalg.svd(A0)

    nharmonic = sum(s < 1.0e-5)
    assert(nharmonic == 0)


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((("CG", 1), ("DG", 0)),
                           (("CG", 1), ("DG", 0)))])
def test_betti2(horiz_complex, vert_complex):
    """
    Verify that the 2-form Hodge Laplacian has kernel of dimension
    equal to the 2nd Betti number of the extruded mesh, i.e. 0.  Also
    verify that the 2-form Hodge Laplacian with Dirichlet boundary
    conditions has kernel of dimension equal to the 0th Betti number
    of the extruded mesh, i.e. 1.
    """
    U0, U1 = horiz_complex
    V0, V1 = vert_complex

    m = UnitIntervalMesh(5)
    mesh = ExtrudedMesh(m, layers=5, layer_height=0.25)
    U0 = FiniteElement(U0[0], "interval", U0[1])
    U1 = FiniteElement(U1[0], "interval", U1[1])
    V0 = FiniteElement(V0[0], "interval", V0[1])
    V1 = FiniteElement(V1[0], "interval", V1[1])

    W1_a = HDiv(OuterProductElement(U1, V0))
    W1_b = HDiv(OuterProductElement(U0, V1))
    W1_elt = W1_a + W1_b
    W2_elt = OuterProductElement(U1, V1)

    W1 = FunctionSpace(mesh, W1_elt)
    W2 = FunctionSpace(mesh, W2_elt)

    W = W1*W2

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    L = assemble((inner(sigma, tau) - div(tau)*u + div(sigma)*v)*dx)

    bc1 = [DirichletBC(W.sub(0), Expression(("0.", "0.")), x)
           for x in [1, 2, "top", "bottom"]]
    L0 = assemble((inner(sigma, tau) - div(tau)*u + div(sigma)*v)*dx, bcs=bc1)

    dW1 = W1.dof_count
    dW2 = W2.dof_count

    A = numpy.zeros((dW1+dW2, dW1+dW2))
    A[:dW1, :dW1] = L.M[0, 0].values
    A[:dW1, dW1:dW1+dW2] = L.M[0, 1].values
    A[dW1:dW1+dW2, :dW1] = L.M[1, 0].values
    A[dW1:dW1+dW2, dW1:dW1+dW2] = L.M[1, 1].values

    u, s, v = linalg.svd(A)

    nharmonic = sum(s < 1.0e-5)
    assert(nharmonic == 0)

    A0 = numpy.zeros((dW1+dW2, dW1+dW2))
    A0[:dW1, :dW1] = L0.M[0, 0].values
    A0[:dW1, dW1:dW1+dW2] = L0.M[0, 1].values
    A0[dW1:dW1+dW2, :dW1] = L0.M[1, 0].values
    A0[dW1:dW1+dW2, dW1:dW1+dW2] = L0.M[1, 1].values

    u, s, v = linalg.svd(A0)

    nharmonic = sum(s < 1.0e-5)
    assert(nharmonic == 1)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
