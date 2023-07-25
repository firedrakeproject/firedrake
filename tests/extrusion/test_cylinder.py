from firedrake import *
from firedrake.utils import ScalarType
import numpy as np
import pytest


@pytest.mark.parametrize("degree", [1, 2])
def test_area(degree):
    expected_conv = degree * 2
    len = 6
    errors = np.zeros(len)
    for i in range(len):
        m = CircleManifoldMesh(2**(i+3), degree=degree)
        mesh = ExtrudedMesh(m, layers=2**i, layer_height=1.0/(2**i))
        fs = FunctionSpace(mesh, "DG", 0)
        f = Function(fs).assign(1)
        # surface area is 2*pi*r*h = 2*pi
        errors[i] = np.abs(assemble(f*dx) - 2*np.pi)

    # area converges quadratically to 2*pi
    for i in range(len-1):
        assert ln(errors[i]/errors[i+1])/ln(2) > 0.98 * expected_conv


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((("CG", 1), ("DG", 0)),
                           (("CG", 1), ("DG", 0)))])
def test_betti0_cylinder(horiz_complex, vert_complex):
    """
    Verify that the 0-form Hodge Laplacian has kernel of dimension
    equal to the 0th Betti number of the periodic extruded interval,
    i.e. 1.  Also verify that the 0-form Hodge Laplacian with
    Dirichlet boundary conditions has kernel of dimension equal to the
    2nd Betti number of the extruded mesh, i.e. 0.
    """
    U0, U1 = horiz_complex
    V0, V1 = vert_complex

    m = CircleManifoldMesh(5)
    mesh = ExtrudedMesh(m, layers=4, layer_height=0.25)
    U0 = FiniteElement(U0[0], "interval", U0[1])
    V0 = FiniteElement(V0[0], "interval", V0[1])

    W0_elt = TensorProductElement(U0, V0)
    W0 = FunctionSpace(mesh, W0_elt)

    u = TrialFunction(W0)
    v = TestFunction(W0)

    L = assemble(inner(grad(u), grad(v))*dx)
    uvecs, s, vvecs = np.linalg.svd(L.M.values)
    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 1

    bcs = [DirichletBC(W0, 0., x) for x in ["top", "bottom"]]
    L = assemble(inner(grad(u), grad(v))*dx, bcs=bcs)
    uvecs, s, vvecs = np.linalg.svd(L.M.values)
    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 0


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((("CG", 1), ("DG", 0)),
                           (("CG", 1), ("DG", 0)))])
def test_betti1_cylinder(horiz_complex, vert_complex):
    """
    Verify that the 1-form Hodge Laplacian has kernel of dimension
    equal to the 1st Betti number of the periodic extruded interval,
    i.e. 1.  Also verify that the 1-form Hodge Laplacian with
    Dirichlet boundary conditions has kernel of dimension equal to the
    2nd Betti number of the periodic extruded interval mesh, i.e. 1.

    """
    U0, U1 = horiz_complex
    V0, V1 = vert_complex

    m = CircleManifoldMesh(5)
    mesh = ExtrudedMesh(m, layers=4, layer_height=0.25)
    xs = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(as_vector((xs[0], xs[1], 0.0)))
    U0 = FiniteElement(U0[0], "interval", U0[1])
    U1 = FiniteElement(U1[0], "interval", U1[1])
    V0 = FiniteElement(V0[0], "interval", V0[1])
    V1 = FiniteElement(V1[0], "interval", V1[1])

    W0_elt = TensorProductElement(U0, V0)

    W1_a = HDiv(TensorProductElement(U1, V0))
    W1_b = HDiv(TensorProductElement(U0, V1))
    W1_elt = W1_a + W1_b

    W0 = FunctionSpace(mesh, W0_elt)
    W1 = FunctionSpace(mesh, W1_elt)

    outward_normal = Function(VectorFunctionSpace(mesh, "DG", 0)).interpolate(
        as_vector((xs[0]/sqrt(xs[0]*xs[0] + xs[1]*xs[1]), xs[1]/sqrt(xs[0]*xs[0] + xs[1]*xs[1]), Constant(0.0))))

    W = W0*W1
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    L = assemble((inner(sigma, tau) - inner(u, cross(outward_normal, grad(tau))) + inner(cross(outward_normal, grad(sigma)), v)
                  + inner(div(u), div(v)))*dx)

    dW0 = W0.dof_count
    dW1 = W1.dof_count

    A = np.zeros((dW0+dW1, dW0+dW1), dtype=ScalarType)
    A[:dW0, :dW0] = L.M[0, 0].values
    A[:dW0, dW0:dW0+dW1] = L.M[0, 1].values
    A[dW0:dW0+dW1, :dW0] = L.M[1, 0].values
    A[dW0:dW0+dW1, dW0:dW0+dW1] = L.M[1, 1].values

    uvecs, s, vvecs = np.linalg.svd(A)

    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 1

    bc0 = [DirichletBC(W.sub(0), 0., x) for x in ["top", "bottom"]]
    bc1 = [DirichletBC(W.sub(1), as_vector((0.0, 0.0, 0.0)), x)
           for x in ["top", "bottom"]]
    L0 = assemble((inner(sigma, tau) - inner(u, cross(outward_normal, grad(tau))) + inner(cross(outward_normal, grad(sigma)), v)
                   + inner(div(u), div(v)))*dx, bcs=(bc0 + bc1))

    A0 = np.zeros((dW0+dW1, dW0+dW1), dtype=ScalarType)
    A0[:dW0, :dW0] = L0.M[0, 0].values
    A0[:dW0, dW0:dW0+dW1] = L0.M[0, 1].values
    A0[dW0:dW0+dW1, :dW0] = L0.M[1, 0].values
    A0[dW0:dW0+dW1, dW0:dW0+dW1] = L0.M[1, 1].values

    u, s, v = np.linalg.svd(A0)

    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 1


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((("CG", 1), ("DG", 0)),
                           (("CG", 1), ("DG", 0)))])
def test_betti2_cylinder(horiz_complex, vert_complex):
    """
    Verify that the 2-form Hodge Laplacian has kernel of dimension
    equal to the 2nd Betti number of the periodic extruded interval
    mesh, i.e. 0.  Also verify that the 2-form Hodge Laplacian with
    Dirichlet boundary conditions has kernel of dimension equal to the
    0th Betti number of the periodic extruded interval mesh, i.e. 1.

    """
    U0, U1 = horiz_complex
    V0, V1 = vert_complex

    m = CircleManifoldMesh(5)
    mesh = ExtrudedMesh(m, layers=4, layer_height=0.25)
    xs = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(as_vector((xs[0], xs[1], Constant(0.0))))
    U0 = FiniteElement(U0[0], "interval", U0[1])
    U1 = FiniteElement(U1[0], "interval", U1[1])
    V0 = FiniteElement(V0[0], "interval", V0[1])
    V1 = FiniteElement(V1[0], "interval", V1[1])

    W1_a = HDiv(TensorProductElement(U1, V0))
    W1_b = HDiv(TensorProductElement(U0, V1))
    W1_elt = W1_a + W1_b
    W2_elt = TensorProductElement(U1, V1)

    W1 = FunctionSpace(mesh, W1_elt)
    W2 = FunctionSpace(mesh, W2_elt)

    W = W1*W2

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    L = assemble((inner(sigma, tau) - inner(u, div(tau)) + inner(div(sigma), v))*dx)

    bc1 = [DirichletBC(W.sub(0), as_vector((0.0, 0.0, 0.0)), x)
           for x in ["top", "bottom"]]
    L0 = assemble((inner(sigma, tau) - inner(u, div(tau)) + inner(div(sigma), v))*dx, bcs=bc1)

    dW1 = W1.dof_count
    dW2 = W2.dof_count

    A = np.zeros((dW1+dW2, dW1+dW2), dtype=ScalarType)
    A[:dW1, :dW1] = L.M[0, 0].values
    A[:dW1, dW1:dW1+dW2] = L.M[0, 1].values
    A[dW1:dW1+dW2, :dW1] = L.M[1, 0].values
    A[dW1:dW1+dW2, dW1:dW1+dW2] = L.M[1, 1].values

    u, s, v = np.linalg.svd(A)

    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 0

    A0 = np.zeros((dW1+dW2, dW1+dW2), dtype=ScalarType)
    A0[:dW1, :dW1] = L0.M[0, 0].values
    A0[:dW1, dW1:dW1+dW2] = L0.M[0, 1].values
    A0[dW1:dW1+dW2, :dW1] = L0.M[1, 0].values
    A0[dW1:dW1+dW2, dW1:dW1+dW2] = L0.M[1, 1].values

    u, s, v = np.linalg.svd(A0)

    nharmonic = sum(s < 1.0e-5)
    assert nharmonic == 1
