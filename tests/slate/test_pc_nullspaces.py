from firedrake import *
from firedrake.slate.preconditioners.sc_nullspaces import (create_trace_nullspace,
                                                           create_sc_nullspace)
import numpy as np

import pytest


@pytest.fixture(scope='module', params=[False, True])
def W(request):
    quadrilateral = request.param
    if quadrilateral:
        mesh = UnitCubedSphereMesh(2)
        mesh.init_cell_orientations(SpatialCoordinate(mesh))
        V = FunctionSpace(mesh, "RTCF", 1)
        Q = FunctionSpace(mesh, "DQ", 0)
    else:
        mesh = UnitIcosahedralSphereMesh(2)
        mesh.init_cell_orientations(SpatialCoordinate(mesh))
        V = FunctionSpace(mesh, "RT", 1)
        Q = FunctionSpace(mesh, "DG", 0)

    W = V*Q

    return W


def test_hybrid_nullspace(W):
    """Tests that the singular vector associated with the Schur
    complement generated via the Hybridization PC is identical
    (subject to tolerance and sign) with the SVD computed singular
    vector.
    """
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = (dot(sigma, tau) + div(sigma)*v + div(tau)*u)*dx
    A = assemble(a, mat_type="aij")

    T = FunctionSpace(W.mesh(), "HDiv Trace", 0)
    gamma = TestFunction(T)

    Wd = FunctionSpace(W.mesh(), MixedElement([BrokenElement(Wi.ufl_element())
                                               for Wi in W]))

    Atilde = Tensor(replace(a, dict(zip(a.arguments(),
                                        (TestFunction(Wd),
                                         TrialFunction(Wd))))))
    sigma, _ = TrialFunctions(Wd)
    K = Tensor(gamma('+') * dot(sigma, FacetNormal(W.mesh())) * dS)

    nullsp = MixedVectorSpaceBasis(W, [W[0], VectorSpaceBasis(constant=True)])
    nullsp._build_monolithic_basis()
    A.petscmat.setNullSpace(nullsp._nullspace)

    Snullsp = create_trace_nullspace(A.petscmat, -K * Atilde,
                                     W, Wd, T, COMM_WORLD)
    v = Snullsp.getVecs()[0].array_r

    S = K * Atilde.inv * K.T
    _, _, vv = np.linalg.svd(assemble(S, mat_type="aij").M.values)
    singular_vector = vv[-1]

    assert np.allclose(np.linalg.norm(v), 1.0, 1e-13)
    assert np.allclose(v.min(), v.max(), 1e-13)
    assert np.allclose(np.absolute(v),
                       np.absolute(singular_vector), 1e-13)


@pytest.mark.parametrize(('dim', 'degree'),
                         [(2, 3), (2, 4), (3, 4)])
def test_static_condensation_nullspace(dim, degree):
    if dim == 2:
        mesh = UnitSquareMesh(4, 4, quadrilateral=False)
        ref_el = "triangle"
    else:
        mesh = UnitCubeMesh(2, 2, 2)
        ref_el = "tetrahedron"

    Pk = FiniteElement("CG", ref_el, degree)
    V = FunctionSpace(mesh, Pk)
    V_interior = FunctionSpace(mesh, Pk["interior"])
    V_facet = FunctionSpace(mesh, Pk["facet"])

    def a(v, u):
        return inner(grad(v), grad(u)) * dx

    M = assemble(a(TestFunction(V), TrialFunction(V)))

    # Supplying constant vector to check computation
    nullsp = VectorSpaceBasis(constant=True,
                              vecs=[Function(V).assign(1.0)])
    nullsp._apply(M)

    A00 = Tensor(a(TestFunction(V_interior), TrialFunction(V_interior)))
    A01 = Tensor(a(TestFunction(V_interior), TrialFunction(V_facet)))
    A10 = Tensor(a(TestFunction(V_facet), TrialFunction(V_interior)))
    A11 = Tensor(a(TestFunction(V_facet), TrialFunction(V_facet)))
    S = A11 - A10 * A00.inv * A01

    Snullsp = create_sc_nullspace(M.petscmat, V, V_facet, COMM_WORLD)
    v = Snullsp.getVecs()[0].array_r

    _, _, vv = np.linalg.svd(assemble(S, mat_type="aij").M.values)
    singular_vector = vv[-1]

    assert np.allclose(np.linalg.norm(v), 1.0, 1e-13)
    assert np.allclose(v.min(), v.max(), 1e-13)
    assert np.allclose(np.absolute(v),
                       np.absolute(singular_vector), 1e-13)
