"""Test curl-grad = 0 and div-curl = 0, using enriched function spaces"""

import pytest

from firedrake import *


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((("CG", 1), ("RT", 1), ("DG", 0)), (("CG", 3), ("DG", 2))),
                          ((("CG", 2), ("RT", 2), ("DG", 1)), (("CG", 2), ("DG", 1))),
                          ((("CG", 3), ("RT", 3), ("DG", 2)), (("CG", 1), ("DG", 0))),
                          ((("CG", 2), ("BDM", 1), ("DG", 0)), (("CG", 1), ("DG", 0))),
                          ((("CG", 3), ("BDM", 2), ("DG", 1)), (("CG", 2), ("DG", 1))),
                          ((("CG", 2, "B", 3), ("BDFM", 2), ("DG", 1)), (("CG", 2), ("DG", 1)))])
def test_feec(extmesh, horiz_complex, vert_complex):
    U0, U1, U2 = horiz_complex
    V0, V1 = vert_complex
    # U0, U1, U2 is our horizontal complex
    # V0, V1 is be our vertical complex

    # W0, W1, W2, W3 will be our product complex, where
    # W0 = U0 x V0
    # W1 = HCurl(U1 x V0) + HCurl(U0 x V1)
    # W2 = HDiv(U2 x V0) + HDiv(U1 x V1)
    # W3 = U2 x V1
    mesh = extmesh(2, 2, 4)
    if len(U0) == 2:
        U0 = FiniteElement(U0[0], "triangle", U0[1])
    else:
        # make bubble space for BDFM
        U0_a = FiniteElement(U0[0], "triangle", U0[1])
        U0_b = FiniteElement(U0[2], "triangle", U0[3])
        U0 = NodalEnrichedElement(U0_a, U0_b)
    U1 = FiniteElement(U1[0], "triangle", U1[1])
    U2 = FiniteElement(U2[0], "triangle", U2[1])
    V0 = FiniteElement(V0[0], "interval", V0[1])
    V1 = FiniteElement(V1[0], "interval", V1[1])

    run_feec(mesh, U0, U1, U2, V0, V1)


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((("CG", 1), ("RTCF", 1), ("DQ", 0)), (("CG", 3), ("DG", 2))),
                          ((("CG", 2), ("RTCF", 2), ("DQ", 1)), (("CG", 2), ("DG", 1))),
                          ((("CG", 3), ("RTCF", 3), ("DQ", 2)), (("CG", 1), ("DG", 0)))])
def test_feec_quadrilateral(extmesh, horiz_complex, vert_complex):
    U0, U1, U2 = horiz_complex
    V0, V1 = vert_complex
    # U0, U1, U2 is our horizontal complex
    # V0, V1 is be our vertical complex

    # W0, W1, W2, W3 will be our product complex, where
    # W0 = U0 x V0
    # W1 = HCurl(U1 x V0) + HCurl(U0 x V1)
    # W2 = HDiv(U2 x V0) + HDiv(U1 x V1)
    # W3 = U2 x V1
    mesh = extmesh(2, 2, 4, quadrilateral=True)
    U0 = FiniteElement(U0[0], "quadrilateral", U0[1])
    U1 = FiniteElement(U1[0], "quadrilateral", U1[1])
    U2 = FiniteElement(U2[0], "quadrilateral", U2[1])
    V0 = FiniteElement(V0[0], "interval", V0[1])
    V1 = FiniteElement(V1[0], "interval", V1[1])

    run_feec(mesh, U0, U1, U2, V0, V1)


def run_feec(mesh, U0, U1, U2, V0, V1):
    W0_elt = TensorProductElement(U0, V0)

    W1_a = HCurl(TensorProductElement(U1, V0))
    W1_b = HCurl(TensorProductElement(U0, V1))
    W1_elt = W1_a + W1_b

    W2_a = HDiv(TensorProductElement(U2, V0))
    W2_b = HDiv(TensorProductElement(U1, V1))
    W2_elt = W2_a + W2_b

    W3_elt = TensorProductElement(U2, V1)

    W0 = FunctionSpace(mesh, W0_elt)
    W1 = FunctionSpace(mesh, W1_elt)
    W2 = FunctionSpace(mesh, W2_elt)
    W3 = FunctionSpace(mesh, W3_elt)

    parms = {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}

    # TEST CURL(GRAD(u)) = 0, for u in W0
    x, y, z = SpatialCoordinate(mesh)

    u = Function(W0)
    u.interpolate(x*y - y*z)

    v1 = TrialFunction(W1)
    v2 = TestFunction(W1)
    a = inner(v1, v2)*dx
    L = inner(grad(u), v2)*dx
    v = Function(W1)
    solve(a == L, v, solver_parameters=parms)

    w1 = TrialFunction(W2)
    w2 = TestFunction(W2)
    a = inner(w1, w2)*dx
    L = inner(curl(v), w2)*dx
    w = Function(W2)
    solve(a == L, w, solver_parameters=parms)
    maxcoeff = max(abs(w.dat.data))
    assert maxcoeff < 1e-11

    # TEST DIV(CURL(v)) = 0, for v in W1

    v = project(as_vector((x*y, -y*z, x*z)), W1)

    w1 = TrialFunction(W2)
    w2 = TestFunction(W2)
    a = inner(w1, w2)*dx
    L = inner(curl(v), w2)*dx
    w = Function(W2)
    solve(a == L, w, solver_parameters=parms)

    y1 = TrialFunction(W3)
    y2 = TestFunction(W3)
    a = inner(y1, y2)*dx
    L = inner(div(w), y2)*dx
    y = Function(W3)
    solve(a == L, y, solver_parameters=parms)
    maxcoeff = max(abs(y.dat.data))
    assert maxcoeff < 1e-11

    # TEST WEAKCURL(WEAKGRAD(y)) = 0, for y in W3

    y = Function(W3)
    y.interpolate(x*y - y*z)

    w1 = TrialFunction(W2)
    w2 = TestFunction(W2)
    a = inner(w1, w2)*dx
    L = inner(-y, div(w2))*dx
    w = Function(W2)
    solve(a == L, w, solver_parameters=parms)

    v1 = TrialFunction(W1)
    v2 = TestFunction(W1)
    a = inner(v1, v2)*dx
    L = -inner(w, curl(v2))*dx
    v = Function(W1)
    solve(a == L, v, solver_parameters=parms)
    maxcoeff = max(abs(v.dat.data))
    assert maxcoeff < 1e-11

    # TEST WEAKDIV(WEAKCURL(w)) = 0, for w in W2

    w = project(as_vector((x*y, -y*z, x*z)), W2)

    v1 = TrialFunction(W1)
    v2 = TestFunction(W1)
    a = inner(v1, v2)*dx
    L = -inner(w, curl(v2))*dx
    v = Function(W1)
    solve(a == L, v, solver_parameters=parms)

    u1 = TrialFunction(W0)
    u2 = TestFunction(W0)
    a = inner(u1, u2)*dx
    L = -inner(v, grad(u2))*dx
    u = Function(W0)
    solve(a == L, u, solver_parameters=parms)
    maxcoeff = max(abs(u.dat.data))
    assert maxcoeff < 3e-11
