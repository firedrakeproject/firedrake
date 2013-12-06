"""Test curl-grad = 0 and div-curl = 0, using enriched function spaces"""

import pytest

from firedrake import *
from common import *


@pytest.mark.parametrize(('horiz_complex', 'vert_complex'),
                         [((("CG", 1), ("RT", 1), ("DG", 0)), (("CG", 3), ("DG", 2))),
                          ((("CG", 2), ("RT", 2), ("DG", 1)), (("CG", 2), ("DG", 1))),
                          ((("CG", 3), ("RT", 3), ("DG", 2)), (("CG", 1), ("DG", 0))),
                          ((("CG", 2), ("BDM", 1), ("DG", 0)), (("CG", 1), ("DG", 0))),
                          ((("CG", 3), ("BDM", 2), ("DG", 1)), (("CG", 2), ("DG", 1)))])
def test_feec(horiz_complex, vert_complex):
    hcV0, hcV1, hcV2 = horiz_complex
    vcV0, vcV1 = vert_complex
    mesh = extmesh(2, 2, 4)
    V0 = FunctionSpace(mesh, hcV0[0], hcV0[1], vfamily=vcV0[0], vdegree=vcV0[1])

    V1_a_horiz = FiniteElement(hcV1[0], "triangle", hcV1[1])
    V1_a_vert = FiniteElement(vcV0[0], "interval", vcV0[1])
    V1_a = HCurl(OuterProductElement(V1_a_horiz, V1_a_vert))

    V1_b_horiz = FiniteElement(hcV0[0], "triangle", hcV0[1])
    V1_b_vert = FiniteElement(vcV1[0], "interval", vcV1[1])
    V1_b = HCurl(OuterProductElement(V1_b_horiz, V1_b_vert))

    V1_elt = EnrichedElement(V1_a, V1_b)
    V1 = FunctionSpace(mesh, V1_elt)

    V2_a_horiz = FiniteElement(hcV1[0], "triangle", hcV1[1])
    V2_a_vert = FiniteElement(vcV1[0], "interval", vcV1[1])
    V2_a = HDiv(OuterProductElement(V2_a_horiz, V2_a_vert))

    V2_b_horiz = FiniteElement(hcV2[0], "triangle", hcV2[1])
    V2_b_vert = FiniteElement(vcV0[0], "interval", vcV0[1])
    V2_b = HDiv(OuterProductElement(V2_b_horiz, V2_b_vert))

    V2_elt = EnrichedElement(V2_a, V2_b)
    V2 = FunctionSpace(mesh, V2_elt)

    V3 = FunctionSpace(mesh, hcV2[0], hcV2[1], vfamily=vcV1[0], vdegree=vcV1[1])

    parms = {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}

    ### TEST CURL(GRAD(u)) = 0, for u in V0 ###

    u = Function(V0)
    u.interpolate(Expression("x[0]*x[1] - x[1]*x[2]"))

    v1 = TrialFunction(V1)
    v2 = TestFunction(V1)
    a = dot(v1, v2)*dx
    L = dot(grad(u), v2)*dx
    v = Function(V1)
    solve(a == L, v, solver_parameters=parms)

    w1 = TrialFunction(V2)
    w2 = TestFunction(V2)
    a = dot(w1, w2)*dx
    L = dot(curl(v), w2)*dx
    w = Function(V2)
    solve(a == L, w, solver_parameters=parms)
    maxcoeff = max(abs(w.dat.data))
    assert maxcoeff < 1e-12

    ### TEST DIV(CURL(v)) = 0, for v in V1 ###

    v = project(Expression(("x[0]*x[1]", "-x[1]*x[2]", "x[0]*x[2]")), V1)

    w1 = TrialFunction(V2)
    w2 = TestFunction(V2)
    a = dot(w1, w2)*dx
    L = dot(curl(v), w2)*dx
    w = Function(V2)
    solve(a == L, w, solver_parameters=parms)

    y1 = TrialFunction(V3)
    y2 = TestFunction(V3)
    a = y1*y2*dx
    L = div(w)*y2*dx
    y = Function(V3)
    solve(a == L, y, solver_parameters=parms)
    maxcoeff = max(abs(y.dat.data))
    assert maxcoeff < 2e-12

    ### TEST WEAKCURL(WEAKGRAD(y)) = 0, for y in V3 ###

    y = Function(V3)
    y.interpolate(Expression("x[0]*x[1] - x[1]*x[2]"))

    w1 = TrialFunction(V2)
    w2 = TestFunction(V2)
    a = dot(w1, w2)*dx
    L = -y*div(w2)*dx
    w = Function(V2)
    solve(a == L, w, solver_parameters=parms)

    v1 = TrialFunction(V1)
    v2 = TestFunction(V1)
    a = dot(v1, v2)*dx
    L = -dot(w, curl(v2))*dx
    v = Function(V1)
    solve(a == L, v, solver_parameters=parms)
    maxcoeff = max(abs(v.dat.data))
    assert maxcoeff < 2e-12

    ### TEST WEAKDIV(WEAKCURL(w)) = 0, for w in V2 ###

    w = project(Expression(("x[0]*x[1]", "-x[1]*x[2]", "x[0]*x[2]")), V2)

    v1 = TrialFunction(V1)
    v2 = TestFunction(V1)
    a = dot(v1, v2)*dx
    L = -dot(w, curl(v2))*dx
    v = Function(V1)
    solve(a == L, v, solver_parameters=parms)

    u1 = TrialFunction(V0)
    u2 = TestFunction(V0)
    a = dot(u1, u2)*dx
    L = -dot(v, grad(u2))*dx
    u = Function(V0)
    solve(a == L, u, solver_parameters=parms)
    maxcoeff = max(abs(u.dat.data))
    assert maxcoeff < 5e-12

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
