from firedrake import *


def test_vanish_on_bdy():
    mesh = UnitSquareMesh(5, 5)
    V_elt = FiniteElement("RT", triangle, 2)
    W2_elt = RestrictedElement(V_elt, "interior")
    W2 = FunctionSpace(mesh, W2_elt)
    g = Function(W2).assign(1)
    n = FacetNormal(mesh)

    # For H(div) elements, interior dofs have u.n = 0 on facets
    assert abs(assemble(dot(g('+'), n('+'))*dS)) < 1e-14
    assert abs(assemble(dot(g, n)*ds)) < 1e-14
