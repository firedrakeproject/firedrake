from firedrake import *


def test_all_dofs_on_facets():
    mesh = UnitSquareMesh(5, 5)
    V_elt = FiniteElement("BDM", triangle, 1)
    # BDM has all dofs on facets, so these should be the same
    W1_elt = RestrictedElement(V_elt, "facet")
    V = FunctionSpace(mesh, V_elt)
    W1 = FunctionSpace(mesh, W1_elt)
    f = Function(V).assign(1)
    g = Function(W1).assign(1)
    n = FacetNormal(mesh)

    assert abs(assemble(inner(f, f)*dx) - assemble(inner(g, g)*dx)) < 2e-12
    assert abs(assemble(inner(f, f)*ds) - assemble(inner(g, g)*ds)) < 2e-12
    assert abs(assemble(inner(f, n)*ds) - assemble(inner(g, n)*ds)) < 2e-12
    assert abs(assemble(inner(f('+'), n('+'))*dS) - assemble(inner(g('+'), n('+'))*dS)) < 2e-12
