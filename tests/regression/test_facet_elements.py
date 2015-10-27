import pytest
from firedrake import *
from tests.common import *


def test_all_dofs_on_facets(mesh):
    V_elt = FiniteElement("BDM", triangle, 1)
    # BDM has all dofs on facets, so these should be the same
    W1_elt = RestrictedElement(V_elt, "facet")
    V = FunctionSpace(mesh, V_elt)
    W1 = FunctionSpace(mesh, W1_elt)
    f = Function(V).assign(1)
    g = Function(W1).assign(1)
    n = FacetNormal(mesh)

    assert abs(assemble(dot(f, f)*dx) - assemble(dot(g, g)*dx)) < 1e-14
    assert abs(assemble(dot(f, f)*ds) - assemble(dot(g, g)*ds)) < 1e-14
    assert abs(assemble(dot(f, n)*ds) - assemble(dot(g, n)*ds)) < 1e-14
    assert abs(assemble(dot(f('+'), n('+'))*dS) - assemble(dot(g('+'), n('+'))*dS)) < 1e-14


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
