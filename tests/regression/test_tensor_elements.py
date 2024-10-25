from firedrake import *
import pytest


rg = RandomGenerator(PCG64(seed=1))


@pytest.fixture(params=("square", "cube"))
def mesh(request):
    if request.param == "square":
        return UnitSquareMesh(2, 2)
    elif request.param == "cube":
        return UnitCubeMesh(2, 2, 2)
    else:
        raise ValueError("Unrecognized mesh type")


@pytest.mark.parametrize("degree", (0, 1, 2))
@pytest.mark.parametrize("family", ("Regge", "HHJ"))
def test_tensor_continuity(mesh, family, degree):
    V = FunctionSpace(mesh, family, degree)
    u = rg.beta(V, 1.0, 2.0)

    dim = mesh.topological_dimension()
    n = FacetNormal(mesh)

    space = V.ufl_element().sobolev_space
    if space == HDivDiv:
        utrace = dot(n, dot(u, n))
    elif space == HEin:
        if dim == 2:
            t = dot(as_matrix([[0, -1], [1, 0]]), n)
            utrace = dot(t, dot(u, t))
        else:
            t = as_matrix([[0, n[2], -n[1]], [-n[2], 0, n[0]], [n[1], -n[0], 0]])
            utrace = dot(t, dot(u, t.T))

    ujump = utrace('+') - utrace('-')
    assert assemble(inner(ujump, ujump)*dS) < 1E-10
