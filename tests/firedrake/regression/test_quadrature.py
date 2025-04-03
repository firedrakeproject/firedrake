from firedrake import *
import pytest


@pytest.fixture
def mesh(request):
    return UnitSquareMesh(5, 5)


def test_hand_specified_quadrature(mesh):
    V = FunctionSpace(mesh, 'CG', 2)
    v = TestFunction(V)

    a = conj(v) * dx

    a_q0 = assemble(a, form_compiler_parameters={'quadrature_degree': 0})
    a_q2 = assemble(a, form_compiler_parameters={'quadrature_degree': 2})

    assert not np.allclose(a_q0.dat.data, a_q2.dat.data)


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("mat_type", ["matfree", "aij"])
@pytest.mark.parametrize("family", ["Quadrature", "Boundary Quadrature"])
def test_quadrature_element(mesh, family, mat_type, diagonal):
    V = FunctionSpace(mesh, family, 2)
    v = TestFunction(V)
    u = TrialFunction(V)
    if family == "Boundary Quadrature":
        a = inner(u, v) * ds + inner(u('+'), v('+')) * dS
    else:
        a = inner(u, v) * dx

    assemble(a, mat_type=mat_type, diagonal=diagonal)
