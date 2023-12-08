from firedrake import *


def test_hand_specified_quadrature():
    mesh = UnitSquareMesh(5, 5)
    V = FunctionSpace(mesh, 'CG', 2)
    v = TestFunction(V)

    a = conj(v) * dx

    a_q0 = assemble(a, form_compiler_parameters={'quadrature_degree': 0})
    a_q2 = assemble(a, form_compiler_parameters={'quadrature_degree': 2})

    assert not np.allclose(a_q0.dat.data, a_q2.dat.data)
