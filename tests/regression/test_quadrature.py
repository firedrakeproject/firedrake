from firedrake import *


def test_hand_specified_quadrature():
    mesh = UnitSquareMesh(5, 5)
    V = FunctionSpace(mesh, 'CG', 2)
    v = TestFunction(V)

    a = v*dx

    norm_q0 = norm(assemble(a, form_compiler_parameters={'quadrature_degree': 0}))
    norm_q2 = norm(assemble(a, form_compiler_parameters={'quadrature_degree': 2}))

    assert norm_q0 != norm_q2
