from __future__ import absolute_import, print_function, division
from firedrake import *
from firedrake.formmanipulation import split_form
from ufl.classes import ListTensor
from ufl.corealg.traversal import unique_pre_traversal
from ufl.algorithms.analysis import extract_arguments
from ufl.algorithms.traversal import iter_expressions
import pytest


@pytest.fixture(scope='module')
def rayleigh_benard():
    mesh = UnitSquareMesh(2, 2)

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = V * W * Q

    upT = Function(Z)
    u, p, T = split(upT)
    v, q, S = TestFunctions(Z)

    Ra = Constant(200.0)  # Rayleigh number
    Pr = Constant(6.8)    # Prandtl number

    g = Constant((0, -1))  # Gravity

    # Residual
    F = (
        inner(grad(u), grad(v))*dx
        + inner(dot(grad(u), u), v)*dx
        - inner(p, div(v))*dx
        - Ra*Pr*inner(T*g, v)*dx
        + inner(div(u), q)*dx
        + inner(dot(grad(T), u), S)*dx
        + 1/Pr * inner(grad(T), grad(S))*dx
    )

    # Jacobian
    J = derivative(F, upT)

    return F, J


def test_foo(rayleigh_benard):
    F, J = rayleigh_benard
    for form in [J]:
        for idx, f in split_form(form):
            assert not any(extract_arguments(n)
                           for e in iter_expressions(f)
                           for n in unique_pre_traversal(e)
                           if isinstance(n, ListTensor))


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
