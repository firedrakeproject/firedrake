import pytest

from ufl import Coefficient, TestFunction, dx, inner, tetrahedron, Mesh, FunctionSpace
from finat.ufl import FiniteElement, MixedElement, VectorElement

from tsfc import compile_form


@pytest.mark.parametrize('mode', ['vanilla', 'coffee', 'spectral'])
def test_delta_elimination(mode):
    # Code sample courtesy of Marco Morandini:
    # https://github.com/firedrakeproject/tsfc/issues/182
    scheme = "default"
    degree = 3

    element_lambda = FiniteElement("Quadrature", tetrahedron, degree,
                                   quad_scheme=scheme)
    element_eps_p = VectorElement("Quadrature", tetrahedron, degree,
                                  dim=6, quad_scheme=scheme)

    element_chi_lambda = MixedElement(element_eps_p, element_lambda)
    domain = Mesh(VectorElement("Lagrange", tetrahedron, 1))
    space = FunctionSpace(domain, element_chi_lambda)

    chi_lambda = Coefficient(space)
    delta_chi_lambda = TestFunction(space)

    L = inner(delta_chi_lambda, chi_lambda) * dx(degree=degree, scheme=scheme)
    kernel, = compile_form(L, parameters={'mode': mode})


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
