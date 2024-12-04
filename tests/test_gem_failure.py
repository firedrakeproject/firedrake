from ufl import (triangle, tetrahedron, FunctionSpace, Mesh,
                 TrialFunction, TestFunction, inner, grad, dx, dS)
from finat.ufl import FiniteElement, VectorElement
from tsfc import compile_form
from FIAT.hdiv_trace import TraceError
import pytest


@pytest.mark.parametrize('cell', [triangle, tetrahedron])
@pytest.mark.parametrize('degree', range(3))
def test_cell_error(cell, degree):
    """Test that tabulating the trace element deliberatly on the
    cell triggers `gem.Failure` to raise the TraceError exception.
    """
    trace_element = FiniteElement("HDiv Trace", cell, degree)
    domain = Mesh(VectorElement("Lagrange", cell, 1))
    space = FunctionSpace(domain, trace_element)
    lambdar = TrialFunction(space)
    gammar = TestFunction(space)

    with pytest.raises(TraceError):
        compile_form(lambdar * gammar * dx)


@pytest.mark.parametrize('cell', [triangle, tetrahedron])
@pytest.mark.parametrize('degree', range(3))
def test_gradient_error(cell, degree):
    """Test that tabulating gradient evaluations of the trace
    element triggers `gem.Failure` to raise the TraceError
    exception.
    """
    trace_element = FiniteElement("HDiv Trace", cell, degree)
    domain = Mesh(VectorElement("Lagrange", cell, 1))
    space = FunctionSpace(domain, trace_element)
    lambdar = TrialFunction(space)
    gammar = TestFunction(space)

    with pytest.raises(TraceError):
        compile_form(inner(grad(lambdar('+')), grad(gammar('+'))) * dS)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
