from __future__ import absolute_import, print_function, division
import logging

import pytest

import ufl
from tsfc import compile_form
from tsfc.logging import logger


class MockHandler(logging.Handler):
    def emit(self, record):
        raise RuntimeError()


def test_estimated_degree():
    cell = ufl.tetrahedron
    mesh = ufl.Mesh(ufl.VectorElement('P', cell, 1))
    V = ufl.FunctionSpace(mesh, ufl.FiniteElement('P', cell, 1))
    f = ufl.Coefficient(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = u * v * ufl.tanh(ufl.sqrt(ufl.sinh(f) / ufl.sin(f**f))) * ufl.dx

    handler = MockHandler()
    logger.addHandler(handler)
    with pytest.raises(RuntimeError):
        compile_form(a)
    logger.removeHandler(handler)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
