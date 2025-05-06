import logging

import ufl
import finat.ufl
from tsfc import compile_form


def test_estimated_degree(caplog):
    cell = ufl.tetrahedron
    mesh = ufl.Mesh(finat.ufl.VectorElement('P', cell, 1))
    V = ufl.FunctionSpace(mesh, finat.ufl.FiniteElement('P', cell, 1))
    f = ufl.Coefficient(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = u * v * ufl.tanh(ufl.sqrt(ufl.sinh(f) / ufl.sin(f**f))) * ufl.dx

    with caplog.at_level(logging.WARNING, logger="tsfc"):
        compile_form(a)
    assert "more than tenfold greater" in caplog.text
