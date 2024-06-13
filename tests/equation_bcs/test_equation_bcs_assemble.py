import pytest

from firedrake import *
import numpy as np


def test_equation_bcs_direct_assemble_one_form():
    mesh = UnitSquareMesh(1, 1, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V).assign(1.)
    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx
    F1 = inner(u, v) * ds(1)
    bc = EquationBC(F1 == 0, u, 1)

    g = assemble(F, bcs=bc.extract_form('F'))
    assert np.allclose(g.dat.data, [0.5, 0.5, 0, 0])


def test_equation_bcs_direct_assemble_two_form():
    mesh = UnitSquareMesh(1, 1, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    a1 = inner(u, v) * ds(1)
    L1 = inner(Constant(0), v) * ds(1)
    sol = Function(V)
    bc = EquationBC(a1 == L1, sol, 1, Jp=2 * inner(u, v) * ds(1))

    # Must preprocess bc to extract appropriate
    # `EquationBCSplit` object.
    A = assemble(a, bcs=bc.extract_form('J'))
    assert np.allclose(A.M.values, [[1 / 3, 1 / 6, 0, 0],
                                    [1 / 6, 1 / 3, 0, 0],
                                    [-1 / 3, -1 / 6, 2 / 3, -1 / 6],
                                    [-1 / 6, -1 / 3, -1 / 6, 2 / 3]])
    A = assemble(a, bcs=bc.extract_form('Jp'))
    assert np.allclose(A.M.values, [[2 / 3, 2 / 6, 0, 0],
                                    [2 / 6, 2 / 3, 0, 0],
                                    [-1 / 3, -1 / 6, 2 / 3, -1 / 6],
                                    [-1 / 6, -1 / 3, -1 / 6, 2 / 3]])
    with pytest.raises(TypeError) as excinfo:
        # Unable to use raw `EquationBC` object, as
        # assembler can not infer merely from the rank
        # which form ('J' or 'Jp') should be assembled.
        assemble(a, bcs=bc)
    assert "EquationBC objects not expected here" in str(excinfo.value)
