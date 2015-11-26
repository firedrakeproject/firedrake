from firedrake import *
import numpy as np
import pytest


@pytest.fixture
def f():
    m = UnitSquareMesh(2, 2)
    me = ExtrudedMesh(m, 2)
    fs = FunctionSpace(me, "CG", 1)

    return Function(fs)


def test_set_top_x(f):

    f.interpolate(Expression("x[0]"))
    f_top = f.extract_subfunction("top")

    integral = assemble(f_top * f_top.function_space().mesh()._dx)

    assert np.round(integral - 0.5, 12) == 0


def test_set_top_z(f):

    f.interpolate(Expression("x[2]"))
    f_top = f.extract_subfunction("top")

    integral = assemble(f_top * f_top.function_space().mesh()._dx)

    assert np.round(integral - 1., 12) == 0


def test_set_bottom_x(f):

    f.interpolate(Expression("x[0]"))
    f_bottom = f.extract_subfunction("bottom")

    integral = assemble(f_bottom * f_bottom.function_space().mesh()._dx)

    assert np.round(integral - 0.5, 12) == 0


def test_set_bottom_z(f):

    f.interpolate(Expression("1. - x[2]"))
    f_bottom = f.extract_subfunction("bottom")

    integral = assemble(f_bottom * f_bottom.function_space().mesh()._dx)

    assert np.round(integral - 1., 12) == 0
