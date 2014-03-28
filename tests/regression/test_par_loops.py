import pytest
from firedrake import *


@pytest.fixture
def f():
    m = UnitIntervalMesh(2)
    cg = FunctionSpace(m, "CG", 1)
    dg = FunctionSpace(m, "DG", 0)

    c = Function(cg)
    d = Function(dg)

    return c, d


@pytest.fixture
def f_extruded():
    i = UnitIntervalMesh(2)
    m = ExtrudedMesh(i, 2, layer_height=0.1)
    cg = FunctionSpace(m, "CG", 1)
    dg = FunctionSpace(m, "DG", 0)

    c = Function(cg)
    d = Function(dg)

    return c, d


def test_cg_max_field(f):
    c, d = f
    d.interpolate(Expression("x[0]"))

    par_loop("""
    for (int i=0; i<c.dofs; i++)
       c[i][0] = fmax(c[i][0], d[0][0]);""",
             dx, {'c': (c, RW), 'd': (d, READ)})

    assert (c.dat.data == [1./4, 3./4, 3./4]).all()


def test_cg_max_field_extruded(f_extruded):
    c, d = f_extruded
    d.interpolate(Expression("x[0]"))

    par_loop("""
    for (int i=0; i<c.dofs; i++)
       c[i][0] = (c[i][0] > d[0][0] ? c[i][0] : d[0][0]);""",
             dx, {'c': (c, RW), 'd': (d, READ)})

    assert (c.dat.data == [1./4, 1./4, 1./4,
                           3./4, 3./4, 3./4,
                           3./4, 3./4, 3./4]).all()
