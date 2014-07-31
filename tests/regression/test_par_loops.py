import pytest
import numpy as np
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
def const(f):
    return Constant(1.0, domain=f[0].function_space().mesh().ufl_domain())


@pytest.fixture
def f_extruded():
    i = UnitIntervalMesh(2)
    m = ExtrudedMesh(i, 2, layer_height=0.1)
    cg = FunctionSpace(m, "CG", 1)
    dg = FunctionSpace(m, "DG", 0)

    c = Function(cg)
    d = Function(dg)

    return c, d


def test_direct_par_loop(f):
    c, d = f

    par_loop("""*c = 1;""", direct, {'c': (c, WRITE)})

    assert all(c.dat.data == 1)


def test_direct_par_loop_read_const(f, const):
    c, d = f
    const.assign(10.0)

    par_loop("""*c = *constant;""", direct, {'c': (c, WRITE), 'constant': (const, READ)})

    assert np.allclose(c.dat.data, const.dat.data)


def test_indirect_par_loop_read_const(f, const):
    c, d = f
    const.assign(10.0)

    par_loop("""for (int i = 0; i < d.dofs; i++) d[0][0] = *constant;""",
             dx, {'d': (d, WRITE), 'constant': (const, READ)})

    assert np.allclose(d.dat.data, const.dat.data)


def test_par_loop_const_write_error(f, const):
    _, d = f
    with pytest.raises(RuntimeError):
        par_loop("""c[0] = d[0];""", direct, {'c': (const, WRITE), 'd': (d, READ)})


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


def test_walk_facets_rt():
    m = UnitSquareMesh(3, 3)
    V = FunctionSpace(m, 'RT', 1)

    f1 = Function(V)
    f2 = Function(V)

    project(Expression(('x[0]', 'x[1]')), f1)

    par_loop("""
    for (int i = 0; i < f1.dofs; i++) {
        f2[i][0] = f1[i][0];
    }""", dS, {'f1': (f1, READ), 'f2': (f2, WRITE)})

    par_loop("""
    for (int i = 0; i < f1.dofs; i++) {
        f2[i][0] = f1[i][0];
    }""", ds, {'f1': (f1, READ), 'f2': (f2, WRITE)})

    assert errornorm(f1, f2, degree_rise=0) < 1e-10
