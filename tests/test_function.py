from firedrake import *


def test_firedrake_function():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    f.interpolate(Expression("1"))
    assert (f.dat.data_ro == 1.0).all()

    g = Function(f)
    assert (g.dat.data_ro == 1.0).all()

    # Check that g is indeed a deep copy
    f.interpolate(Expression("2"))

    assert (f.dat.data_ro == 2.0).all()
    assert (g.dat.data_ro == 1.0).all()
