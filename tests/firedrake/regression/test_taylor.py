from firedrake import *
import numpy

# Test the 1D Taylor DG elements from FIAT


def test_Taylor():
    mesh = UnitIntervalMesh(10)
    VT = FunctionSpace(mesh, "TDG", 2)
    V1 = FunctionSpace(mesh, "DG", 2)
    V0 = FunctionSpace(mesh, "DG", 0)
    x = SpatialCoordinate(mesh)
    v = Function(V1).interpolate(sin(2*pi*x[0]))
    v0 = Function(V0).project(v)
    vt = Function(VT).project(v)

    assert numpy.abs(v0.dat.data - vt.dat.data[::3]).max() < 1.0e-10

    vt.dat.data[2::3] = 0.
    v0_1 = Function(V0).project(vt)

    assert numpy.abs(v0.dat.data - v0_1.dat.data).max() < 1.0e-10
