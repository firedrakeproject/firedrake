from firedrake import *
from firedrake.__future__ import *
import numpy
from firedrake.utils import IntType


def test_poisson_variable_layers():
    # +                   +
    # |\                 /|
    # +-+               +-+
    # | |\             /| |
    # +-+-+-+-+-+-+-+-+-+-+
    # | | | | | | | | | | |
    # +-+-+-+-+-+-+-+-+-+-+
    # | | | | | | | | | | |
    # +-+-+-+-+-+-+-+-+-+-+
    #
    # Homogeneous Neumann on left and right
    # Dirichlet on top and bottom with value 1 + y.
    # Exact solution 1 + y.
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "DG", 0)

    x, = SpatialCoordinate(mesh)

    selector = assemble(interpolate(
        conditional(
            Or(real(x) < 0.1,
               real(x) > 0.9),
            4,
            conditional(Or(And(real(x) > 0.1, real(x) < 0.2),
                           And(real(x) > 0.8, real(x) < 0.9)),
                        3, 2)),
        V))

    layers = numpy.empty((10, 2), dtype=IntType)

    layers[:, 0] = 0
    layers[:, 1] = selector.dat.data_ro.real

    extmesh = ExtrudedMesh(mesh, layers=layers,
                           layer_height=0.25)

    extmesh.coordinates.dat.data[9, 1] = 0.75
    extmesh.coordinates.dat.data[13, 1] = 0.5
    extmesh.coordinates.dat.data[-6, 1] = 0.75
    extmesh.coordinates.dat.data[-11, 1] = 0.5

    V = FunctionSpace(extmesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(0), v)*dx

    x, y = SpatialCoordinate(extmesh)

    exact = 1 + y

    bcs = [DirichletBC(V, exact, "bottom"),
           DirichletBC(V, exact, "top")]

    uh = Function(V)
    solve(a == L, uh, bcs=bcs)

    assert numpy.allclose(uh.dat.data_ro, assemble(interpolate(exact, V)).dat.data_ro)
