from firedrake import *
from firedrake.__future__ import *
import numpy
from firedrake.utils import IntType


def test_steady_advection_variable_layers():
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
    # Constant advecting velocity, (1, 0)
    # Constant inflow on left wall.
    #   1 if 0.25 < y < 0.75
    #   0.5 otherwise
    # Outflow on all other boundaries (so right wall and downslope of "top").
    #
    # Expected solution:
    # In the bottom half of the domain, we just advect the inflow.
    # In top top half, we advect and then it flows out on the
    # downslope of columns 1 and 2.  Hence the right triangle has zero
    # advected quantity.
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

    # BDM1 element on a quad
    W0_h = FiniteElement("CG", "interval", 1)
    W0_v = FiniteElement("DG", "interval", 1)
    W0 = HDiv(TensorProductElement(W0_h, W0_v))

    W1_h = FiniteElement("DG", "interval", 1)
    W1_v = FiniteElement("CG", "interval", 1)
    W1 = HDiv(TensorProductElement(W1_h, W1_v))

    W = FunctionSpace(extmesh, W0+W1)

    DG0 = FunctionSpace(extmesh, "DG", 0)

    velocity = as_vector([1, 0])

    u0 = project(velocity, W)

    x, y = SpatialCoordinate(extmesh)
    inflow = conditional(And(real(y) > 0.25, real(y) < 0.75),
                         1.0,
                         0.5)

    n = FacetNormal(extmesh)

    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DG0)
    phi = TestFunction(DG0)

    a1 = -inner(D, dot(u0, grad(phi)))*dx
    a2 = inner(un('+')*D('+') - un('-')*D('-'), jump(phi))*dS_v
    a3 = inner(D*un, phi)*ds_v(2)  # outflow at right-hand wall
    a4 = inner(un*D, phi)*ds_t     # outflow on top boundary
    a = a1 + a2 + a3 + a4

    L = -inner(inflow*dot(u0, n), phi)*ds_v(1)  # inflow at left-hand wall

    out = Function(DG0)
    solve(a == L, out)

    expected = assemble(interpolate(conditional(real(x) > 0.5,
                                                conditional(real(y) < 0.25,
                                                            0.5,
                                                            conditional(real(y) < 0.5,
                                                                        1.0,
                                                                        0.0)),
                                                conditional(And(real(y) > 0.25, real(y) < 0.75),
                                                            1.0,
                                                            0.5)),
                                    DG0))

    assert numpy.allclose(out.dat.data_ro, expected.dat.data_ro)
