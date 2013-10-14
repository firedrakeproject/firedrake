"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

# Begin demo
import pytest
import sys
from firedrake import *


def helmholtz(test_mode, pwr=None):
    if pwr is None:
        power = int(sys.argv[1])
    else:
        power = pwr
    # Create mesh and define function space
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 2 ** power + 1

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.

    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / (layers - 1))

    V = FunctionSpace(mesh, "Lagrange", 1, vfamily="Lagrange", vdegree=1)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    f.interpolate(
        Expression("(1+12*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2)*cos(x[2]*pi*2)"))
    a = (dot(grad(v), grad(u)) + v * u) * dx
    L = f * v * dx

    # Compute solution
    assemble(a)
    assemble(L)
    x = Function(V)
    solve(a == L, x)

    # Analytical solution
    f.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)*cos(x[2]*pi*2)"))

    if not test_mode:
        print sqrt(assemble((x - f) * (x - f) * dx))
        # Save solution in VTK format
        file = File("helmholtz.pvd")
        file << x
        file << f
    else:
        return sqrt(assemble((x - f) * (x - f) * dx))


@pytest.mark.skipif("config.option.short")
def test_firedrake_extrusion_helmholtz():
    import numpy as np
    l2_diff = np.array([helmholtz(test_mode=True, pwr=i) for i in range(4, 8)])
    print "L2 error norms:", l2_diff
    l2_conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    print "Convergence order:", l2_conv
    assert (l2_conv > 1.8).all()
