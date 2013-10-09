"""This demo program projects an analytic expression into a function space
"""

# Begin demo
from firedrake import *
import sys


def project(test_mode, pwr=None):
    if pwr is None:
        power = int(sys.argv[1])
    else:
        power = pwr
    # Create mesh and define function space
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 2 ** power + 1

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.

    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / (2 ** power))
    V = FunctionSpace(mesh, "Lagrange", 1, vfamily="DG", vdegree=0)
    W = FunctionSpace(mesh, "Lagrange", 3, vfamily="Lagrange", vdegree=3)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    g = Function(W)
    f.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)*cos(x[2]*pi*2)"))
    g.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)*cos(x[2]*pi*2)"))
    a = u * v * dx
    L = f * v * dx
    x = Function(V)
    solve(a == L, x)

    if not test_mode:
        print sqrt(assemble((x - g) * (x - g) * dx))
        # Save solution in VTK format
        file = File("project.pvd")
        file << x
        file << f
    else:
        return sqrt(assemble((x - g) * (x - g) * dx))


def run_test(test_mode=False):
    import numpy as np
    l2_diff = [project(test_mode, pwr=i) for i in range(4, 8)]

    from math import log
    conv = [log(l2_diff[i] / l2_diff[i + 1], 2)
            for i in range(len(l2_diff) - 1)]
    return np.array(l2_diff), np.array(conv)

l2_diff, l2_conv = run_test(test_mode=True)
if __name__ == '__main__':
    print "L2 difference to analytic solution: %s" % l2_diff
    print "Convergence ratios: %s" % l2_conv
