from firedrake import *
import numpy as np


def test_pi():
    len = 7
    errors = np.zeros(len)
    for i in range(2, 2+len):
        m = CircleManifoldMesh(2**i)
        mesh = ExtrudedMesh(m, layers=2**i, layer_height=1.0/(2**i), extrusion_type="radial")
        fs = FunctionSpace(mesh, "DG", 0)
        f = Function(fs).assign(1)
        # area is pi*(2^2) - pi*(1^2) = 3*pi
        errors[i-2] = np.abs(assemble(f*dx) - 3*np.pi)

    # area converges linearly to 3*pi
    for i in range(len-1):
        assert ln(errors[i]/errors[i+1])/ln(2) > 0.95


def test_poisson():
    # u = x^2 + y^2 is a solution to the Poisson equation
    # -div(grad(u)) = -4 on the annulus with inner radius
    # 2, outer radius 5

    len = 4
    errors = np.zeros(len)

    for i in range(4, 4+len):
        m = CircleManifoldMesh(2**i, radius=2.0)
        mesh = ExtrudedMesh(m, layers=2**i, layer_height=3.0/(2**i), extrusion_type="radial")
        V = FunctionSpace(mesh, "CG", 1)

        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(-4)

        bcs = [DirichletBC(V, 4, "bottom"),
               DirichletBC(V, 25, "top")]

        out = Function(V)

        solve(inner(grad(u), grad(v))*dx == inner(f, v)*dx, out, bcs=bcs)

        exactfs = FunctionSpace(mesh, "CG", 2)
        xs = SpatialCoordinate(mesh)
        exact = Function(exactfs).interpolate(xs[0]*xs[0] + xs[1]*xs[1])

        errors[i-4] = sqrt(assemble((out-exact)*(out-exact)*dx))/sqrt(21*np.pi)  # normalised

    # we seem to get second-order convergence...
    for i in range(len-1):
        assert ln(errors[i]/errors[i+1])/ln(2) > 1.7
