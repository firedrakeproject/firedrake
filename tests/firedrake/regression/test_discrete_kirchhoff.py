import numpy as np
from firedrake import *


def discrete_kirchhoff_error(n):
    """Return the broken-Hessian error of the discrete Kirchhoff triangle on an n-by-n mesh."""
    mesh = UnitSquareMesh(n, n)
    W = FunctionSpace(mesh, "Reduced-Hermite", 3)
    Theta = FunctionSpace(mesh, "Rotated-Bernardi-Raugel", 1)

    w = TrialFunction(W)
    v = TestFunction(W)
    a = inner(grad(interpolate(grad(w), Theta)), grad(interpolate(grad(v), Theta))) * dx

    x, y = SpatialCoordinate(mesh)
    w_exact = x**2 * (1 - x)**2 * y**2 * (1 - y)**2
    L = div(grad(div(grad(w_exact)))) * v * dx

    wh = Function(W)
    solve(a == L, wh, bcs=DirichletBC(W, 0, "on_boundary"),
          solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    hessian_error = grad(interpolate(grad(wh), Theta)) - grad(grad(w_exact))
    return sqrt(assemble(inner(hessian_error, hessian_error) * dx(degree=12)))


def test_discrete_kirchhoff_convergence():
    errors = np.array([discrete_kirchhoff_error(n) for n in (4, 8, 16)])
    rates = np.log2(errors[:-1] / errors[1:])
    assert (rates > 0.8).all()
