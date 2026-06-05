import numpy as np
from firedrake import *
from firedrake.utils import single_mode

# fp32: relaxed to the ~1e-5 residual floor (1e-7 is below single-precision eps).


def test_semicoarsened_poisson():
    N = 10
    base = UnitIntervalMesh(N)
    hierarchy = SemiCoarsenedExtrudedHierarchy(base, 1.0, base_layer=1, nref=2)
    mesh = hierarchy[-1]
    x, y = SpatialCoordinate(mesh)
    H1 = FunctionSpace(mesh, 'CG', 1)
    f = Function(H1).interpolate(8.0 * pi * pi * cos(2 * pi * x) * cos(2 * pi * y))
    g = Function(H1).interpolate(cos(2 * pi * x) * cos(2 * pi * y))  # boundary condition and exact solution
    u = Function(H1)
    v = TestFunction(H1)
    F = (inner(grad(u), grad(v)) - inner(f, v)) * dx
    params = {'snes_type': 'ksponly',
              'ksp_rtol': 1e-5 if single_mode else 1e-14,
              'ksp_type': 'cg',
              'pc_type': 'mg'}
    solve(F == 0, u, bcs=[DirichletBC(H1, g, 1)], solver_parameters=params)
    uh = Function(u)

    u.assign(0)
    solve(F == 0, u, bcs=[DirichletBC(H1, g, 1)])

    assert np.allclose(uh.dat.data_ro, u.dat.data_ro, atol=1e-4 if single_mode else 1e-8)
    assert errornorm(uh, g) < 0.04
