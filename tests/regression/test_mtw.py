from firedrake import *
import numpy as np


convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])


def test_mtw():
    N_base = 2
    msh = UnitSquareMesh(N_base, N_base)
    mh = MeshHierarchy(msh, 5)

    V = FunctionSpace(msh, msh.coordinates.ufl_element())
    eps = Constant(1 / 2**(N_base-1))
    x, y = SpatialCoordinate(msh)
    new = Function(V).interpolate(as_vector([x + eps*sin(2*pi*x)*sin(2*pi*y),
                                             y - eps*sin(2*pi*x)*sin(2*pi*y)]))

    # And propagate to refined meshes
    coords = [new]
    for msh in mh[1:]:
        fine = Function(msh.coordinates.function_space())
        prolong(new, fine)
        coords.append(fine)
        new = fine

    for msh, coord in zip(mh, coords):
        msh.coordinates.assign(coord)

    params = {"snes_type": "newtonls",
              "snes_linesearch_type": "basic",
              "snes_monitor": None,
              "mat_type": "aij",
              "snes_max_it": 10,
              "snes_lag_jacobian": -2,
              "snes_lag_preconditioner": -2,
              "ksp_type": "preonly",
              "pc_type": "lu",
              "pc_factor_shift_type": "inblocks",
              "snes_rtol": 1e-16,
              "snes_atol": 1e-25}

    l2_u = []
    l2_p = []
    for msh in mh[1:]:
        x, y = SpatialCoordinate(msh)
        pex = sin(pi * x) * sin(2 * pi * y)
        uex = -grad(pex)
        f = div(uex)

        V = FunctionSpace(msh, "MTW", 3)
        W = FunctionSpace(msh, "DG", 0)
        Z = V * W

        up = Function(Z)
        u, p = split(up)
        v, w = TestFunctions(Z)

        F = (inner(u, v) * dx - inner(p, div(v)) * dx
             + inner(div(u), w) * dx - inner(f, w) * dx)

        solve(F == 0, up, solver_parameters=params)

        u, p = up.split()
        l2_u.append(errornorm(uex, u))
        l2_p.append(errornorm(pex, p))

    assert min(convergence_orders(l2_u)) > 1.8
    assert min(convergence_orders(l2_p)) > 0.8
