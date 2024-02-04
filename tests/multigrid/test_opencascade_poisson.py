from firedrake import *

import pytest
import os


@pytest.fixture(scope='module', params=[("cylinder.step", 20), ("t_twist.step", 1.5), ("disk.step", 1)])
def stepdata(request):
    (stepfile, h) = request.param
    curpath = os.path.dirname(os.path.realpath(__file__))
    return (os.path.join(curpath, os.path.pardir, "meshes", stepfile), h)


@pytest.fixture(scope='module', params=[1, 2])
def order(request):
    return request.param


def test_opencascade_poisson(stepdata, order):
    (stepfile, h) = stepdata
    try:
        mh = OpenCascadeMeshHierarchy(stepfile, element_size=h, levels=2, order=order, cache=False, verbose=True)
    except ImportError:
        pytest.skip(reason="OpenCascade unavailable, skipping test")

    # Solve Poisson
    mesh = mh[-1]
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    f = Constant(1)
    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
    bcs = DirichletBC(V, Constant(0), 1)

    params = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "fgmres",
        "ksp_max_it": 20,
        "ksp_monitor_true_residual": None,
        "pc_type": "mg",
        "pc_mg_type": "full",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "sor",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
        "mg_coarse_mat_mumps_icntl_14": 200,
    }

    solve(F == 0, u, bcs, solver_parameters=params)
