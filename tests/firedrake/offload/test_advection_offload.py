
from firedrake import *
import pytest
import numpy as np


@pytest.fixture(scope="module",
                params=[("cg", "sor"), ("cg", "gamg"), ("preonly", "lu")],
                ids=lambda x: '_'.join(x)
                )
def ksp_pc(request):
    return request.param


def run_test_advection_offload(ksp_type, pc_type):

    mesh = PeriodicIntervalMesh(100, length=2)
    x = SpatialCoordinate(mesh)[0]
    u_init = sin(2*pi*x)

    nested_parameters = {
        "pc_type": "ksp",
        "ksp": {
            "ksp_type": ksp_type,
            "ksp_max_it": 50,
            "ksp_rtol": 1e-5,
            "ksp_monitor": None,
            "pc_type": pc_type,
            'ksp_converged_reason': None,
        },
    }
    parameters = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.OffloadPC",
        "offload": nested_parameters,
    }

    nu = Constant(1e-2)

    V = FunctionSpace(mesh, "Lagrange", 2)

    u_n1 = Function(V, name="u^{n+1}")
    u_n = Function(V, name="u^{n}")
    v = TestFunction(V)

    u_n.interpolate(u_init)
    dt = 0.01

    F = (((u_n1 - u_n)/dt) * v + u_n1 * u_n1.dx(0) * v + nu*u_n1.dx(0)*v.dx(0))*dx

    problem = NonlinearVariationalProblem(F, u_n1)
    solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)

    t = 0
    steps = 1200
    t_end = steps * dt

    while t <= t_end:
        solver.solve()
        maxchange = sqrt(assemble((u_n - u_n1)**2 * dx))
        u_n.assign(u_n1)
        t += dt
        if maxchange < 1e-5 or np.isnan(maxchange):
            break
    assert maxchange < 1e-5
