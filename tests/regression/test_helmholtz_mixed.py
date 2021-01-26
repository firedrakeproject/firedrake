import pytest
from firedrake import *


def helmholtz_mixed(r, V1, V2, action=False):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2**r, 2**r)
    V1 = FunctionSpace(mesh, *V1, name="V")
    V2 = FunctionSpace(mesh, *V2, name="P")
    W = V1 * V2

    # Define variational problem
    lmbda = 1
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Function(V2)

    x = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2))
    a = (inner(p, q) - inner(div(u), q) + lmbda*inner(u, v) + inner(p, div(v))) * dx
    L = inner(f, q) * dx

    # Compute solution
    sol = Function(W)

    if action:
        system = action(a, sol) - L == 0
    else:
        system = a == L

    # Block system is:
    # V Ct
    # Ch P
    # Eliminate V by forming a schur complement
    solve(system, sol, solver_parameters={'pc_type': 'fieldsplit',
                                          'pc_fieldsplit_type': 'schur',
                                          'ksp_type': 'cg',
                                          'pc_fieldsplit_schur_fact_type': 'FULL',
                                          'fieldsplit_V_ksp_type': 'cg',
                                          'fieldsplit_P_ksp_type': 'cg'})

    # Analytical solution
    f.interpolate(sin(x[0]*pi*2)*sin(x[1]*pi*2))
    return sqrt(assemble(inner(sol[2] - f, sol[2] - f) * dx))


@pytest.mark.parametrize(('V1', 'V2', 'threshold', 'action'),
                         [(('RT', 1), ('DG', 0), 1.9, False),
                          (('BDM', 1), ('DG', 0), 1.89, False),
                          (('BDM', 1), ('DG', 0), 1.89, True),
                          (('BDFM', 2), ('DG', 1), 1.9, False)])
def test_firedrake_helmholtz(V1, V2, threshold, action):
    import numpy as np
    diff = np.array([helmholtz_mixed(i, V1, V2) for i in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > threshold).all()
