import pytest
from firedrake import *


def helmholtz_mixed(x, V1, V2):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2**x, 2**x)
    V1 = FunctionSpace(mesh, *V1, name="V")
    V2 = FunctionSpace(mesh, *V2, name="P")
    W = V1 * V2

    # Define variational problem
    lmbda = 1
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Function(V2)

    f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    a = (p*q - q*div(u) + lmbda*inner(v, u) + div(v)*p) * dx
    L = f*q*dx

    # Compute solution
    x = Function(W)

    # Block system is:
    # V Ct
    # Ch P
    # Eliminate V by forming a schur complement
    solve(a == L, x, solver_parameters={'pc_type': 'fieldsplit',
                                        'pc_fieldsplit_type': 'schur',
                                        'ksp_type': 'cg',
                                        'pc_fieldsplit_schur_fact_type': 'FULL',
                                        'fieldsplit_V_ksp_type': 'cg',
                                        'fieldsplit_P_ksp_type': 'cg'})

    # Analytical solution
    f.interpolate(Expression("sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    return sqrt(assemble(dot(x[2] - f, x[2] - f) * dx))


@pytest.mark.parametrize(('V1', 'V2', 'threshold'),
                         [(('RT', 1), ('DG', 0), 1.9),
                          (('BDM', 1), ('DG', 0), 1.9),
                          (('BDFM', 2), ('DG', 1), 1.9)])
def test_firedrake_helmholtz(V1, V2, threshold):
    import numpy as np
    diff = np.array([helmholtz_mixed(i, V1, V2) for i in range(3, 6)])
    print "l2 error norms:", diff
    conv = np.log2(diff[:-1] / diff[1:])
    print "convergence order:", conv
    assert (np.array(conv) > threshold).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
