from firedrake import *


def helmholtz_mixed(x):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2**x, 2**x)
    RT1 = FunctionSpace(mesh, "RT", 1)
    P0 = FunctionSpace(mesh, "DG", 0)
    W = RT1 * P0

    # Define variational problem
    lmbda = 1
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Function(P0)

    f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    a = (p*q - q*div(u) + lmbda*inner(v, u) + div(v)*p) * dx
    L = f*q*dx

    # Compute solution
    x = Function(W)
    solve(a == L, x, solver_parameters={'pc_type': 'none'})

    # Analytical solution
    f.interpolate(Expression("sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    return sqrt(assemble(dot(x[2] - f, x[2] - f) * dx))


def test_firedrake_helmholtz():
    import numpy as np
    diff = np.array([helmholtz_mixed(i) for i in range(3, 7)])
    print "l2 error norms:", diff
    conv = np.log2(diff[:-1] / diff[1:])
    print "convergence order:", conv
    assert (np.array(conv) > 1.9).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
