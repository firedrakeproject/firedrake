import pytest
from tests.common import longtest
from firedrake import *
import numpy as np


def run_hdiv_l2(refinement, hdiv_space, degree):
    mesh = UnitIcosahedralSphereMesh(refinement_level=refinement)

    mesh.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]')))
    Ve = FunctionSpace(mesh, "DG", max(3, degree + 1))

    V = FunctionSpace(mesh, hdiv_space, degree)
    Q = FunctionSpace(mesh, "DG", degree - 1)

    W = V*Q

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    g = Function(Q).interpolate(Expression("x[0]*x[1]*x[2]"))
    u_exact = Function(Ve, name="exact").interpolate(Expression("-x[0]*x[1]*x[2]/12.0"))

    a = (inner(sigma, tau) + div(sigma)*v + div(tau)*u)*dx
    L = g*v*dx

    w = Function(W)

    nullspace = MixedVectorSpaceBasis([W[0], VectorSpaceBasis(constant=True)])
    solve(a == L, w, nullspace=nullspace, solver_parameters={'pc_type': 'fieldsplit',
                                                             'pc_fieldsplit_type': 'schur',
                                                             'fieldsplit_0_pc_type': 'lu',
                                                             'pc_fieldsplit_schur_fact_type': 'FULL',
                                                             'fieldsplit_0_ksp_max_it': 100})

    sigma, u = w.split()

    L2_error_u = errornorm(u_exact, u, degree_rise=1)

    h = 1.0/(2**refinement * sin(2*pi/5))
    return L2_error_u, h, assemble(u*dx)


@longtest
@pytest.mark.parametrize(('hdiv_space', 'degree', 'conv_order'),
                         [('RT', 1, 0.75),
                          ('BDM', 1, 0.8)])
def test_hdiv_l2(hdiv_space, degree, conv_order):
    errors = [run_hdiv_l2(r, hdiv_space, degree) for r in range(1, 4)]
    errors = np.asarray(errors)
    l2err = errors[:, 0]
    l2conv = np.log2(l2err[:-1] / l2err[1:])
    assert (l2conv > conv_order).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
