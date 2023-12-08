import pytest
from firedrake import *
import numpy as np


def run_hdiv_l2(MeshClass, refinement, hdiv_space, degree):
    mesh = MeshClass(refinement_level=refinement)
    x = SpatialCoordinate(mesh)

    mesh.init_cell_orientations(x)
    Ve = FunctionSpace(mesh, "DG", max(3, degree + 1))

    V = FunctionSpace(mesh, hdiv_space, degree)
    Q = FunctionSpace(mesh, "DG", degree - 1)

    W = V*Q

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    g = Function(Q).interpolate(x[0]*x[1]*x[2])
    u_exact = Function(Ve, name="exact").interpolate(-x[0]*x[1]*x[2]/12.0)

    a = (inner(sigma, tau) + inner(div(sigma), v) + inner(u, div(tau)))*dx
    L = inner(g, v)*dx

    w = Function(W)

    nullspace = MixedVectorSpaceBasis(W, [W[0], VectorSpaceBasis(constant=True)])
    solve(a == L, w, nullspace=nullspace, solver_parameters={'pc_type': 'fieldsplit',
                                                             'pc_fieldsplit_type': 'schur',
                                                             'fieldsplit_0_pc_type': 'bjacobi',
                                                             'fieldsplit_0_sub_pc_type': 'ilu',
                                                             'fieldsplit_1_pc_type': 'none',
                                                             'pc_fieldsplit_schur_fact_type': 'FULL',
                                                             'fieldsplit_0_ksp_max_it': 100})

    sigma, u = w.subfunctions

    L2_error_u = errornorm(u_exact, u, degree_rise=1)

    h = 1.0/(2**refinement * sin(2*pi/5))
    return L2_error_u, h, assemble(u*dx)


@pytest.mark.parametrize(('MeshClass', 'hdiv_space', 'degree', 'refinement', 'conv_order'),
                         [(UnitIcosahedralSphereMesh, 'RT', 1, (1, 4), 0.75),
                          (UnitIcosahedralSphereMesh, 'BDM', 1, (1, 4), 0.8),
                          (UnitCubedSphereMesh, 'RTCF', 1, (2, 5), 0.8),
                          (UnitCubedSphereMesh, 'RTCF', 2, (2, 5), 1.7),
                          (UnitCubedSphereMesh, 'RTCF', 3, (2, 5), 1.8)])
def test_hdiv_l2(MeshClass, hdiv_space, degree, refinement, conv_order):
    errors = [run_hdiv_l2(MeshClass, r, hdiv_space, degree) for r in range(*refinement)]
    errors = np.asarray(errors)
    l2err = errors[:, 0]
    l2conv = np.log2(l2err[:-1] / l2err[1:])
    assert (l2conv > conv_order).all()


@pytest.mark.parallel
def test_hdiv_l2_cubedsphere_parallel():
    errors = [run_hdiv_l2(UnitCubedSphereMesh, r, 'RTCF', 2) for r in range(2, 5)]
    errors = np.asarray(errors)
    l2err = errors[:, 0]
    l2conv = np.log2(l2err[:-1] / l2err[1:])
    assert (l2conv > 1.7).all()
