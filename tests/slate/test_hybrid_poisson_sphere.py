from __future__ import absolute_import, print_function, division
import pytest
from firedrake import *
import numpy as np


def run_hybrid_poisson_sphere(MeshClass, refinement, hdiv_space):
    mesh = MeshClass(refinement_level=refinement)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))

    V = FunctionSpace(mesh, hdiv_space, 1)
    U = FunctionSpace(mesh, "DG", 0)
    W = V * U

    f = Function(U)
    expr = Expression("x[0]*x[1]*x[2]")
    f.interpolate(expr)

    u_exact = Function(U).interpolate(Expression("x[0]*x[1]*x[2]/12.0"))

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = (dot(sigma, tau) - div(tau)*u + v*div(sigma))*dx
    L = f*v*dx
    w = Function(W)

    nullsp = MixedVectorSpaceBasis(W, [W[0], VectorSpaceBasis(constant=True)])
    solve(a == L, w,
          nullspace=nullsp,
          solver_parameters={'mat_type': 'matfree',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'trace_pc_type': 'hypre',
                             'trace_ksp_type': 'preonly'})
    _, u_h = w.split()
    error = errornorm(u_exact, u_h)
    return error


@pytest.mark.longtest
@pytest.mark.parametrize(('MeshClass', 'hdiv_space'),
                         [(UnitIcosahedralSphereMesh, 'RT'),
                          (UnitIcosahedralSphereMesh, 'BDM')])
def test_hybrid_conv(MeshClass, hdiv_space):
    errors = [run_hybrid_poisson_sphere(MeshClass, r, hdiv_space)
              for r in range(2, 5)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])
    assert (l2conv > 1.8).all()


@pytest.mark.parallel(nprocs=4)
def test_hybrid_conv_parallel():
    errors = [run_hybrid_poisson_sphere(UnitIcosahedralSphereMesh, r, 'BDM')
              for r in range(2, 5)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])
    assert (l2conv > 1.8).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
