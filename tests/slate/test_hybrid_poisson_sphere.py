from __future__ import absolute_import, print_function, division
import pytest
from firedrake import *
import numpy as np


def run_hybrid_poisson_sphere(MeshClass, refinement, hdiv_space):
    mesh = MeshClass(refinement_level=refinement)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
    x, y, z = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, hdiv_space, 1)
    U = FunctionSpace(mesh, "DG", 0)
    W = V * U

    f = Function(U)
    f.interpolate(x*y*z)

    u_exact = Function(U).interpolate(x*y*z/12.0)

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


def test_hybrid_conv():
    """Should expect approximately quadratic convergence for lowest order
    mixed method.
    """
    errors = [run_hybrid_poisson_sphere(UnitIcosahedralSphereMesh, r, 'BDM')
              for r in range(1, 4)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])[-1]
    assert l2conv > 1.8


@pytest.mark.parallel
def test_hybrid_conv_parallel():
    """Should expect approximately quadratic convergence for lowest order
    mixed method.
    """
    errors = [run_hybrid_poisson_sphere(UnitIcosahedralSphereMesh, r, 'RT')
              for r in range(1, 4)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])[-1]
    assert l2conv > 1.8


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
