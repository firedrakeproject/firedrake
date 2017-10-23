import pytest
from firedrake import *


@pytest.mark.parametrize('degree', [3, 4, 5])
def test_primal_poisson_sphere(degree):
    mesh = UnitIcosahedralSphereMesh(refinement_level=3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    V = FunctionSpace(mesh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V).interpolate(x[0]*x[1]*x[2])

    a = dot(grad(v), grad(u)) * dx
    L = f * v * dx
    nullspace = VectorSpaceBasis(constant=True)

    # Solve using static condensation
    u_hsc = Function(V)
    sc_params = {'mat_type': 'matfree',
                 'ksp_type': 'preonly',
                 'pc_type': 'python',
                 'pc_python_type': 'firedrake.StaticCondensationPC',
                 'static_condensation': {'ksp_type': 'preonly',
                                         'pc_type': 'lu',
                                         'pc_factor_mat_solver_package': 'mumps'}}
    solve(a == L, u_hsc, nullspace=nullspace, solver_parameters=sc_params)

    # Solve without static condensation
    u_h = Function(V)
    params = {'ksp_type': 'preonly',
              'pc_type': 'lu',
              'pc_factor_mat_solver_package': 'mumps'}
    solve(a == L, u_h, nullspace=nullspace, solver_parameters=params)

    assert errornorm(u_h, u_hsc) < 1.0e-10


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
