import pytest
from firedrake import *


@pytest.mark.parametrize('degree', [3, 4, 5])
def test_static_condensation(degree):
    mesh = UnitSquareMesh(8, 8, quadrilateral=False)
    V = FunctionSpace(mesh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    x = SpatialCoordinate(mesh)
    f = Function(V)
    f.interpolate((1+8*pi*pi)*cos(2*pi*x[0])*cos(2*pi*x[1]))

    a = (dot(grad(v), grad(u)) + v*u) * dx
    L = f * v * dx

    # Solve using static condensation
    u_hsc = Function(V)
    sc_params = {'mat_type': 'matfree',
                 'ksp_type': 'preonly',
                 'pc_type': 'python',
                 'pc_python_type': 'firedrake.StaticCondensationPC',
                 'static_condensation': {'ksp_type': 'preonly',
                                         'pc_type': 'lu'}}
    solve(a == L, u_hsc, solver_parameters=sc_params)

    # Solve without static condensation
    u_h = Function(V)
    solve(a == L, u_h, solver_parameters={'ksp_type': 'preonly',
                                          'pc_type': 'lu'})

    assert errornorm(u_h, u_hsc) < 1.0e-13


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
