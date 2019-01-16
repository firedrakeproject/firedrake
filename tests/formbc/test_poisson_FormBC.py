# Simple Poisson equation
# =========================

import pytest

from firedrake import *

def test_nonlinear_FormBC():

    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(-8.0*pi*pi*cos(x*pi*2)*cos(y*pi*2))

    a = - dot(grad(v), grad(u)) * dx
    L = f * v * dx

    g1=Function(V)
    g1.interpolate(cos(2*pi*y))

    g3=Function(V)
    g3.interpolate(cos(2*pi*x))

    bc1 = FormBC( V, v*(u-g1)*ds(1)==0, 1 )
    bc2 = DirichletBC( V, cos(2*pi*y), 2 )
    bc3 = FormBC( V, v*(u-g3)*ds(3)==0, 3 )
    bc4 = DirichletBC( V, cos(2*pi*x), 4 )

    solve(a -L == 0, u, bcs = [ bc1, bc2, bc3, bc4 ], solver_parameters={'ksp_type': 'gmres','ksp_atol': 1e-12, 'ksp_rtol': 1e-20, 'ksp_divtol': 1e8})

    f.interpolate(cos(x*pi*2)*cos(y*pi*2))
    err=sqrt(assemble(dot(u - f, u - f) * dx))

    assert(err<0.05)

def test_linear_FormBC():

    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(-8.0*pi*pi*cos(x*pi*2)*cos(y*pi*2))

    a = - dot(grad(v), grad(u)) * dx
    L = f * v * dx

    g1=Function(V)
    g1.interpolate(cos(2*pi*y))

    g3=Function(V)
    g3.interpolate(cos(2*pi*x))

    bc1 = FormBC( V, v*u*ds(1)==v*g1*ds(1), 1 )
    bc2 = DirichletBC( V, cos(2*pi*y), 2 )
    bc3 = FormBC( V, v*u*ds(3)==v*g3*ds(3), 3 )
    bc4 = DirichletBC( V, cos(2*pi*x), 4 )

    u = Function(V)

    solve(a == L, u, bcs = [ bc1, bc2, bc3, bc4 ], solver_parameters={'ksp_type': 'gmres','ksp_atol': 1e-12, 'ksp_rtol': 1e-20, 'ksp_divtol': 1e8})

    f.interpolate(cos(x*pi*2)*cos(y*pi*2))
    err=sqrt(assemble(dot(u - f, u - f) * dx))

    assert(err<0.05)

if __name__ == "__main__":
    import os
    print(__file__)
    pytest.main(os.path.abspath(__file__))
