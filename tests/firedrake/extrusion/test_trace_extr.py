"""Tests projections onto the HDiv Trace space on extruded meshes"""
import pytest

from firedrake import *


@pytest.mark.parametrize('quad', [False, True])
@pytest.mark.parametrize('degree', range(2))
def test_trace_galerkin_projection_extr(degree, quad):
    mesh = ExtrudedMesh(UnitSquareMesh(4, 4, quadrilateral=quad), 2)
    f = Function(FunctionSpace(mesh, "CG", degree + 1))
    x, y, z = SpatialCoordinate(mesh)
    f.interpolate(cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z))

    # degree=('horizontal degree', 'vertical degree')
    T = FunctionSpace(mesh, "HDiv Trace", degree=(degree + 1, degree + 1))
    u = TrialFunction(T)
    v = TestFunction(T)

    a_ds = inner(u, v)*ds_t + inner(u, v)*ds_b + inner(u, v)*ds_v
    a_dS = inner(u('+'), v('+'))*dS_h + inner(u('+'), v('+'))*dS_v
    A = a_ds + a_dS
    l_ds = inner(f, v)*ds_t + inner(f, v)*ds_b + inner(f, v)*ds_v
    l_dS = inner(f('+'), v('+'))*dS_h + inner(f('+'), v('+'))*dS_v
    L = l_ds + l_dS

    sol = Function(T)
    solve(A == L, sol, solver_parameters={'ksp_rtol': 1e-14})

    m = FacetArea(mesh)
    diff = sol - f
    error = (m*inner(diff, diff)*ds_t + m*inner(diff, diff)*ds_b + m*inner(diff, diff)*ds_v
             + m*inner(diff('+'), diff('+'))*dS_h + m*inner(diff('+'), diff('+'))*dS_v)

    trace_error = sqrt(assemble(error))
    assert trace_error < 1e-12
