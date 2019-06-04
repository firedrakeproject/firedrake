from firedrake import *
import numpy
import pytest


@pytest.fixture(params=[True, False],
                ids=["hex", "prism"],
                scope="module")
def mesh(request):
    mesh2d = UnitSquareMesh(4, 4, quadrilateral=request.param)
    return ExtrudedMesh(mesh2d, 5, 0.1)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_facet_avg_extruded(mesh, degree):
    Vt = FunctionSpace(mesh, 'DGT', degree)
    ft = Function(Vt, name='f_trace')

    x, y, z = SpatialCoordinate(mesh)
    source = (2*x + z - y*10)**degree

    test = TestFunction(Vt)
    trial = TrialFunction(Vt)
    a = inner(trial, test)*(ds_v + ds_t + ds_b) + inner(avg(trial), avg(test))*(dS_v + dS_h)
    l = inner(facet_avg(source), test)*(ds_v + ds_t + ds_b) + inner(avg(facet_avg(source)), avg(test))*(dS_v + dS_h)

    solve(a == l, ft, solver_parameters={"pc_type": "lu", "ksp_type": "preonly"})

    # reference solution
    # Do global projection into P0 trace space.
    ft_ref = Function(Vt, name='ref_sol')
    Vt0 = FunctionSpace(mesh, 'DGT', 0)
    ft_ref_p0 = Function(Vt0, name='ref_sol')

    v = TestFunction(Vt0)
    u = TrialFunction(Vt0)
    a0 = inner(u, v)*(ds_v + ds_t + ds_b) + inner(avg(u), avg(v))*(dS_v + dS_h)
    L0 = inner(source, v)*(ds_v + ds_t + ds_b) + inner(avg(source), avg(v))*(dS_v + dS_h)
    solve(a0 == L0, ft_ref_p0, solver_parameters={"pc_type": "lu", "ksp_type": "preonly"})

    l_ref = inner(ft_ref_p0, test)*(ds_v + ds_t + ds_b) + inner(avg(ft_ref_p0), avg(test))*(dS_v + dS_h)
    solve(a == l_ref, ft_ref, solver_parameters={"pc_type": "lu", "ksp_type": "preonly"})

    assert numpy.allclose(ft_ref.dat.data_ro, ft.dat.data_ro)
