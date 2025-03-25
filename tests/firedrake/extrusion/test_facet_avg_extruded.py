from firedrake import *
import numpy
import pytest


def form(test, trial, degree=None):
    _ds = ds_v(degree=degree) + ds_t(degree=degree) + ds_b(degree=degree)
    _dS = dS_v(degree=degree) + dS_h(degree=degree)
    return inner(trial, test)*_ds + inner(avg(trial), avg(test))*_dS


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
    a = form(test, trial, degree=2*degree)
    l = form(test, facet_avg(source), degree=degree)
    solve(a == l, ft, solver_parameters={"pc_type": "lu", "ksp_type": "preonly"})

    # reference solution
    # Do global projection into P0 trace space.
    ft_ref = Function(Vt, name='ref_sol')
    Vt0 = FunctionSpace(mesh, 'DGT', 0)
    ft_ref_p0 = Function(Vt0, name='ref_sol')

    v = TestFunction(Vt0)
    u = TrialFunction(Vt0)
    a0 = form(v, u, degree=0)
    L0 = form(v, source, degree=degree)
    solve(a0 == L0, ft_ref_p0, solver_parameters={"pc_type": "lu", "ksp_type": "preonly"})

    l_ref = form(test, ft_ref_p0, degree=degree)
    solve(a == l_ref, ft_ref, solver_parameters={"pc_type": "lu", "ksp_type": "preonly"})

    assert numpy.allclose(ft_ref.dat.data_ro, ft.dat.data_ro)
