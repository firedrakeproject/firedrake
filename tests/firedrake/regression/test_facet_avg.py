from firedrake import *
import numpy
import pytest


def form(test, trial, degree=None):
    return inner(trial, test)*ds(degree=degree) + inner(avg(trial), avg(test))*dS(degree=degree)


@pytest.fixture(params=["triangle", "quadrilateral", "tetrahedron"],
                scope="module")
def mesh(request):
    if request.param == "triangle":
        return UnitSquareMesh(10, 10, quadrilateral=False)
    elif request.param == "quadrilateral":
        return UnitSquareMesh(10, 10, quadrilateral=True)
    elif request.param == "tetrahedron":
        return UnitCubeMesh(5, 3, 6)
    else:
        raise ValueError("Unhandled mesh request")


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_facet_avg(degree, mesh):

    Vt = FunctionSpace(mesh, 'DGT', degree)
    ft = Function(Vt, name='f_trace')

    X = SpatialCoordinate(mesh)
    source = sum(c * scale for c, scale in zip(X, [2, -10, 12]))
    source = source**degree

    test = TestFunction(Vt)
    trial = TrialFunction(Vt)
    a = form(test, trial)
    l = form(test, facet_avg(source))
    solve(a == l, ft, solver_parameters={"pc_type": "lu", "ksp_type": "preonly"})

    # reference solution
    # Do global projection into P0 trace space.
    ft_ref = Function(Vt, name='ref_sol')
    Vt0 = FunctionSpace(mesh, 'DGT', 0)
    ft_ref_p0 = Function(Vt0, name='ref_sol')

    v = TestFunction(Vt0)
    u = TrialFunction(Vt0)
    a0 = form(v, u)
    L0 = form(v, source)
    solve(a0 == L0, ft_ref_p0, solver_parameters={"pc_type": "lu", "ksp_type": "preonly"})

    l_ref = form(test, ft_ref_p0)
    solve(a == l_ref, ft_ref, solver_parameters={"pc_type": "lu", "ksp_type": "preonly"})

    assert numpy.allclose(ft_ref.dat.data_ro, ft.dat.data_ro)
