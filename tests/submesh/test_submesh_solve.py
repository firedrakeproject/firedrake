import pytest
from firedrake import *
from firedrake.cython import dmcommon


def _solve_helmholtz(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    u_exact = sin(x[0]) * sin(x[1])
    f = Function(V).interpolate(2 * u_exact)
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx
    bc = DirichletBC(V, u_exact, "on_boundary")
    sol = Function(V)
    solve(a == L, sol, bcs=[bc], solver_parameters={'ksp_type': 'preonly',
                                                    'pc_type': 'lu'})
    return sqrt(assemble((sol - u_exact)**2 * dx))


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('nelem', [2, 4])
@pytest.mark.parametrize('distribution_parameters', [None, {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}])
def test_submesh_solve_simple(nelem, distribution_parameters):
    # Compute reference error.
    mesh = RectangleMesh(nelem, nelem * 2, 1., 1., quadrilateral=True, distribution_parameters=distribution_parameters)
    error = _solve_helmholtz(mesh)
    # Compute submesh error.
    mesh = RectangleMesh(nelem * 2, nelem * 2, 2., 1., quadrilateral=True, distribution_parameters=distribution_parameters)
    x, y = SpatialCoordinate(mesh)
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(conditional(x < 1., 1, 0))
    mesh.mark_entities(indicator_function, dmcommon.CELL_SETS_LABEL, 999)
    mesh = Submesh(mesh, dmcommon.CELL_SETS_LABEL, 999, mesh.topological_dimension())
    suberror = _solve_helmholtz(mesh)
    assert abs(error - suberror) < 1e-15
