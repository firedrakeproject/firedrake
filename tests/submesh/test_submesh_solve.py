import pytest
from os.path import abspath, dirname, join
import numpy as np
from firedrake import *
from firedrake.cython import dmcommon


cwd = abspath(dirname(__file__))


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
    mesh.mark_entities(indicator_function, 999)
    mesh = Submesh(mesh, dmcommon.CELL_SETS_LABEL, 999, mesh.topological_dimension())
    suberror = _solve_helmholtz(mesh)
    assert abs(error - suberror) < 1e-15


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('simplex', [True, False])
def test_submesh_solve_cell_cell_mixed_scalar(dim, simplex):
    if dim == 2:
        if simplex:
            mesh = Mesh("./docs/notebooks/stokes-control.msh")
            bid = (1, 2, 3, 4, 5)
            submesh_expr = lambda x: conditional(x[0] < 10., 1, 0)
            solution_expr = lambda x: x[0] + x[1]
        else:
            mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"))
            HDivTrace0 = FunctionSpace(mesh, "HDiv Trace", 0)
            x, y = SpatialCoordinate(mesh)
            hdivtrace0x = Function(HDivTrace0).interpolate(conditional(And(x > .001, x < .999), 0, 1))
            hdivtrace0y = Function(HDivTrace0).interpolate(conditional(And(y > .001, y < .999), 0, 1))
            mesh = RelabeledMesh(mesh, [hdivtrace0x, hdivtrace0y], [111, 222])
            bid = (111, 222)
            submesh_expr = lambda x: conditional(x[0] < .5, 1, 0)
            solution_expr = lambda x: x[0] + x[1]
    elif dim == 3:
        if simplex:
            nref = 3
            mesh = BoxMesh(2 ** nref, 2 ** nref, 2 ** nref, 1., 1., 1., hexahedral=False)
            HDivTrace0 = FunctionSpace(mesh, "HDiv Trace", 0)
        else:
            mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"))
            HDivTrace0 = FunctionSpace(mesh, "Q", 2)
        x, y, z = SpatialCoordinate(mesh)
        hdivtrace0x = Function(HDivTrace0).interpolate(conditional(And(x > .001, x < .999), 0, 1))
        hdivtrace0y = Function(HDivTrace0).interpolate(conditional(And(y > .001, y < .999), 0, 1))
        hdivtrace0z = Function(HDivTrace0).interpolate(conditional(And(z > .001, z < .999), 0, 1))
        mesh = RelabeledMesh(mesh, [hdivtrace0x, hdivtrace0y, hdivtrace0z], [111, 222, 333])
        bid = (111, 222, 333)
        submesh_expr = lambda x: conditional(x[0] > .5, 1, 0)
        solution_expr = lambda x: x[0] + x[1] + x[2]
    else:
        raise NotImplementedError
    DG0 = FunctionSpace(mesh, "DG", 0)
    submesh_function = Function(DG0).interpolate(submesh_expr(SpatialCoordinate(mesh)))
    submesh_label = 999
    mesh.mark_entities(submesh_function, submesh_label)
    subm = Submesh(mesh, dmcommon.CELL_SETS_LABEL, submesh_label, mesh.topological_dimension())
    V0 = FunctionSpace(mesh, "CG", 2)
    V1 = FunctionSpace(subm, "CG", 3)
    V = V0 * V1
    u = TrialFunction(V)
    v = TestFunction(V)
    u0, u1 = split(u)
    v0, v1 = split(v)
    dx0 = Measure("cell", domain=mesh)
    dx1 = Measure("cell", domain=subm)
    a = inner(grad(u0), grad(v0)) * dx0 + inner(u0 - u1, v1) * dx1
    L = inner(Constant(0.), v1) * dx1
    g = Function(V0).interpolate(solution_expr(SpatialCoordinate(mesh)))
    bc = DirichletBC(V.sub(0), g, bid)
    solution = Function(V)
    solve(a == L, solution, bcs=[bc])
    target = Function(V1).interpolate(solution_expr(SpatialCoordinate(subm)))
    assert np.allclose(solution.subfunctions[1].dat.data_ro_with_halos, target.dat.data_ro_with_halos)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('simplex', [True, False])
def test_submesh_solve_cell_cell_mixed_vector(dim, simplex):
    if dim == 2:
        if simplex:
            mesh = Mesh("./docs/notebooks/stokes-control.msh")
            submesh_expr = lambda x: conditional(x[0] < 10., 1, 0)
            solution_expr = lambda x: x
            elem0 = FiniteElement("RT", "triangle", 3)
            elem1 = VectorElement("P", "triangle", 3)
        else:
            mesh = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"))
            HDivTrace0 = FunctionSpace(mesh, "HDiv Trace", 0)
            x, y = SpatialCoordinate(mesh)
            hdivtrace0x = Function(HDivTrace0).interpolate(conditional(And(x > .001, x < .999), 0, 1))
            hdivtrace0y = Function(HDivTrace0).interpolate(conditional(And(y > .001, y < .999), 0, 1))
            mesh = RelabeledMesh(mesh, [hdivtrace0x, hdivtrace0y], [111, 222])
            submesh_expr = lambda x: conditional(x[0] < .5, 1, 0)
            solution_expr = lambda x: x
            elem0 = FiniteElement("RTCF", "quadrilateral", 2)
            elem1 = VectorElement("Q", "quadrilateral", 3)
    elif dim == 3:
        if simplex:
            nref = 3
            mesh = BoxMesh(2 ** nref, 2 ** nref, 2 ** nref, 1., 1., 1., hexahedral=False)
            x, y, z = SpatialCoordinate(mesh)
            HDivTrace0 = FunctionSpace(mesh, "HDiv Trace", 0)
            hdivtrace0x = Function(HDivTrace0).interpolate(conditional(And(x > .001, x < .999), 0, 1))
            hdivtrace0y = Function(HDivTrace0).interpolate(conditional(And(y > .001, y < .999), 0, 1))
            hdivtrace0z = Function(HDivTrace0).interpolate(conditional(And(z > .001, z < .999), 0, 1))
            mesh = RelabeledMesh(mesh, [hdivtrace0x, hdivtrace0y, hdivtrace0z], [111, 222, 333])
            submesh_expr = lambda x: conditional(x[0] > .5, 1, 0)
            solution_expr = lambda x: x
            elem0 = FiniteElement("N1F", "tetrahedron", 3)
            elem1 = VectorElement("P", "tetrahedron", 3)
        else:
            mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"))
            HDivTrace0 = FunctionSpace(mesh, "Q", 2)
            x, y, z = SpatialCoordinate(mesh)
            hdivtrace0x = Function(HDivTrace0).interpolate(conditional(And(x > .001, x < .999), 0, 1))
            hdivtrace0y = Function(HDivTrace0).interpolate(conditional(And(y > .001, y < .999), 0, 1))
            hdivtrace0z = Function(HDivTrace0).interpolate(conditional(And(z > .001, z < .999), 0, 1))
            mesh = RelabeledMesh(mesh, [hdivtrace0x, hdivtrace0y, hdivtrace0z], [111, 222, 333])
            elem0 = FiniteElement("NCF", "hexahedron", 2)
            elem1 = VectorElement("Q", "hexahedron", 3)
            submesh_expr = lambda x: conditional(x[0] > .5, 1, 0)
            solution_expr = lambda x: x
            with pytest.raises(NotImplementedError):
                notImplementedV = FunctionSpace(mesh, elem0)
            return
    else:
        raise NotImplementedError
    DG0 = FunctionSpace(mesh, "DG", 0)
    submesh_function = Function(DG0).interpolate(submesh_expr(SpatialCoordinate(mesh)))
    submesh_label = 999
    mesh.mark_entities(submesh_function, submesh_label)
    subm = Submesh(mesh, dmcommon.CELL_SETS_LABEL, submesh_label, mesh.topological_dimension())
    V0 = FunctionSpace(mesh, elem0)
    V1 = FunctionSpace(subm, elem1)
    V = V0 * V1
    u = TrialFunction(V)
    v = TestFunction(V)
    u0, u1 = split(u)
    v0, v1 = split(v)
    dx0 = Measure("cell", domain=mesh)
    dx1 = Measure("cell", domain=subm)
    a = inner(u0, v0) * dx0 + inner(u0 - u1, v1) * dx1
    L = inner(SpatialCoordinate(mesh), v0) * dx0
    solution = Function(V)
    solve(a == L, solution)
    s0, s1 = split(solution)
    x = SpatialCoordinate(subm)
    assert assemble(inner(s1 - x, s1 - x) * dx1) < 1.e-20
