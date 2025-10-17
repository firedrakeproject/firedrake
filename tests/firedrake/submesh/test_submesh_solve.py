import os
import pytest
from os.path import abspath, dirname, join
import numpy as np
from firedrake import *
from firedrake.cython import dmcommon
from petsc4py import PETSc


cwd = abspath(dirname(__file__))


# Solve an uncoupled system of two equations in one shot.
def test_submesh_solve_uncoupled():
    mesh0 = UnitTriangleMesh()
    mesh1 = UnitSquareMesh(1, 1, quadrilateral=True)
    V0 = FunctionSpace(mesh0, "P", 1)
    V1 = FunctionSpace(mesh1, "Q", 1)
    V = V0 * V1
    u = TrialFunction(V)
    v = TestFunction(V)
    u0, u1 = split(u)
    v0, v1 = split(v)
    dx0 = Measure(
        "dx", domain=mesh0,
    )
    dx1 = Measure(
        "dx", domain=mesh1,
    )
    a = (
        inner(u0, v0) * dx0
        + inner(u1, v1) * dx1
    )
    L = (
        inner(Constant(3.), v0) * dx0
        + inner(Constant(4.), v1) * dx1
    )
    solution = Function(V)
    solve(a == L, solution)
    assert np.allclose(solution.subfunctions[0].dat.data_ro, 3.)
    assert np.allclose(solution.subfunctions[1].dat.data_ro, 4.)


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
    dim = 2
    # Compute reference error.
    mesh = RectangleMesh(nelem, nelem * 2, 1., 1., quadrilateral=True, distribution_parameters=distribution_parameters)
    error = _solve_helmholtz(mesh)
    # Compute submesh error.
    mesh = RectangleMesh(nelem * 2, nelem * 2, 2., 1., quadrilateral=True, distribution_parameters=distribution_parameters)
    x, y = SpatialCoordinate(mesh)
    DQ0 = FunctionSpace(mesh, "DQ", 0)
    indicator_function = Function(DQ0).interpolate(conditional(x < 1., 1, 0))
    mesh.mark_entities(indicator_function, 999)
    mesh = Submesh(mesh, dim, 999)
    suberror = _solve_helmholtz(mesh)
    assert abs(error - suberror) < 1e-15


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('simplex', [True, False])
def test_submesh_solve_cell_cell_mixed_scalar(dim, simplex):
    if dim == 2:
        if simplex:
            mesh = Mesh(join(cwd, "..", "..", "..", "docs", "notebooks/stokes-control.msh"))
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
    subm = Submesh(mesh, dim, submesh_label)
    V0 = FunctionSpace(mesh, "CG", 2)
    V1 = FunctionSpace(subm, "CG", 3)
    V = V0 * V1
    u = TrialFunction(V)
    v = TestFunction(V)
    u0, u1 = split(u)
    v0, v1 = split(v)
    dx0 = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", subm),))
    dx1 = Measure("dx", domain=subm, intersect_measures=(Measure("dx", mesh),))
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
            mesh = Mesh(join(cwd, "..", "..", "..", "docs", "notebooks/stokes-control.msh"))
            submesh_expr = lambda x: conditional(x[0] < 10., 1, 0)
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
            submesh_expr = lambda x: conditional(x[0] > .5, 1, 0)
            elem0 = FiniteElement("NCF", "hexahedron", 2)
            elem1 = VectorElement("Q", "hexahedron", 3)
            with pytest.raises(NotImplementedError):
                _ = FunctionSpace(mesh, elem0)
            return
    else:
        raise NotImplementedError
    DG0 = FunctionSpace(mesh, "DG", 0)
    submesh_function = Function(DG0).interpolate(submesh_expr(SpatialCoordinate(mesh)))
    submesh_label = 999
    mesh.mark_entities(submesh_function, submesh_label)
    subm = Submesh(mesh, dim, submesh_label)
    V0 = FunctionSpace(mesh, elem0)
    V1 = FunctionSpace(subm, elem1)
    V = V0 * V1
    u = TrialFunction(V)
    v = TestFunction(V)
    u0, u1 = split(u)
    v0, v1 = split(v)
    dx0 = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", subm),))
    dx1 = Measure("dx", domain=subm, intersect_measures=(Measure("dx", mesh),))
    a = inner(u0, v0) * dx0 + inner(u0 - u1, v1) * dx1
    L = inner(SpatialCoordinate(mesh), v0) * dx0
    solution = Function(V)
    solve(a == L, solution)
    s0, s1 = split(solution)
    x = SpatialCoordinate(subm)
    assert assemble(inner(s1 - x, s1 - x) * dx1) < 1.e-20


def _mixed_poisson_create_mesh_2d(nref, quadrilateral, submesh_region, label_submesh, label_submesh_compl):
    #        y
    #        |
    #        |
    #  1.0   +--17---+--18---+
    #        |       |       |
    #       12      20       14
    #        |       |       |
    #  0.5   +--21---+--22---+
    #        |       |       |
    #       11      19       13
    #        |       |       |
    #  0.0   +--15---+--16---+----x
    #
    #       0.0     0.5     1.0
    mesh = UnitSquareMesh(2 ** nref, 2 ** nref, quadrilateral=quadrilateral)
    eps = 1. / (2 ** nref) / 100.
    x, y = SpatialCoordinate(mesh)
    HDivTrace0 = FunctionSpace(mesh, "HDiv Trace", 0)
    f11 = Function(HDivTrace0).interpolate(conditional(And(x < eps, y < .5), 1, 0))
    f12 = Function(HDivTrace0).interpolate(conditional(And(x < eps, y > .5), 1, 0))
    f13 = Function(HDivTrace0).interpolate(conditional(And(x > 1 - eps, y < .5), 1, 0))
    f14 = Function(HDivTrace0).interpolate(conditional(And(x > 1 - eps, y > .5), 1, 0))
    f15 = Function(HDivTrace0).interpolate(conditional(And(x < .5, y < eps), 1, 0))
    f16 = Function(HDivTrace0).interpolate(conditional(And(x > .5, y < eps), 1, 0))
    f17 = Function(HDivTrace0).interpolate(conditional(And(x < .5, y > 1 - eps), 1, 0))
    f18 = Function(HDivTrace0).interpolate(conditional(And(x > .5, y > 1 - eps), 1, 0))
    f19 = Function(HDivTrace0).interpolate(conditional(And(And(x > .5 - eps, x < .5 + eps), y < .5), 1, 0))
    f20 = Function(HDivTrace0).interpolate(conditional(And(And(x > .5 - eps, x < .5 + eps), y > .5), 1, 0))
    f21 = Function(HDivTrace0).interpolate(conditional(And(x < .5, And(y > .5 - eps, y < .5 + eps)), 1, 0))
    f22 = Function(HDivTrace0).interpolate(conditional(And(x > .5, And(y > .5 - eps, y < .5 + eps)), 1, 0))
    DG0 = FunctionSpace(mesh, "DG", 0)
    if submesh_region == "left":
        submesh_function = Function(DG0).interpolate(conditional(x < .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(x > .5, 1, 0))
    elif submesh_region == "right":
        submesh_function = Function(DG0).interpolate(conditional(x > .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(x < .5, 1, 0))
    elif submesh_region == "bottom":
        submesh_function = Function(DG0).interpolate(conditional(y < .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(y > .5, 1, 0))
    elif submesh_region == "top":
        submesh_function = Function(DG0).interpolate(conditional(y > .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(y < .5, 1, 0))
    else:
        raise NotImplementedError(f"Unknown submesh_region: {submesh_region}")
    return RelabeledMesh(mesh, [f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, submesh_function, submesh_function_compl],
                               [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, label_submesh, label_submesh_compl])


def _mixed_poisson_solve_2d(nref, degree, quadrilateral, submesh_region):
    dim = 2
    label_submesh = 999
    label_submesh_compl = 888
    mesh = _mixed_poisson_create_mesh_2d(nref, quadrilateral, submesh_region, label_submesh, label_submesh_compl)
    x, y = SpatialCoordinate(mesh)
    subm = Submesh(mesh, dim, label_submesh)
    subx, suby = SpatialCoordinate(subm)
    if submesh_region == "left":
        boun_ext = (11, 12)
        boun_int = (19, 20)
        boun_dirichlet = (15, 17)
    elif submesh_region == "right":
        boun_ext = (13, 14)
        boun_int = (19, 20)
        boun_dirichlet = (16, 18)
    elif submesh_region == "bottom":
        boun_ext = (15, 16)
        boun_int = (21, 22)
        boun_dirichlet = (11, 13)
    elif submesh_region == "top":
        boun_ext = (17, 18)
        boun_int = (21, 22)
        boun_dirichlet = (12, 14)
    else:
        raise NotImplementedError(f"Unknown submesh_region: {submesh_region}")
    BDM = FunctionSpace(subm, "RTCF" if quadrilateral else "BDM", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = BDM * DG
    tau, v = TestFunctions(W)
    nsub = FacetNormal(subm)
    u_exact = Function(DG).interpolate(cos(2 * pi * x) * cos(2 * pi * y))
    sigma_exact = Function(BDM).project(as_vector([- 2 * pi * sin(2 * pi * subx) * cos(2 * pi * suby), - 2 * pi * cos(2 * pi * subx) * sin(2 * pi * suby)]),
                                        solver_parameters={"ksp_type": "cg", "ksp_rtol": 1.e-16})
    f = Function(DG).interpolate(- 8 * pi * pi * cos(2 * pi * x) * cos(2 * pi * y))
    dx0 = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", subm),))
    dx1 = Measure("dx", domain=subm, intersect_measures=(Measure("dx", mesh),))
    ds0 = Measure("ds", domain=mesh, intersect_measures=(Measure("ds", subm),))
    ds1_ext = Measure("ds", domain=subm, intersect_measures=(Measure("ds", mesh),))
    ds1_int = Measure("ds", domain=subm, intersect_measures=(Measure("dS", mesh),))
    dS0 = Measure("dS", domain=mesh, intersect_measures=(Measure("ds", subm),))
    bc = DirichletBC(W.sub(0), sigma_exact, boun_dirichlet)
    # Do the base case.
    w = Function(W)
    sigma, u = split(w)
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx1 + inner(u - u_exact, v) * dx0(label_submesh_compl)
    L = inner(f, v) * dx1 + inner((u('+') + u('-')) / 2., dot(tau, nsub)) * dS0(boun_int) + inner(u_exact, dot(tau, nsub)) * ds0(boun_ext)
    solve(a - L == 0, w, bcs=[bc])
    # Change domains of integration.
    w_ = Function(W)
    sigma_, u_ = split(w_)
    a_ = (inner(sigma_, tau) + inner(u_, div(tau)) + inner(div(sigma_), v)) * dx1 + inner(u_ - u_exact, v) * dx0(label_submesh_compl)
    L_ = inner(f, v) * dx0(label_submesh) + inner((u_('+') + u_('-')) / 2., dot(tau, nsub)) * ds1_int(boun_int) + inner(u_exact, dot(tau, nsub)) * ds1_ext(boun_ext)
    solve(a_ - L_ == 0, w_, bcs=[bc])
    assert assemble(inner(sigma_ - sigma, sigma_ - sigma) * dx1) < 1.e-20
    assert assemble(inner(u_ - u, u_ - u) * dx0(label_submesh)) < 1.e-20
    sigma_error = sqrt(assemble(inner(sigma - sigma_exact, sigma - sigma_exact) * dx1))
    u_error = sqrt(assemble(inner(u - u_exact, u - u_exact) * dx0(label_submesh)))
    return sigma_error, u_error


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('nref', [1, 2, 3, 4])
@pytest.mark.parametrize('degree', [1])
@pytest.mark.parametrize('quadrilateral', [False, True])
@pytest.mark.parametrize('submesh_region', ["left", "right", "bottom", "top"])
def test_submesh_solve_mixed_poisson_check_sanity_2d(nref, degree, quadrilateral, submesh_region):
    _, _ = _mixed_poisson_solve_2d(nref, degree, quadrilateral, submesh_region)


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('quadrilateral', [True])
@pytest.mark.parametrize('degree', [3])
@pytest.mark.parametrize('submesh_region', ["left", "right"])
def test_submesh_solve_mixed_poisson_check_convergence_2d(quadrilateral, degree, submesh_region):
    nrefs = [5, 6, 7]
    start = nrefs[0]
    s_error_array = np.zeros(len(nrefs))
    u_error_array = np.zeros(len(nrefs))
    for nref in nrefs:
        i = nref - start
        s_error_array[i], u_error_array[i] = _mixed_poisson_solve_2d(nref, degree, quadrilateral, submesh_region)
    assert (np.log2(s_error_array[:-1] / s_error_array[1:]) > degree + .95).all()
    assert (np.log2(u_error_array[:-1] / u_error_array[1:]) > degree + .95).all()


def _mixed_poisson_create_mesh_3d(hexahedral, submesh_region, label_submesh, label_submesh_compl):
    if hexahedral:
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"))
        DG0 = FunctionSpace(mesh, "DQ", 0)
        HDivTrace0 = FunctionSpace(mesh, "Q", 2)
    else:
        mesh = BoxMesh(4, 4, 4, 1., 1., 1., hexahedral=False)
        DG0 = FunctionSpace(mesh, "DP", 0)
        HDivTrace0 = FunctionSpace(mesh, "HDiv Trace", 0)
    x, y, z = SpatialCoordinate(mesh)
    eps = 1.e-6
    f101 = Function(HDivTrace0).interpolate(conditional(x < eps, 1, 0))
    f102 = Function(HDivTrace0).interpolate(conditional(x > 1. - eps, 1, 0))
    f103 = Function(HDivTrace0).interpolate(conditional(y < eps, 1, 0))
    f104 = Function(HDivTrace0).interpolate(conditional(y > 1. - eps, 1, 0))
    f105 = Function(HDivTrace0).interpolate(conditional(z < eps, 1, 0))
    f106 = Function(HDivTrace0).interpolate(conditional(z > 1. - eps, 1, 0))
    if submesh_region == "left":
        submesh_function = Function(DG0).interpolate(conditional(x < .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(x > .5, 1, 0))
    elif submesh_region == "right":
        submesh_function = Function(DG0).interpolate(conditional(x > .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(x < .5, 1, 0))
    elif submesh_region == "front":
        submesh_function = Function(DG0).interpolate(conditional(y < .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(y > .5, 1, 0))
    elif submesh_region == "back":
        submesh_function = Function(DG0).interpolate(conditional(y > .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(y < .5, 1, 0))
    elif submesh_region == "bottom":
        submesh_function = Function(DG0).interpolate(conditional(z < .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(z > .5, 1, 0))
    elif submesh_region == "top":
        submesh_function = Function(DG0).interpolate(conditional(z > .5, 1, 0))
        submesh_function_compl = Function(DG0).interpolate(conditional(z < .5, 1, 0))
    else:
        raise NotImplementedError(f"Unknown submesh_region: {submesh_region}")
    return RelabeledMesh(mesh, [f101, f102, f103, f104, f105, f106, submesh_function, submesh_function_compl],
                               [101, 102, 103, 104, 105, 106, label_submesh, label_submesh_compl])


def _mixed_poisson_solve_3d(hexahedral, degree, submesh_region):
    dim = 3
    label_submesh = 999
    label_submesh_compl = 888
    mesh = _mixed_poisson_create_mesh_3d(hexahedral, submesh_region, label_submesh, label_submesh_compl)
    x, y, z = SpatialCoordinate(mesh)
    subm = Submesh(mesh, dim, label_submesh)
    subx, suby, subz = SpatialCoordinate(subm)
    if submesh_region == "left":
        boun_ext = (101, )
        boun_dirichlet = (103, 104, 105, 106)
    elif submesh_region == "right":
        boun_ext = (102, )
        boun_dirichlet = (103, 104, 105, 106)
    elif submesh_region == "front":
        boun_ext = (103, )
        boun_dirichlet = (101, 102, 105, 106)
    elif submesh_region == "back":
        boun_ext = (104, )
        boun_dirichlet = (101, 102, 105, 106)
    elif submesh_region == "bottom":
        boun_ext = (105, )
        boun_dirichlet = (101, 102, 103, 104)
    elif submesh_region == "top":
        boun_ext = (106, )
        boun_dirichlet = (101, 102, 103, 104)
    else:
        raise NotImplementedError(f"Unknown submesh_region: {submesh_region}")
    boun_int = (107, )  # labeled automatically.
    NCF = FunctionSpace(subm, "NCF" if hexahedral else "N2F", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = NCF * DG
    tau, v = TestFunctions(W)
    nsub = FacetNormal(subm)
    u_exact = Function(DG).interpolate(cos(2 * pi * x) * cos(2 * pi * y) * cos(2 * pi * z))
    sigma_exact = Function(NCF).project(as_vector([- 2 * pi * sin(2 * pi * subx) * cos(2 * pi * suby) * cos(2 * pi * subz),
                                                   - 2 * pi * cos(2 * pi * subx) * sin(2 * pi * suby) * cos(2 * pi * subz),
                                                   - 2 * pi * cos(2 * pi * subx) * cos(2 * pi * suby) * sin(2 * pi * subz)]),
                                        solver_parameters={"ksp_type": "cg", "ksp_rtol": 1.e-16})
    f = Function(DG).interpolate(- 12 * pi * pi * cos(2 * pi * x) * cos(2 * pi * y) * cos(2 * pi * z))
    dx0 = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", subm),))
    dx1 = Measure("dx", domain=subm, intersect_measures=(Measure("dx", mesh),))
    ds0 = Measure("ds", domain=mesh, intersect_measures=(Measure("ds", subm),))
    ds1 = Measure("ds", domain=subm, intersect_measures=(Measure("dS", mesh),))
    bc = DirichletBC(W.sub(0), sigma_exact, boun_dirichlet)
    # Do the base case.
    w = Function(W)
    sigma, u = split(w)
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx1 + inner(u - u_exact, v) * dx0(label_submesh_compl)
    L = inner(f, v) * dx1 + inner((u('+') + u('-')) / 2., dot(tau, nsub)) * ds1(boun_int) + inner(u_exact, dot(tau, nsub)) * ds0(boun_ext)
    solve(a - L == 0, w, bcs=[bc])
    sigma_error = sqrt(assemble(inner(sigma - sigma_exact, sigma - sigma_exact) * dx1))
    u_error = sqrt(assemble(inner(u - u_exact, u - u_exact) * dx0(label_submesh)))
    return sigma_error, u_error


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('hexahedral', [False])
@pytest.mark.parametrize('degree', [4])
@pytest.mark.parametrize('submesh_region', ["left", "right", "front", "back", "bottom", "top"])
def test_submesh_solve_mixed_poisson_check_sanity_3d(hexahedral, degree, submesh_region):
    sigma_error, u_error = _mixed_poisson_solve_3d(hexahedral, degree, submesh_region)
    assert sigma_error < 0.07
    assert u_error < 0.003


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('simplex', [True, False])
@pytest.mark.parametrize('nref', [1, 3])
@pytest.mark.parametrize('degree', [2, 4])
def test_submesh_solve_cell_cell_equation_bc(nref, degree, simplex):
    dim = 2
    mesh = RectangleMesh(3 ** nref, 2 ** nref, 3., 2., quadrilateral=not simplex)
    x, y = SpatialCoordinate(mesh)
    label_outer = 101
    label_inner = 100
    label_interface = 5  # automatically labeled by Submesh
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_outer = Function(DG0).interpolate(conditional(Or(Or(x < 1., x > 2.), y > 1.), 1, 0))
    f_inner = Function(DG0).interpolate(conditional(And(And(x > 1., x < 2.), y < 1.), 1, 0))
    mesh = RelabeledMesh(mesh, [f_outer, f_inner], [label_outer, label_inner])
    x, y = SpatialCoordinate(mesh)
    mesh_outer = Submesh(mesh, dim, label_outer)
    x_outer, y_outer = SpatialCoordinate(mesh_outer)
    mesh_inner = Submesh(mesh, dim, label_inner)
    x_inner, y_inner = SpatialCoordinate(mesh_inner)
    V_outer = FunctionSpace(mesh_outer, "CG", degree)
    V_inner = FunctionSpace(mesh_inner, "CG", degree)
    V = V_outer * V_inner
    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)
    u_outer, u_inner = split(u)
    v_outer, v_inner = split(v)
    dx_outer = Measure("dx", domain=mesh_outer, intersect_measures=(Measure("dx", mesh), Measure("dx", mesh_inner)))
    dx_inner = Measure("dx", domain=mesh_inner, intersect_measures=(Measure("dx", mesh), Measure("dx", mesh_outer)))
    ds_outer = Measure("ds", domain=mesh_outer, intersect_measures=(Measure("ds", mesh_inner),))
    a = inner(grad(u_outer), grad(v_outer)) * dx_outer + \
        inner(u_inner, v_inner) * dx_inner
    L = inner(x * y, v_inner) * dx_inner
    dbc = DirichletBC(V.sub(0), x_outer * y_outer, (1, 2, 3, 4))
    ebc = EquationBC(inner(u_outer - u_inner, v_outer) * ds_outer(label_interface) == inner(Constant(0.), v_outer) * ds_outer(label_interface), sol, label_interface, V=V.sub(0))
    solve(a == L, sol, bcs=[dbc, ebc])
    assert sqrt(assemble(inner(sol[0] - x * y, sol[0] - x * y) * dx_outer)) < 1.e-12
    assert sqrt(assemble(inner(sol[1] - x * y, sol[1] - x * y) * dx_inner)) < 1.e-12


def _test_submesh_solve_quad_triangle_poisson(nref, degree):
    dim = 2
    label_ext = 1
    label_interf = 2
    distribution_parameters_noop = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.NONE, 0),
    }
    mesh = Mesh(os.path.join(cwd, "..", "meshes", "mixed_cell_unit_square.msh"), distribution_parameters=distribution_parameters_noop)
    plex = mesh.topology_dm
    for _ in range(nref):
        plex = plex.refine()
    plex.removeLabel("pyop2_core")
    plex.removeLabel("pyop2_owned")
    plex.removeLabel("pyop2_ghost")
    mesh = Mesh(plex)
    h = 0.1 / 2**nref  # roughly
    mesh.topology_dm.markBoundaryFaces(dmcommon.FACE_SETS_LABEL, label_ext)
    mesh_t = Submesh(mesh, dim, PETSc.DM.PolytopeType.TRIANGLE, label_name="celltype", name="mesh_tri")
    x_t, y_t = SpatialCoordinate(mesh_t)
    n_t = FacetNormal(mesh_t)
    mesh_q = Submesh(mesh, dim, PETSc.DM.PolytopeType.QUADRILATERAL, label_name="celltype", name="mesh_quad")
    x_q, y_q = SpatialCoordinate(mesh_q)
    n_q = FacetNormal(mesh_q)
    V_t = FunctionSpace(mesh_t, "P", degree)
    V_q = FunctionSpace(mesh_q, "Q", degree)
    V = V_t * V_q
    u = TrialFunction(V)
    v = TestFunction(V)
    u_t, u_q = split(u)
    v_t, v_q = split(v)
    dx_t = Measure("dx", mesh_t)
    dx_q = Measure("dx", mesh_q)
    ds_t = Measure("ds", mesh_t, intersect_measures=(Measure("ds", mesh_q),))
    ds_q = Measure("ds", mesh_q, intersect_measures=(Measure("ds", mesh_t),))
    g_t = cos(2 * pi * x_t) * cos(2 * pi * y_t)
    g_q = cos(2 * pi * x_q) * cos(2 * pi * y_q)
    f_t = 8 * pi**2 * g_t
    f_q = 8 * pi**2 * g_q
    a = (
        inner(grad(u_t), grad(v_t)) * dx_t + inner(grad(u_q), grad(v_q)) * dx_q
        - inner(
            (grad(u_q) + grad(u_t)) / 2,
            (v_q * n_q + v_t * n_t)
        ) * ds_q(label_interf)
        - inner(
            (u_q * n_q + u_t * n_t),
            (grad(v_q) + grad(v_t)) / 2
        ) * ds_t(label_interf)
        + 100 / h * inner(u_q - u_t, v_q - v_t) * ds_q(label_interf)
    )
    L = (
        inner(f_t, v_t) * dx_t + inner(f_q, v_q) * dx_q
    )
    sol = Function(V)
    bc_q = DirichletBC(V.sub(1), g_q, label_ext)
    solve(a == L, sol, bcs=[bc_q])
    sol_t, sol_q = split(sol)
    L2Error_t = assemble(inner(sol_t - g_t, sol_t - g_t) * dx_t)
    L2Error_q = assemble(inner(sol_q - g_q, sol_q - g_q) * dx_q)
    H1Error_t = L2Error_t + assemble(inner(grad(sol_t - g_t), grad(sol_t - g_t)) * dx_t)
    H1Error_q = L2Error_q + assemble(inner(grad(sol_q - g_q), grad(sol_q - g_q)) * dx_q)
    return sqrt(L2Error_t + L2Error_q), sqrt(H1Error_t + H1Error_q)


@pytest.mark.parallel(nprocs=8)
def test_submesh_solve_quad_triangle_poisson_convergence():
    for degree in range(1, 5):
        L2Errors = []
        H1Errors = []
        for nref in range(4):
            L2Error, H1Error = _test_submesh_solve_quad_triangle_poisson(nref, degree)
            L2Errors.append(L2Error)
            H1Errors.append(H1Error)
        L2Errors = [np.log2(c) - np.log2(f) for c, f in zip(L2Errors[:-1], L2Errors[1:])]
        H1Errors = [np.log2(c) - np.log2(f) for c, f in zip(H1Errors[:-1], H1Errors[1:])]
        assert (np.array(L2Errors) > (degree + 1) * 0.995).all()
        assert (np.array(H1Errors) > (degree) * 0.995).all()


def _test_submesh_solve_3d_2d_poisson(simplex, direction, nref, degree):
    distribution_parameters_noop = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.NONE, 0),
    }
    distribution_parameters = {
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
    }
    dim = 3
    interf_at = 0.499
    if simplex:
        nref_simplex = 3
        mesh = BoxMesh(2 ** nref_simplex, 2 ** nref_simplex, 2 ** nref_simplex, 1., 1., 1., hexahedral=False, distribution_parameters=distribution_parameters_noop)
        xyz = SpatialCoordinate(mesh)
        DG0 = FunctionSpace(mesh, "DG", 0)
        c1 = Function(DG0).interpolate(conditional(xyz[direction] < interf_at, 1, 0))
        c2 = Function(DG0).interpolate(conditional(xyz[direction] > interf_at, 1, 0))
        mesh = RelabeledMesh(mesh, [c1, c2], [1, 2])
        family = "P"
    else:
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"), distribution_parameters=distribution_parameters_noop)
        xyz = SpatialCoordinate(mesh)
        DG0 = FunctionSpace(mesh, "DQ", 0)
        c1 = Function(DG0).interpolate(conditional(xyz[direction] < interf_at, 1, 0))
        c2 = Function(DG0).interpolate(conditional(xyz[direction] > interf_at, 1, 0))
        HDivTrace0 = FunctionSpace(mesh, "Q", 2)
        f1 = Function(HDivTrace0).interpolate(conditional(xyz[0] < .001, 1, 0))
        f2 = Function(HDivTrace0).interpolate(conditional(xyz[0] > .999, 1, 0))
        f3 = Function(HDivTrace0).interpolate(conditional(xyz[1] < .001, 1, 0))
        f4 = Function(HDivTrace0).interpolate(conditional(xyz[1] > .999, 1, 0))
        f5 = Function(HDivTrace0).interpolate(conditional(xyz[2] < .001, 1, 0))
        f6 = Function(HDivTrace0).interpolate(conditional(xyz[2] > .999, 1, 0))
        mesh = RelabeledMesh(mesh, [c1, c2, f1, f2, f3, f4, f5, f6], [1, 2, 1, 2, 3, 4, 5, 6])
        family = "Q"
    plex = mesh.topology_dm
    for _ in range(nref):
        plex = plex.refine()
    plex.removeLabel("pyop2_core")
    plex.removeLabel("pyop2_owned")
    plex.removeLabel("pyop2_ghost")
    mesh = Mesh(plex, distribution_parameters=distribution_parameters)
    mesh1 = Submesh(mesh, dim, 1)
    x1, y1, z1 = SpatialCoordinate(mesh1)
    mesh2 = Submesh(mesh, dim, 2)
    x2, y2, z2 = SpatialCoordinate(mesh2)
    label_interf = 7  # max + 1
    mesh12 = Submesh(mesh2, dim - 1, label_interf)
    dx1 = Measure("dx", mesh1)
    dx2 = Measure("dx", mesh2)
    ds1_ds2 = Measure("ds", mesh1, intersect_measures=(Measure("ds", mesh2),))
    dx12_ds1_ds2 = Measure(
        "dx", mesh12,
        intersect_measures=(
            Measure("ds", mesh1),
            Measure("ds", mesh2),
        )
    )
    # Check sanity.
    vol1 = assemble(Constant(1) * dx1)
    vol2 = assemble(Constant(1) * dx2)
    assert abs(vol1 + vol2 - 1.) < 1.e-13
    # Solve Poisson problem.
    V1 = FunctionSpace(mesh1, family, degree)
    V12 = FunctionSpace(mesh12, family, degree)
    V2 = FunctionSpace(mesh2, family, degree)
    V = V1 * V12 * V2
    u = TrialFunction(V)
    v = TestFunction(V)
    u1, u12, u2 = split(u)
    v1, v12, v2 = split(v)
    g1 = cos(2 * pi * x1) * cos(2 * pi * y1) * cos(2 * pi * z1)
    g2 = cos(2 * pi * x2) * cos(2 * pi * y2) * cos(2 * pi * z2)
    f1 = 12 * pi**2 * g1
    f2 = 12 * pi**2 * g2
    n1 = FacetNormal(mesh1)
    n2 = FacetNormal(mesh2)
    h = 0.1 / 2**nref  # roughly
    a = (
        inner(grad(u1), grad(v1)) * dx1 + inner(grad(u2), grad(v2)) * dx2
        - inner(
            u12,
            (v1 - v2)
        ) * dx12_ds1_ds2
        - inner(
            (u1 * n1 + u2 * n2),
            (grad(v1) + grad(v2)) / 2
        ) * dx12_ds1_ds2
        + 100 / h * inner(u1 - u2, v1 - v2) * ds1_ds2(label_interf)  # Can also use dx12_ds1_ds2.
        + inner(
            (dot(grad(u1), n1) - dot(grad(u2), n2)) / 2 - u12,
            v12
        ) * dx12_ds1_ds2
    )
    L = (
        inner(f1, v1) * dx1 + inner(f2, v2) * dx2
    )
    sol = Function(V)
    bc1 = DirichletBC(V.sub(0), g1, [i for i in range(1, 7) if i != 2 * direction + 2])
    bc2 = DirichletBC(V.sub(2), g2, [i for i in range(1, 7) if i != 2 * direction + 1])
    solver_parameters = {
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_0_fields": "1",
        "pc_fieldsplit_1_fields": "0, 2",
        "fieldsplit_0_ksp_type": "cg",
        "fieldsplit_0_ksp_rtol": 1e-14,
        "fieldsplit_0_pc_type": "jacobi",
        "fieldsplit_1_ksp_type": "cg",
        "fieldsplit_1_ksp_rtol": 1e-14,
        "fieldsplit_1_pc_type": "jacobi",
    }
    solve(a == L, sol, bcs=[bc1, bc2], solver_parameters=solver_parameters)
    sol1, sol12, sol2 = split(sol)
    L2Error1 = assemble(inner(sol1 - g1, sol1 - g1) * dx1)
    L2Error2 = assemble(inner(sol2 - g2, sol2 - g2) * dx2)
    H1Error1 = L2Error1 + assemble(inner(grad(sol1 - g1), grad(sol1 - g1)) * dx1)
    H1Error2 = L2Error2 + assemble(inner(grad(sol2 - g2), grad(sol2 - g2)) * dx2)
    return sqrt(L2Error1 + L2Error2), sqrt(H1Error1 + H1Error2)


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize('simplex', [True, False])
@pytest.mark.parametrize('direction', [0, 1, 2])
def test_submesh_solve_3d_2d_poisson_sanity(simplex, direction):
    nref = 0
    degree = 4
    L2Error, H1Error = _test_submesh_solve_3d_2d_poisson(simplex, direction, nref, degree)
    assert L2Error < 6.e-5
    assert H1Error < 5.e-3


@pytest.mark.parallel(nprocs=8)
@pytest.mark.parametrize('simplex', [False])
@pytest.mark.parametrize('direction', [0])
@pytest.mark.parametrize('degree', [3])
def test_submesh_solve_3d_2d_poisson_convergence(simplex, direction, degree):
    L2Errors = []
    H1Errors = []
    for nref in range(2):
        L2Error, H1Error = _test_submesh_solve_3d_2d_poisson(simplex, direction, nref, degree)
        L2Errors.append(L2Error)
        H1Errors.append(H1Error)
    L2Errors = [np.log2(c) - np.log2(f) for c, f in zip(L2Errors[:-1], L2Errors[1:])]
    H1Errors = [np.log2(c) - np.log2(f) for c, f in zip(H1Errors[:-1], H1Errors[1:])]
    assert (np.array(L2Errors) > (degree + 1) * 0.96).all()
    assert (np.array(H1Errors) > (degree) * 0.96).all()


@pytest.mark.parallel(nprocs=7)
def test_submesh_solve_2d_1d_poisson_hermite():
    distribution_parameters_noop = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.NONE, 0),
    }
    distribution_parameters = {
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
    }
    mesh3d = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"), distribution_parameters=distribution_parameters_noop)
    plex = mesh3d.topology_dm
    for _ in range(2):
        plex = plex.refine()
    plex.removeLabel("pyop2_core")
    plex.removeLabel("pyop2_owned")
    plex.removeLabel("pyop2_ghost")
    mesh3d = Mesh(plex, distribution_parameters=distribution_parameters)
    xyz = SpatialCoordinate(mesh3d)
    HDivTrace0 = FunctionSpace(mesh3d, "Q", 2)
    f1 = Function(HDivTrace0).interpolate(conditional(xyz[0] < .001, 1, 0))
    f2 = Function(HDivTrace0).interpolate(conditional(xyz[0] > .999, 1, 0))
    f3 = Function(HDivTrace0).interpolate(conditional(xyz[1] < .001, 1, 0))
    f4 = Function(HDivTrace0).interpolate(conditional(xyz[1] > .999, 1, 0))
    f5 = Function(HDivTrace0).interpolate(conditional(xyz[2] < .001, 1, 0))
    f6 = Function(HDivTrace0).interpolate(conditional(xyz[2] > .999, 1, 0))
    mesh3d = RelabeledMesh(mesh3d, [f1, f2, f3, f4, f5, f6], [1, 2, 3, 4, 5, 6])
    mesh2d = Submesh(mesh3d, mesh3d.topological_dimension - 1, 6)
    mesh1d = Submesh(mesh2d, mesh2d.topological_dimension - 1, 4)
    x2d = SpatialCoordinate(mesh2d)
    x1d = SpatialCoordinate(mesh1d)
    g2d = sin(2 * pi * x2d[0]) * sin(2 * pi * x2d[1])
    f2d = 2 * (2 * pi)**2 * g2d
    g1d = sin(2 * pi * x1d[0])
    f1d = (2 * pi)**4 * g1d
    V2d = FunctionSpace(mesh2d, "Q", 3)
    V1d = FunctionSpace(mesh1d, "Hermite", 3)
    V = V2d * V1d
    u = TrialFunction(V)
    v = TestFunction(V)
    u2d, u1d = split(u)
    v2d, v1d = split(v)
    dx2d = Measure("dx", mesh2d)
    dx1d = Measure("dx", mesh1d)
    ds2d_dx1d = Measure("ds", mesh2d, intersect_measures=(Measure("dx", mesh1d),))
    a = inner(grad(u2d), grad(v2d)) * dx2d - \
        inner(2 * pi * u1d, v2d) * ds2d_dx1d(4) + \
        inner(grad(grad(u1d)), grad(grad(v1d))) * dx1d
    L = inner(f2d, v2d) * dx2d + \
        inner(f1d, v1d) * dx1d
    sol = Function(V)
    bc2d = DirichletBC(V.sub(0), g2d, (1, 2, 3))
    bc1d = DirichletBC(V.sub(1), g1d, (1, 2))
    solve(a == L, sol, bcs=[bc2d, bc1d])
    error2d = assemble(inner(sol[0] - g2d, sol[0] - g2d) * dx2d)
    error1d = assemble(inner(sol[1] - g1d, sol[1] - g1d) * dx1d)
    assert sqrt(error2d) < 1.e-5
    assert sqrt(error1d) < 5.e-5
