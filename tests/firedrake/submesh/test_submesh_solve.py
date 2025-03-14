import pytest
from os.path import abspath, dirname, join
import numpy as np
from firedrake import *


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
    dx0 = Measure("dx", domain=mesh)
    dx1 = Measure("dx", domain=subm)
    ds0 = Measure("ds", domain=mesh)
    ds1 = Measure("ds", domain=subm)
    dS0 = Measure("dS", domain=mesh)
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
    L_ = inner(f, v) * dx0(label_submesh) + inner((u_('+') + u_('-')) / 2., dot(tau, nsub)) * ds1(boun_int) + inner(u_exact, dot(tau, nsub)) * ds1(boun_ext)
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
    dx0 = Measure("dx", domain=mesh)
    dx1 = Measure("dx", domain=subm)
    ds0 = Measure("ds", domain=mesh)
    ds1 = Measure("ds", domain=subm)
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
    dx_outer = Measure("dx", domain=mesh_outer)
    dx_inner = Measure("dx", domain=mesh_inner)
    ds_outer = Measure("ds", domain=mesh_outer)
    a = inner(grad(u_outer), grad(v_outer)) * dx_outer + \
        inner(u_inner, v_inner) * dx_inner
    L = inner(x * y, v_inner) * dx_inner
    dbc = DirichletBC(V.sub(0), x_outer * y_outer, (1, 2, 3, 4))
    ebc = EquationBC(inner(u_outer - u_inner, v_outer) * ds_outer(label_interface) == inner(Constant(0.), v_outer) * ds_outer(label_interface), sol, label_interface, V=V.sub(0))
    solve(a == L, sol, bcs=[dbc, ebc])
    assert sqrt(assemble(inner(sol[0] - x * y, sol[0] - x * y) * dx_outer)) < 1.e-12
    assert sqrt(assemble(inner(sol[1] - x * y, sol[1] - x * y) * dx_inner)) < 1.e-12
