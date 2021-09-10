import pytest
from firedrake import *


@pytest.fixture(params=[2, 3],
                ids=["Rectangle", "Box"])
def mesh(request):
    nx = 4
    distribution = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    m = UnitSquareMesh(nx, nx, quadrilateral=True, distribution_parameters=distribution)
    if request.param == 3:
        layers = layers_array = np.array([[0, nx]]*(nx*nx))
        # m = ExtrudedMesh(m, layers=layers, layer_height=1/nx)
        m = ExtrudedMesh(m, nx)
    x = SpatialCoordinate(m)
    xnew = as_vector([acos(1-2*xj)/pi for xj in x])
    m.coordinates.interpolate(xnew)
    return m


@pytest.fixture
def expected(mesh):
    if mesh.topological_dimension() == 2:
        return [4, 4, 4]
    elif mesh.topological_dimension() == 3:
        return [7, 7, 7]


@pytest.mark.skipcomplex
def not_test_p_independence(mesh, expected):
    nits = []
    for p in range(3, 6):
        V = FunctionSpace(mesh, "Lagrange", p)

        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(grad(u), grad(v))*dx

        L = inner(Constant(1), v)*dx

        asm = "firedrake.ASMStarPC"
        subs = ("on_boundary",)
        if mesh.topological_dimension() == 3:
            asm = "firedrake.ASMExtrudedStarPC"
            subs += ("top", "bottom")
        bcs = [DirichletBC(V, zero(V.ufl_element().value_shape()), sub) for sub in subs]

        uh = Function(V)
        problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

        solver = LinearVariationalSolver(problem, solver_parameters={
            "mat_type": "matfree",
            "ksp_type": "cg",
            "ksp_atol": 0.0E0,
            "ksp_rtol": 1.0E-8,
            "ksp_converged_reason": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.P1PC",
            "pmg_mg_levels": {
                "ksp_type": "chebyshev",
                "esteig_ksp_type": "cg",
                "esteig_ksp_norm_type": "unpreconditioned",
                "ksp_chebyshev_esteig": "0.75,0.25,0.0,1.0",
                "ksp_chebyshev_esteig_noisy": True,
                "ksp_chebyshev_esteig_steps": 7,
                "ksp_norm_type": "unpreconditioned",
                "ksp_monitor_true_residual": None,
                "pc_type": "python",
                "pc_python_type": "firedrake.FDMPC",
                "fdm": {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": asm,
                    "pc_star_backend": "petscasm",
                    "pc_star_sub_sub_ksp_type": "preonly",
                    "pc_star_sub_sub_pc_type": "cholesky",
                }
            },
            "pmg_mg_coarse": {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled_pc_type": "cholesky",
            },
            "ksp_monitor": None})

        solver.solve()

        nits.append(solver.snes.ksp.getIterationNumber())
    assert (nits == expected)


def outer_jump(v, n):
    return outer(v('+'), n('+'))+outer(v('-'), n('-'))


@pytest.fixture(params=["cg", "dg", "rt"],
                ids=["cg", "dg", "rt"])
def fs(request, mesh):
    degree = 3
    ndim = mesh.topological_dimension()
    element = request.param
    if element == "rt":
        family = "RTCF" if ndim == 2 else "NCF"
        return FunctionSpace(mesh, family, degree)
    else:
        if ndim == 1:
            family = "DG" if element == "dg" else "CG"
        else:
            family = "DQ" if element == "dg" else "Q"
        return VectorFunctionSpace(mesh, family, degree)


@pytest.mark.skipcomplex
def test_direct_solver(fs):
    mesh = fs.mesh()
    x = SpatialCoordinate(mesh)
    u_exact = dot(x, x)
    ncomp = fs.ufl_element().value_size()
    if ncomp > 1:
        u_exact = as_vector([u_exact + Constant(k) for k in range(ncomp)])

    n = FacetNormal(mesh)
    f_exact = grad(u_exact)
    B = u_exact - div(f_exact)
    T = dot(f_exact, n)
    uh = Function(fs)
    u = TrialFunction(fs)
    v = TestFunction(fs)

    subs = (1, 3)
    if mesh.cell_set._extruded:
        subs += ("top",)

    bcs = [DirichletBC(fs, u_exact, sub) for sub in subs]

    # sub_Dir = vertical subdomains for Dirichlet BCs
    # sub_Neu = vertical subdomains for Neumann BCs
    sub_Dir = "everywhere" if "on_boundary" in subs else tuple(s for s in subs if type(s) == int)
    if sub_Dir == "everywhere":
        sub_Neu = ()
    else:
        sub_Neu = tuple(set(mesh.exterior_facets.unique_markers) - set(s for s in subs if type(s) == int))

    if mesh.cell_set._extruded:
        dS_int = dS_v + dS_h
        ds_Dir = ds_v(sub_Dir)
        ds_Neu = ds_v(sub_Neu)
        if "bottom" in subs:
            ds_Dir += ds_b
        else:
            ds_Neu += ds_b
        if "top" in subs:
            ds_Dir += ds_t
        else:
            ds_Neu += ds_t
    else:
        dS_int = dS
        ds_Dir = ds(sub_Dir)
        ds_Neu = ds(sub_Neu)

    N = fs.ufl_element().degree()
    try:
        N, = set(N)
    except TypeError:
        pass

    eta = Constant((N+1)**2)
    h = CellVolume(mesh)/FacetArea(mesh)
    penalty = eta/h

    a = (inner(v, u)*dx
         + inner(grad(v), grad(u))*dx
         + inner(outer_jump(v, n), avg(penalty) * outer_jump(u, n)) * dS_int
         - inner(avg(grad(v)), outer_jump(u, n)) * dS_int
         - inner(avg(grad(u)), outer_jump(v, n)) * dS_int
         + inner(v, penalty * u) * ds_Dir
         - inner(grad(v), outer(u, n)) * ds_Dir
         - inner(grad(u), outer(v, n)) * ds_Dir)

    L = (inner(v, B)*dx
         + inner(v, T)*ds_Neu
         + inner(v, penalty * u_exact) * ds_Dir
         - inner(grad(v), outer(u_exact, n)) * ds_Dir)

    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters={
        "mat_type": "matfree",
        "ksp_type": "cg",
        "ksp_atol": 0.0E0,
        "ksp_rtol": 1.0E-8,
        "ksp_max_it": 20,
        "ksp_monitor": None,
        "ksp_norm_type": "unpreconditioned",
        "pc_type": "python",
        "pc_python_type": "firedrake.FDMPC",
        "fdm_pc_type": "cholesky",
        "fdm_pc_factor_mat_solver_type": "mumps",
        "fdm_pc_factor_mat_ordering_type": "nd",
    }, appctx={"eta": eta, })
    solver.solve()

    assert solver.snes.ksp.getIterationNumber() == 1 and norm(u_exact-uh, "H1") < 1.0E-8
