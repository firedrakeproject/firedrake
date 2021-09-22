import pytest
from firedrake import *


@pytest.fixture(params=[2, 3],
                ids=["Rectangle", "Box"])
def mesh(request):
    nx = 4
    distribution = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    m = UnitSquareMesh(nx, nx, quadrilateral=True, distribution_parameters=distribution)
    if request.param == 3:
        m = ExtrudedMesh(m, nx)

    x = SpatialCoordinate(m)
    xnew = as_vector([acos(1-2*xj)/pi for xj in x])
    m.coordinates.interpolate(xnew)
    return m


@pytest.fixture
def expected(mesh):
    if mesh.topological_dimension() == 2:
        return [5, 5, 5]
    elif mesh.topological_dimension() == 3:
        return [8, 8, 8]


@pytest.mark.skipcomplex
def test_p_independence(mesh, expected):
    nits = []
    for p in range(3, 6):
        V = FunctionSpace(mesh, "Lagrange", p)
        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(grad(v), grad(u))*dx
        L = inner(v, Constant(1))*dx

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
            "ksp_norm_type": "unpreconditioned",
            "ksp_monitor_true_residual": None,
            "ksp_converged_reason": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.P1PC",
            "pmg_mg_levels": {
                "ksp_type": "chebyshev",
                "esteig_ksp_type": "cg",
                "esteig_ksp_norm_type": "unpreconditioned",
                "ksp_chebyshev_esteig": "0.8,0.2,0.0,1.0",
                "ksp_chebyshev_esteig_noisy": True,
                "ksp_chebyshev_esteig_steps": 8,
                "ksp_norm_type": "unpreconditioned",
                "pc_type": "python",
                "pc_python_type": "firedrake.FDMPC",
                "fdm": {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": asm,
                    "pc_star_backend": "petscasm",
                    "pc_star_sub_sub_pc_type": "cholesky",
                    "pc_star_sub_sub_pc_mat_factor_type": "cholmod",
                }
            },
            "pmg_mg_coarse": {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled_pc_type": "cholesky",
            }})
        solver.solve()
        nits.append(solver.snes.ksp.getIterationNumber())
    assert (nits == expected)


@pytest.mark.skipcomplex
def test_variable_coefficient(mesh):
    ndim = mesh.geometric_dimension()
    k = 4
    V = FunctionSpace(mesh, "Lagrange", k)
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    x -= Constant([0.5]*ndim)

    # variable coefficients
    alphas = [0.1+10*dot(x, x)]*ndim
    alphas[0] = 1+10*exp(-dot(x, x))
    alpha = diag(as_vector(alphas))
    beta = ((10*cos(3*pi*x[0]) + 20*sin(2*pi*x[1]))*cos(pi*x[ndim-1]))**2

    a = (inner(grad(v), dot(alpha, grad(u))) + inner(v, beta*u))*dx(degree=3*k+2)
    L = inner(v, Constant(1))*dx

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
        "ksp_norm_type": "unpreconditioned",
        "ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.P1PC",
        "pmg_mg_levels": {
            "ksp_type": "chebyshev",
            "esteig_ksp_type": "cg",
            "esteig_ksp_norm_type": "unpreconditioned",
            "ksp_chebyshev_esteig": "0.8,0.2,0.0,1.0",
            "ksp_chebyshev_esteig_noisy": True,
            "ksp_chebyshev_esteig_steps": 8,
            "ksp_norm_type": "unpreconditioned",
            "pc_type": "python",
            "pc_python_type": "firedrake.FDMPC",
            "fdm": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": asm,
                "pc_star_backend": "petscasm",
                "pc_star_sub_sub_pc_type": "cholesky",
                "pc_star_sub_sub_pc_mat_factor_type": "cholmod",
            }
        },
        "pmg_mg_coarse": {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "cholesky",
        }})
    solver.solve()
    assert solver.snes.ksp.getIterationNumber() <= 14


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
        return VectorFunctionSpace(mesh, family, degree, dim=5-ndim)


@pytest.mark.skipcomplex
def test_direct_solver(fs):
    mesh = fs.mesh()
    x = SpatialCoordinate(mesh)
    u_exact = dot(x, x)
    ndim = mesh.geometric_dimension()
    ncomp = fs.ufl_element().value_size()
    if ncomp > 1:
        u_exact = as_vector([u_exact + Constant(k) for k in range(ncomp)])

    N = fs.ufl_element().degree()
    try:
        N, = set(N)
    except TypeError:
        pass

    Nq = 2*(N+1)-1
    uh = Function(fs)
    u = TrialFunction(fs)
    v = TestFunction(fs)

    # problem coefficients
    A1 = diag(Constant(range(1, ndim+1)))
    A2 = diag(Constant(range(1, ncomp+1)))
    alpha = lambda grad_u: dot(dot(A2, grad_u), A1)
    beta = diag(Constant(range(2, ncomp+2)))

    n = FacetNormal(mesh)
    f_exact = alpha(grad(u_exact))
    B = dot(beta, u_exact) - div(f_exact)
    T = dot(f_exact, n)

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

    dxq = dx(degree=Nq)
    if mesh.cell_set._extruded:
        dS_int = dS_v(degree=Nq) + dS_h(degree=Nq)
        ds_Dir = ds_v(sub_Dir, degree=Nq)
        ds_Neu = ds_v(sub_Neu, degree=Nq)
        if "bottom" in subs:
            ds_Dir += ds_b(degree=Nq)
        else:
            ds_Neu += ds_b(degree=Nq)
        if "top" in subs:
            ds_Dir += ds_t(degree=Nq)
        else:
            ds_Neu += ds_t(degree=Nq)
    else:
        dS_int = dS(degree=Nq)
        ds_Dir = ds(sub_Dir, degree=Nq)
        ds_Neu = ds(sub_Neu, degree=Nq)

    eta = Constant((N+1)**2)
    h = CellVolume(mesh)/FacetArea(mesh)
    penalty = eta/h

    outer_jump = lambda w, n: outer(w('+'), n('+')) + outer(w('-'), n('-'))
    num_flux = lambda w: alpha(avg(penalty/2) * outer_jump(w, n))
    num_flux_b = lambda w: alpha((penalty/2) * outer(w, n))

    a = (inner(v, dot(beta, u)) * dxq
         + inner(grad(v), alpha(grad(u))) * dxq
         + inner(outer_jump(v, n), num_flux(u)-avg(alpha(grad(u)))) * dS_int
         + inner(outer_jump(u, n), num_flux(v)-avg(alpha(grad(v)))) * dS_int
         + inner(outer(v, n), num_flux_b(u)-alpha(grad(u))) * ds_Dir
         + inner(outer(u, n), num_flux_b(v)-alpha(grad(v))) * ds_Dir)

    L = (inner(v, B)*dxq
         + inner(v, T)*ds_Neu
         + inner(outer(u_exact, n), 2*num_flux_b(v)-alpha(grad(v))) * ds_Dir)

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
        "fdm_pc_factor_mat_solver_type": "cholmod",
        "fdm_pc_factor_mat_ordering_type": "nd",
    }, appctx={"eta": eta, })
    solver.solve()

    assert solver.snes.ksp.getIterationNumber() == 1 and norm(u_exact-uh, "H1") < 1.0E-8
