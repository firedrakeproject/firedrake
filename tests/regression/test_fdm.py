import pytest
from firedrake import *
from pyop2.utils import as_tuple
from firedrake.petsc import DEFAULT_DIRECT_SOLVER

ksp = {
    "mat_type": "matfree",
    "ksp_type": "cg",
    "ksp_atol": 0.0E0,
    "ksp_rtol": 1.0E-8,
    "ksp_norm_type": "natural",
    "ksp_monitor": None,
}

coarse = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "cholesky",
}

# FDM without static condensation
fdmstar = {
    "pc_type": "python",
    "pc_python_type": "firedrake.P1PC",
    "pmg_mg_coarse_compiler_mode": "vanilla",  # Turn off sum-factorization at lowest-order
    "pmg_mg_coarse": coarse,
    "pmg_mg_levels": {
        "ksp_max_it": 1,
        "ksp_type": "chebyshev",
        "ksp_norm_type": "none",
        "esteig_ksp_type": "cg",
        "esteig_ksp_norm_type": "natural",
        "ksp_chebyshev_esteig": "0.5,0.5,0.0,1.0",
        "pc_type": "python",
        "pc_python_type": "firedrake.FDMPC",
        "fdm": {
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMExtrudedStarPC",
            "pc_star_mat_ordering_type": "nd",
            "pc_star_sub_sub_pc_type": "cholesky",
        }
    }
}

# FDM with static condensation
facetstar = {
    "pc_type": "python",
    "pc_python_type": "firedrake.FacetSplitPC",
    "facet_pc_type": "python",
    "facet_pc_python_type": "firedrake.FDMPC",
    "facet_fdm_static_condensation": True,
    "facet_fdm_pc_use_amat": False,
    "facet_fdm_pc_type": "fieldsplit",
    "facet_fdm_pc_fieldsplit_type": "symmetric_multiplicative",
    "facet_fdm_fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "bjacobi",
        "sub_pc_type": "icc",  # this is exact for the sparse approximation used in FDM
    },
    "facet_fdm_fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.P1PC",
        "pmg_mg_coarse_compiler_mode": "vanilla",  # Turn off sum-factorization at lowest-order
        "pmg_mg_coarse": coarse,
        "pmg_mg_levels": {
            "ksp_max_it": 1,
            "ksp_type": "chebyshev",
            "ksp_norm_type": "none",
            "esteig_ksp_type": "cg",
            "esteig_ksp_norm_type": "natural",
            "ksp_chebyshev_esteig": "0.5,0.5,0.0,1.0",
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMExtrudedStarPC",
            "pc_star_mat_ordering_type": "nd",
            "pc_star_sub_sub_pc_type": "cholesky",
        }
    }
}

fdmstar.update(ksp)
facetstar.update(ksp)


def build_riesz_map(V, d):
    beta = Constant(1E-4)
    subs = [(1, 3)]
    if V.mesh().cell_set._extruded:
        subs += ["top"]

    x = SpatialCoordinate(V.mesh())
    x -= Constant([0.5]*len(x))
    if V.value_shape == ():
        u_exact = exp(-10*dot(x, x))
        u_bc = u_exact
    else:
        A = Constant([[-1.]*len(x)]*len(x)) + diag(Constant([len(x)]*len(x)))
        u_exact = dot(A, x) * exp(-10*dot(x, x))
        u_bc = Function(V)
        u_bc.project(u_exact, solver_parameters={"mat_type": "matfree", "pc_type": "jacobi"})

    bcs = [DirichletBC(V, u_bc, sub) for sub in subs]

    uh = Function(V)
    test = TestFunction(V)
    trial = TrialFunction(V)
    a = lambda v, u: inner(v, beta*u)*dx + inner(d(v), d(u))*dx
    return LinearVariationalProblem(a(test, trial), a(test, u_exact), uh, bcs=bcs)


def solve_riesz_map(problem, solver_parameters):
    problem.u.assign(0)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()
    return solver.snes.ksp.getIterationNumber()


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


@pytest.fixture(params=[None, "fdm"], ids=["spectral", "fdm"])
def variant(request):
    return request.param


@pytest.mark.skipcomplex
def test_p_independence_hgrad(mesh, variant):
    family = "Lagrange"
    expected = [16, 12] if mesh.topological_dimension() == 3 else [9, 7]
    solvers = [fdmstar] if variant is None else [fdmstar, facetstar]
    for degree in range(3, 6):
        V = FunctionSpace(mesh, family, degree, variant=variant)
        problem = build_riesz_map(V, grad)
        for sp, expected_it in zip(solvers, expected):
            assert solve_riesz_map(problem, sp) <= expected_it


@pytest.mark.skipmumps
@pytest.mark.skipcomplex
def test_p_independence_hcurl(mesh):
    family = "NCE" if mesh.topological_dimension() == 3 else "RTCE"
    expected = [13, 10] if mesh.topological_dimension() == 3 else [6, 6]
    solvers = [fdmstar, facetstar]
    for degree in range(3, 6):
        V = FunctionSpace(mesh, family, degree, variant="fdm")
        problem = build_riesz_map(V, curl)
        for sp, expected_it in zip(solvers, expected):
            assert solve_riesz_map(problem, sp) <= expected_it


@pytest.mark.skipmumps
@pytest.mark.skipcomplex
def test_p_independence_hdiv(mesh):
    family = "NCF" if mesh.topological_dimension() == 3 else "RTCF"
    expected = [6, 6]
    solvers = [fdmstar, facetstar]
    for degree in range(3, 6):
        V = FunctionSpace(mesh, family, degree, variant="fdm")
        problem = build_riesz_map(V, div)
        for sp, expected_it in zip(solvers, expected):
            assert solve_riesz_map(problem, sp) <= expected_it


@pytest.mark.skipcomplex
def test_variable_coefficient(mesh):
    gdim = mesh.geometric_dimension()
    k = 4
    V = FunctionSpace(mesh, "Lagrange", k)
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    x -= Constant([0.5]*len(x))

    # variable coefficients
    alphas = [0.1+10*dot(x, x)]*gdim
    alphas[0] = 1+10*exp(-dot(x, x))
    alpha = diag(as_vector(alphas))
    beta = ((10*cos(3*pi*x[0]) + 20*sin(2*pi*x[1]))*cos(pi*x[gdim-1]))**2

    a = (inner(grad(v), dot(alpha, grad(u))) + inner(v, beta*u))*dx(degree=3*k+2)
    L = inner(v, Constant(1))*dx

    subs = ("on_boundary",)
    if mesh.cell_set._extruded:
        subs += ("top", "bottom")
    bcs = [DirichletBC(V, 0, sub) for sub in subs]

    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=fdmstar)
    solver.solve()
    expected = 23 if gdim == 3 else 14
    assert solver.snes.ksp.getIterationNumber() <= expected


@pytest.fixture(params=["cg", "dg", "rt"],
                ids=["cg", "dg", "rt"])
def fs(request, mesh):
    degree = 3
    tdim = mesh.topological_dimension()
    element = request.param
    variant = "fdm_ipdg"
    if element == "rt":
        family = "RTCF" if tdim == 2 else "NCF"
        return FunctionSpace(mesh, family, degree, variant=variant)
    else:
        if tdim == 1:
            family = "DG" if element == "dg" else "CG"
        else:
            family = "DQ" if element == "dg" else "Q"
        return VectorFunctionSpace(mesh, family, degree, dim=5-tdim, variant=variant)


@pytest.mark.skipcomplex
def test_ipdg_direct_solver(fs):
    mesh = fs.mesh()
    x = SpatialCoordinate(mesh)
    gdim = mesh.geometric_dimension()
    ncomp = fs.value_size

    homogenize = gdim > 2
    if homogenize:
        rg = RandomGenerator(PCG64(seed=123456789))
        uh = rg.uniform(fs, -1, 1)
        u_exact = zero(uh.ufl_shape)
        u_bc = 0
    else:
        uh = Function(fs)
        u_exact = dot(x, x)
        if ncomp:
            u_exact = as_vector([u_exact + Constant(k) for k in range(ncomp)])
        u_bc = u_exact

    degree = max(as_tuple(fs.ufl_element().degree()))
    quad_degree = 2*(degree+1)-1

    # problem coefficients
    A1 = diag(Constant(range(1, gdim+1)))
    A2 = diag(Constant(range(1, ncomp+1)))
    alpha = lambda grad_u: dot(dot(A2, grad_u), A1)
    beta = diag(Constant(range(2, ncomp+2)))

    extruded = mesh.cell_set._extruded
    subs = (1,)
    if gdim > 1:
        subs += (3,)
    if extruded:
        subs += ("top",)
    bcs = [DirichletBC(fs, u_bc, sub) for sub in subs]

    dirichlet_ids = subs
    if "on_boundary" in dirichlet_ids:
        neumann_ids = []
    else:
        make_tuple = lambda s: s if type(s) == tuple else (s,)
        neumann_ids = list(set(mesh.exterior_facets.unique_markers) - set(sum([make_tuple(s) for s in subs if type(s) != str], ())))
    if extruded:
        if "top" not in dirichlet_ids:
            neumann_ids.append("top")
        if "bottom" not in dirichlet_ids:
            neumann_ids.append("bottom")

    dxq = dx(degree=quad_degree, domain=mesh)
    if extruded:
        dS_int = dS_v(degree=quad_degree) + dS_h(degree=quad_degree)
        ds_ext = {"on_boundary": ds_v(degree=quad_degree), "bottom": ds_b(degree=quad_degree), "top": ds_t(degree=quad_degree)}
        ds_Dir = [ds_ext.get(s) or ds_v(s, degree=quad_degree) for s in dirichlet_ids]
        ds_Neu = [ds_ext.get(s) or ds_v(s, degree=quad_degree) for s in neumann_ids]
    else:
        dS_int = dS(degree=quad_degree)
        ds_ext = {"on_boundary": ds(degree=quad_degree)}
        ds_Dir = [ds_ext.get(s) or ds(s, degree=quad_degree) for s in dirichlet_ids]
        ds_Neu = [ds_ext.get(s) or ds(s, degree=quad_degree) for s in neumann_ids]

    ds_Dir = sum(ds_Dir, ds(tuple()))
    ds_Neu = sum(ds_Neu, ds(tuple()))
    n = FacetNormal(mesh)
    h = CellVolume(mesh) / FacetArea(mesh)
    eta = Constant((degree+1)**2)
    penalty = eta / h

    num_flux = lambda u: avg(penalty) * avg(outer(u, n))
    num_flux_b = lambda u: (penalty/2) * outer(u, n)
    a_int = lambda v, u: inner(2 * avg(outer(v, n)), alpha(num_flux(u) - avg(grad(u))))
    a_Dir = lambda v, u: inner(outer(v, n), alpha(num_flux_b(u) - grad(u)))

    v = TestFunction(fs)
    u = TrialFunction(fs)
    a = ((inner(v, dot(beta, u)) + inner(grad(v), alpha(grad(u)))) * dxq
         + (a_int(v, u) + a_int(u, v)) * dS_int
         + (a_Dir(v, u) + a_Dir(u, v)) * ds_Dir)

    if homogenize:
        L = 0
    else:
        f_exact = alpha(grad(u_exact))
        B = dot(beta, u_exact) - div(f_exact)
        T = dot(f_exact, n)
        L = (inner(v, B)*dxq + inner(v, T)*ds_Neu
             + inner(outer(u_exact, n), alpha(2*num_flux_b(v) - grad(v))) * ds_Dir)

    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters={
        "mat_type": "matfree",
        "ksp_type": "cg",
        "ksp_atol": 0.0E0,
        "ksp_rtol": 1.0E-8,
        "ksp_max_it": 3,
        "ksp_monitor": None,
        "ksp_norm_type": "unpreconditioned",
        "pc_type": "python",
        "pc_python_type": "firedrake.PoissonFDMPC",
        "fdm_pc_type": "cholesky",
        "fdm_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER,
        "fdm_pc_factor_mat_ordering_type": "nd",
    }, appctx={"eta": eta, })
    solver.solve()

    assert solver.snes.ksp.getIterationNumber() == 1
    if homogenize:
        with uh.dat.vec_ro as uvec:
            assert uvec.norm() < 1E-8
    else:
        assert norm(u_exact-uh, "H1") < 1.0E-8
