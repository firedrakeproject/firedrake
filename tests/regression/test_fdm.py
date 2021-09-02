import pytest
from firedrake import *


@pytest.fixture(params=[2, 3],
                ids=["Rectangle", "Box"])
def mesh(request):
    distribution = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    if request.param == 2:
        return RectangleMesh(10, 20, 2, 3, quadrilateral=True, distribution_parameters=distribution)
    if request.param == 3:
        base = RectangleMesh(5, 3, 1, 2, quadrilateral=True, distribution_parameters=distribution)
        return ExtrudedMesh(base, 5, layer_height=3/5)


@pytest.fixture
def expected(mesh):
    if mesh.topological_dimension() == 2:
        return [5, 5, 5]
    elif mesh.topological_dimension() == 3:
        return [6, 6, 6]


@pytest.mark.skipcomplex
def test_p_independence(mesh, expected):
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
            asm = "firedrake.ASMHexStarPC"
            subs += ("top", "bottom")
        bcs = [DirichletBC(V, zero(V.ufl_element().value_shape()), sub) for sub in subs]

        uh = Function(V)
        problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

        solver = LinearVariationalSolver(problem, solver_parameters={
            "mat_type": "matfree",
            "ksp_type": "cg",
            "ksp_converged_reason": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.P1PC",
            "pmg_mg_levels_ksp_type": "chebyshev",
            "pmg_mg_levels_ksp_norm_type": "unpreconditioned",
            "pmg_mg_levels_ksp_monitor_true_residual": None,
            "pmg_mg_levels_pc_type": "python",
            "pmg_mg_levels_pc_python_type": "firedrake.FDMPC",
            "pmg_mg_levels_fdm": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": asm,
                "pc_star_backend": "petscasm",
                "pc_star_sub_sub_ksp_type": "preonly",
                "pc_star_sub_sub_pc_type": "cholesky",
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


def test_direct_solver(fs):
    mesh = fs.mesh()
    x = SpatialCoordinate(mesh)
    uex = dot(x, x)
    ncomp = fs.ufl_element().value_size()
    if ncomp > 1:
        uex = as_vector([uex + Constant(k) for k in range(ncomp)])

    rhs = -div(grad(uex))
    uh = Function(fs)
    u = TrialFunction(fs)
    v = TestFunction(fs)
    L = inner(v, rhs)*dx
    a = inner(grad(v), grad(u))*dx

    subs = (1, 3)
    if mesh.layers:
        subs += ("top",)

    bcs = [DirichletBC(fs, uex, sub) for sub in subs]

    sub_Dir = "everywhere" if "on_boundary" in subs else tuple(s for s in subs if type(s) == int)
    if mesh.layers:
        dS_int = dS_v + dS_h
        ds_Dir = ds_v(sub_Dir)
        if "bottom" in subs:
            ds_Dir += ds_b
        if "top" in subs:
            ds_Dir += ds_t
    else:
        dS_int = dS
        ds_Dir = ds(sub_Dir)

    N = fs.ufl_element().degree()
    try:
        N, = set(N)
    except TypeError:
        pass

    eta = Constant((N+1)**2)
    n = FacetNormal(mesh)
    h = CellVolume(mesh)/FacetArea(mesh)
    penalty = eta/h

    a += (inner(outer_jump(v, n), avg(penalty) * outer_jump(u, n)) * dS_int
          - inner(avg(grad(v)), outer_jump(u, n)) * dS_int
          - inner(avg(grad(u)), outer_jump(v, n)) * dS_int
          + inner(v, penalty * u) * ds_Dir
          - inner(grad(v), outer(u, n)) * ds_Dir
          - inner(grad(u), outer(v, n)) * ds_Dir)

    L += (inner(v, penalty * uex) * ds_Dir
          - inner(grad(v), outer(uex, n)) * ds_Dir)

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
    assert solver.snes.ksp.getIterationNumber() == 1
