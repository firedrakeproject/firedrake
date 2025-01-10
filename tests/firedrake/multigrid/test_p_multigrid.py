import pytest
from firedrake import *


@pytest.fixture(params=[2, 3],
                ids=["Rectangle", "Box"])
def tp_mesh(request):
    nx = 4
    distribution = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    m = UnitSquareMesh(nx, nx, quadrilateral=True, distribution_parameters=distribution)
    if request.param == 3:
        m = ExtrudedMesh(m, nx)

    x = SpatialCoordinate(m)
    xnew = as_vector([acos(1-2*xj)/pi for xj in x])
    m.coordinates.interpolate(xnew)
    return m


@pytest.fixture(params=[0, 1, 2],
                ids=["H1", "HCurl", "HDiv"])
def tp_family(tp_mesh, request):
    tdim = tp_mesh.topological_dimension()
    if tdim == 3:
        families = ["Q", "NCE", "NCF"]
    else:
        families = ["Q", "RTCE", "RTCF"]
    return families[request.param]


@pytest.fixture(params=[None, "integral", "fdm"],
                ids=["spectral", "integral", "fdm"])
def variant(request):
    return request.param


@pytest.fixture(params=[0, 1],
                ids=["CG-DG", "HDiv-DG"])
def mixed_family(tp_mesh, request):
    if request.param == 0:
        Vfamily = "Q"
    else:
        tdim = tp_mesh.topological_dimension()
        Vfamily = "NCF" if tdim == 3 else "RTCF"
    Qfamily = "DQ"
    return Vfamily, Qfamily


def test_reconstruct_degree(tp_mesh, mixed_family):
    """ Construct a complicated mixed element and ensure we may recover it by
        p-refining or p-coarsening an element of the same family with different
        degree.
    """
    elist = []
    Vfamily, Qfamily = mixed_family
    for degree in [7, 2, 31]:
        if Vfamily in ["NCF", "RTCF"]:
            V = FunctionSpace(tp_mesh, Vfamily, degree)
        else:
            V = VectorFunctionSpace(tp_mesh, Vfamily, degree)
        Q = FunctionSpace(tp_mesh, Qfamily, degree-2)
        Z = MixedFunctionSpace([V, Q])
        e = Z.ufl_element()

        elist.append(e)
        assert e == PMGPC.reconstruct_degree(elist[0], degree)


def test_prolong_de_rham(tp_mesh):
    """ Interpolate a linear vector function between [H1]^d, HCurl and HDiv spaces
        where it can be exactly represented
    """
    from firedrake.preconditioners.pmg import prolongation_matrix_matfree

    tdim = tp_mesh.topological_dimension()
    b = Constant(list(range(tdim)))
    mat = diag(Constant([tdim+1]*tdim)) + Constant([[-1]*tdim]*tdim)
    expr = dot(mat, SpatialCoordinate(tp_mesh)) + b

    cell = tp_mesh.ufl_cell()
    elems = [VectorElement(FiniteElement("Q", cell=cell, degree=2)),
             FiniteElement("NCE" if tdim == 3 else "RTCE", cell=cell, degree=2),
             FiniteElement("NCF" if tdim == 3 else "RTCF", cell=cell, degree=2)]
    fs = [FunctionSpace(tp_mesh, e) for e in elems]
    us = [Function(V) for V in fs]
    us[0].interpolate(expr)
    for u in us:
        for v in us:
            if u != v:
                P = prolongation_matrix_matfree(u, v).getPythonContext()
                P._prolong()
                assert norm(v-expr, "L2") < 1E-14


def test_prolong_low_order_to_restricted(tp_mesh, tp_family, variant):
    """ Interpolate a low-order function to interior and facet high-order spaces
        and ensure that the sum of the two high-order functions is equal to the
        low-order function
    """
    from firedrake.preconditioners.pmg import prolongation_matrix_matfree

    degree = 5
    cell = tp_mesh.ufl_cell()
    element = FiniteElement(tp_family, cell=cell, degree=degree, variant=variant)
    Vi = FunctionSpace(tp_mesh, RestrictedElement(element, restriction_domain="interior"))
    Vf = FunctionSpace(tp_mesh, RestrictedElement(element, restriction_domain="facet"))
    Vc = FunctionSpace(tp_mesh, tp_family, degree=1)

    ui = Function(Vi)
    uf = Function(Vf)
    uc = Function(Vc)
    uc.dat.data[0::2] = 0.0
    uc.dat.data[1::2] = 1.0

    for v in [ui, uf]:
        P = prolongation_matrix_matfree(uc, v).getPythonContext()
        P._prolong()

    assert norm(ui + uf - uc, "L2") < 1E-13


@pytest.fixture(params=["triangles", "quadrilaterals"], scope="module")
def mesh(request):
    if request.param == "triangles":
        base = UnitSquareMesh(2, 2)
        mh = MeshHierarchy(base, 1)
        mesh = mh[-1]
    elif request.param == "quadrilaterals":
        base = UnitSquareMesh(2, 2, quadrilateral=True)
        mh = MeshHierarchy(base, 1)
        mesh = mh[-1]
    return mesh


@pytest.fixture(params=["matfree", "aij"], scope="module")
def mat_type(request):
    return request.param


def test_p_multigrid_scalar(mesh, mat_type):
    V = FunctionSpace(mesh, "CG", 4)

    u = Function(V)
    v = TestFunction(V)
    f = Constant(1)
    bcs = DirichletBC(V, 0, "on_boundary")

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx

    relax = {"ksp_type": "chebyshev",
             "ksp_monitor_true_residual": None,
             "ksp_norm_type": "unpreconditioned",
             "ksp_max_it": 3,
             "pc_type": "jacobi"}

    sp = {"snes_monitor": None,
          "snes_type": "ksponly",
          "ksp_type": "fgmres",
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.PMGPC",
          "pmg_pc_mg_type": "multiplicative",
          "pmg_mg_levels": relax,
          "pmg_mg_levels_transfer_mat_type": mat_type,
          "pmg_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_pc_type": "mg",
          "pmg_mg_coarse_pc_mg_type": "multiplicative",
          "pmg_mg_coarse_mg_levels": relax,
          "pmg_mg_coarse_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_mg_coarse_pc_type": "gamg",
          "pmg_mg_coarse_mg_coarse_pc_gamg_threshold": 0}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()

    assert solver.snes.ksp.its <= 5
    ppc = solver.snes.ksp.pc.getPythonContext().ppc
    assert ppc.getMGLevels() == 3
    assert ppc.getMGCoarseSolve().pc.getMGLevels() == 2


def test_p_multigrid_nonlinear_scalar(mesh, mat_type):
    V = FunctionSpace(mesh, "CG", 4)

    u = Function(V)
    v = TestFunction(V)
    f = Constant(1)
    bcs = DirichletBC(V, 0, "on_boundary")

    F = inner((Constant(1.0) + u**2) * grad(u), grad(v))*dx - inner(f, v)*dx

    relax = {"ksp_type": "chebyshev",
             "ksp_monitor_true_residual": None,
             "ksp_norm_type": "unpreconditioned",
             "ksp_max_it": 3,
             "pc_type": "jacobi"}

    sp = {"snes_monitor": None,
          "snes_type": "newtonls",
          "ksp_type": "fgmres",
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.PMGPC",
          "pmg_pc_mg_type": "multiplicative",
          "pmg_mg_levels": relax,
          "pmg_mg_levels_transfer_mat_type": mat_type,
          "pmg_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_pc_type": "mg",
          "pmg_mg_coarse_pc_mg_type": "multiplicative",
          "pmg_mg_coarse_mg_levels": relax,
          "pmg_mg_coarse_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_mg_coarse_pc_type": "gamg",
          "pmg_mg_coarse_mg_coarse_pc_gamg_threshold": 0}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()

    assert solver.snes.its <= 3


@pytest.mark.skipcomplex
def test_p_multigrid_vector():
    mesh = UnitSquareMesh(2, 2)

    V = VectorFunctionSpace(mesh, "CG", 4)
    u = Function(V)

    rho = Constant(2700)
    g = Constant(-9.81)
    B = Constant((0.0, rho*g))  # Body force per unit volume

    # Elasticity parameters
    E_, nu = 6.9e10, 0.334
    mu, lmbda = Constant(E_/(2*(1 + nu))), Constant(E_*nu/((1 + nu)*(1 - 2*nu)))

    # Linear elastic energy
    E = 0.5 * (
               2*mu * inner(sym(grad(u)), sym(grad(u)))*dx     # noqa: E126
               + lmbda * inner(div(u), div(u))*dx             # noqa: E126
               - inner(B, u)*dx                               # noqa: E126
    )                                                         # noqa: E126

    bcs = DirichletBC(V, zero((2,)), 1)

    F = derivative(E, u, TestFunction(V))
    sp = {"snes_monitor": None,
          "snes_type": "ksponly",
          "ksp_type": "fgmres",
          "ksp_rtol": 1.0e-8,
          "ksp_atol": 1.0e-8,
          "ksp_converged_reason": None,
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.PMGPC",
          "pmg_pc_mg_type": "full",
          "pmg_mg_levels_ksp_type": "chebyshev",
          "pmg_mg_levels_ksp_monitor_true_residual": None,
          "pmg_mg_levels_ksp_norm_type": "unpreconditioned",
          "pmg_mg_levels_ksp_max_it": 2,
          "pmg_mg_levels_pc_type": "pbjacobi",
          "pmg_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_pc_type": "lu"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()

    assert solver.snes.ksp.its <= 20
    assert solver.snes.ksp.pc.getPythonContext().ppc.getMGLevels() == 3


@pytest.mark.skipcomplex
def test_p_multigrid_mixed(mat_type):
    mesh = UnitSquareMesh(1, 1, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 4)
    Z = MixedFunctionSpace([V, V])
    x = SpatialCoordinate(mesh) - Constant((0.5, 0.5))
    z_exact = as_vector([dot(x, x), dot(x, x)-Constant(1/6)])
    B = -div(grad(z_exact))
    T = dot(grad(z_exact), FacetNormal(mesh))
    z = Function(Z)
    E = 0.5 * inner(grad(z), grad(z))*dx - inner(B, z)*dx - inner(T, z)*ds
    F = derivative(E, z, TestFunction(Z))
    bcs = [DirichletBC(Z.sub(0), z_exact[0], "on_boundary")]

    relax = {"transfer_mat_type": mat_type,
             "ksp_type": "chebyshev",
             "ksp_monitor_true_residual": None,
             "ksp_norm_type": "unpreconditioned",
             "ksp_max_it": 3,
             "pc_type": "jacobi"}

    coarse = {"mat_type": "aij",  # This circumvents the need for AssembledPC
              "ksp_type": "richardson",
              "ksp_max_it": 1,
              "ksp_norm_type": "unpreconditioned",
              "ksp_monitor": None,
              "pc_type": "cholesky",
              "pc_factor_shift_type": "nonzero",
              "pc_factor_shift_amount": 1E-10}

    sp = {"snes_monitor": None,
          "snes_type": "ksponly",
          "ksp_type": "cg",
          "ksp_rtol": 1E-12,
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.PMGPC",
          "mat_type": mat_type,
          "pmg_pc_mg_type": "multiplicative",
          "pmg_mg_levels": relax,
          "pmg_mg_coarse": coarse}

    # Make the Function spanning the nullspace
    c_basis = assemble(TestFunction(Z.sub(1))*dx)
    f_basis = Function(c_basis.function_space().dual(), val=c_basis.dat)

    basis = VectorSpaceBasis([f_basis])
    basis.orthonormalize()
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), basis])
    problem = NonlinearVariationalProblem(F, z, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, nullspace=nullspace)
    solver.solve()
    assert solver.snes.ksp.its <= 7
    ppc = solver.snes.ksp.pc.getPythonContext().ppc
    assert ppc.getMGLevels() == 3

    # test that nullspace component is zero
    assert abs(assemble(z[1]*dx)) < 1E-12
    # test that we converge to the exact solution
    assert norm(z-z_exact, "H1") < 1E-12

    # test that we have coarsened the nullspace correctly
    ctx_levels = 0
    level = solver._ctx
    while level is not None:
        nsp = level._nullspace
        assert isinstance(nsp, MixedVectorSpaceBasis)
        assert nsp._bases[0].index == 0
        assert isinstance(nsp._bases[1], VectorSpaceBasis)
        assert len(nsp._bases[1]._petsc_vecs) == 1
        level = level._coarse
        ctx_levels += 1
    assert ctx_levels == 3

    # test that caches are parallel-safe
    dummy_eq = type(object).__eq__
    for cache in (PMGPC._coarsen_cache, PMGPC._transfer_cache):
        assert len(cache) > 0
        for k in cache:
            assert type(k).__eq__ is dummy_eq


def test_p_fas_scalar():
    mat_type = "matfree"
    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 4)

    # This problem is fabricated such that the exact solution
    # is resolved before reaching the finest level, hence no
    # work should be done in the finest level.
    # This will no longer be true for non-homogenous bcs, due
    # to the way firedrake imposes the bcs before injection.
    u = Function(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    f = x[0]*(1-x[0]) + x[1]*(1-x[1])
    bcs = DirichletBC(V, 0, "on_boundary")

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx

    # Due to the convoluted nature of the nested iteration
    # it is better to specify absolute tolerances only
    rhs = assemble(F, bcs=bcs)
    with rhs.dat.vec_ro as Fvec:
        Fnorm = Fvec.norm()

    rtol = 1E-8
    atol = rtol * Fnorm

    coarse = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "ksp_norm_type": None,
        "pc_type": "cholesky"}

    relax = {
        "ksp_type": "chebyshev",
        "ksp_monitor_true_residual": None,
        "ksp_norm_type": "unpreconditioned",
        "pc_type": "jacobi"}

    pmg = {
        "snes_type": "ksponly",
        "ksp_atol": atol,
        "ksp_rtol": 1E-50,
        "ksp_type": "cg",
        "ksp_converged_reason": None,
        "ksp_monitor_true_residual": None,
        "ksp_norm_type": "unpreconditioned",
        "pc_type": "python",
        "pc_python_type": "firedrake.PMGPC",
        "pmg_pc_mg_type": "multiplicative",
        "pmg_mg_levels": relax,
        "pmg_mg_levels_transfer_mat_type": mat_type,
        "pmg_mg_coarse": coarse}

    pfas = {
        "mat_type": mat_type,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": atol,
        "snes_rtol": 1E-50,
        "snes_type": "python",
        "snes_python_type": "firedrake.PMGSNES",
        "pfas_snes_fas_type": "kaskade",
        "pfas_fas_levels": pmg,
        "pfas_fas_coarse": coarse}

    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=pfas)
    solver.solve()

    ppc = solver.snes.getPythonContext().ppc
    levels = ppc.getFASLevels()
    assert levels == 3
    assert ppc.getFASSmoother(levels-1).getLinearSolveIterations() == 0


@pytest.mark.skipcomplex
def test_p_fas_nonlinear_scalar():
    mat_type = "matfree"
    degree = 4
    dxq = dx(degree=3*degree+2)  # here we also test coarsening of quadrature degree

    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", degree)
    u = Function(V)
    f = Constant(1)
    bcs = DirichletBC(V, 0, "on_boundary")

    # Regularized p-Laplacian
    p = 5
    eps = Constant(1)
    y = eps + inner(grad(u), grad(u))
    E = (1/p)*(y**(p/2))*dxq - inner(f, u)*dxq
    F = derivative(E, u, TestFunction(V))

    fcp = {"quadrature_degree": 3*degree+2}
    problem = NonlinearVariationalProblem(F, u, bcs, form_compiler_parameters=fcp)

    # Due to the convoluted nature of the nested iteration
    # it is better to specify absolute tolerances only
    rhs = assemble(F, bcs=bcs)
    with rhs.dat.vec_ro as Fvec:
        Fnorm = Fvec.norm()

    rtol = 1E-8
    atol = rtol * Fnorm
    rtol = 0.0
    newton = {
        "mat_type": "aij",
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_type": "newtonls",
        "snes_max_it": 20,
        "snes_atol": atol,
        "snes_rtol": rtol}

    coarse = {
        "ksp_type": "preonly",
        "ksp_norm_type": None,
        "pc_type": "cholesky"}

    relax = {
        "ksp_type": "chebyshev",
        "ksp_norm_type": "unpreconditioned",
        "ksp_chebyshev_esteig": "0.75,0.25,0,1",
        "ksp_max_it": 3,
        "pc_type": "jacobi"}

    pmg = {
        "ksp_atol": atol*1E-1,
        "ksp_rtol": rtol,
        "ksp_type": "cg",
        "ksp_converged_reason": None,
        "ksp_monitor_true_residual": None,
        "ksp_norm_type": "unpreconditioned",
        "pc_type": "python",
        "pc_python_type": "firedrake.PMGPC",
        "pmg_pc_mg_type": "multiplicative",
        "pmg_mg_levels": relax,
        "pmg_mg_levels_transfer_mat_type": mat_type,
        "pmg_mg_coarse": coarse}

    npmg = {**newton, **pmg}
    ncrs = {**newton, **coarse}

    pfas = {
        "mat_type": "aij",
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": atol,
        "snes_rtol": rtol,
        "snes_type": "python",
        "snes_python_type": "firedrake.PMGSNES",
        "pfas_snes_fas_type": "kaskade",
        "pfas_fas_levels": npmg,
        "pfas_fas_coarse": ncrs}

    def check_coarsen_quadrature(solver):
        # Go through p-MG hierarchy
        # Extract quadrature degree from forms and compiler parameters, Nq
        # Extract degree from solution, Nl
        # Assert that Nq == 3*Nl+2
        level = solver._ctx
        while level is not None:
            p = level._problem
            Nq = set()
            for form in filter(None, (p.F, p.J, p.Jp)):
                Nq.update(set(f.metadata().get("quadrature_degree", set()) for f in form.integrals()))
            if p.form_compiler_parameters is not None:
                Nfcp = p.form_compiler_parameters.get("quadrature_degree", None)
                Nq.update([Nfcp] if Nfcp else [])
            Nq, = Nq
            Nl = p.u.ufl_element().degree()
            try:
                Nl = max(Nl)
            except TypeError:
                pass
            assert Nq == 3*Nl+2
            level = level._coarse

    solver_pfas = NonlinearVariationalSolver(problem, solver_parameters=pfas)
    solver_pfas.solve()

    check_coarsen_quadrature(solver_pfas)
    ppc = solver_pfas.snes.getPythonContext().ppc
    levels = ppc.getFASLevels()
    assert levels == 3

    iter_pfas = ppc.getFASSmoother(levels-1).getLinearSolveIterations()

    u.assign(zero())
    solver_npmg = NonlinearVariationalSolver(problem, solver_parameters=npmg)
    solver_npmg.solve()

    check_coarsen_quadrature(solver_npmg)
    iter_npmg = solver_npmg.snes.getLinearSolveIterations()
    assert 2*iter_pfas <= iter_npmg


@pytest.fixture
def piola_mesh():
    return UnitDiskMesh(3)


@pytest.mark.parametrize("mat_type", ("matfree", "aij"))
@pytest.mark.parametrize("mixed", (False, True), ids=("standalone", "mixed"))
@pytest.mark.parametrize("family, degree", (("CG", 4), ("N2curl", 2), ("N1div", 3)))
def test_pmg_transfer_piola(piola_mesh, family, degree, mixed, mat_type):
    """Test prolongation and restriction kernels for piola-mapped elements.
    """
    from firedrake.preconditioners.pmg import prolongation_matrix_matfree, prolongation_matrix_aij
    Vf = FunctionSpace(piola_mesh, family, degree)
    if mixed:
        DG = FunctionSpace(Vf.mesh(), "DG", 2)
        Vf = Vf * Vf * DG
    Vc = Vf.reconstruct(degree=1)

    Vf_bcs = [DirichletBC(Vf.sub(0), 0, "on_boundary")]
    Vc_bcs = [DirichletBC(Vc.sub(0), 0, "on_boundary")]
    if mat_type == "matfree":
        P = prolongation_matrix_matfree(Vc, Vf, Vc_bcs, Vf_bcs)
    else:
        P = prolongation_matrix_aij(Vc, Vf, Vc_bcs, Vf_bcs)

    uc = Function(Vc)
    uf = Function(Vf)
    with uc.dat.vec_wo as xc:
        xc.setRandom()
    for bc in Vc_bcs:
        bc.zero(uc)
    with uc.dat.vec_ro as xc, uf.dat.vec as xf:
        P.mult(xc, xf)
    assert norm(uf - uc) < 1E-12

    rc = Cofunction(Vc.dual())
    rf = Cofunction(Vf.dual())
    with rf.dat.vec_wo as xf:
        xf.setRandom()
    for bc in Vf_bcs:
        bc.zero(rf)
    with rf.dat.vec_ro as xf, rc.dat.vec as xc:
        P.multTranspose(xf, xc)

    assert abs(assemble(action(rf, uf)) - assemble(action(rc, uc))) < 1E-11
