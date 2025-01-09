from firedrake import *
from firedrake.petsc import DEFAULT_AMG_PC
import pytest


@pytest.mark.parametrize("parameters",
                         [{
                             # Newton
                             "snes_type": "newtonls",
                             "snes_linesearch_type": "bt",
                             "ksp_type": "cg",
                             "ksp_rtol": 1e-6,
                             # Fieldsplit PC
                             "pc_type": "fieldsplit",
                             "pc_fieldsplit_type": "additive",
                             # V-cycle for the p-laplacian (using GMG)
                             "fieldsplit_0_ksp_type": "preonly",
                             "fieldsplit_0_pc_type": "mg",
                             "fieldsplit_0_mg_levels_ksp_type": "chebyshev",
                             "fieldsplit_0_mg_levels_ksp_max_it": 4,
                             "fieldsplit_0_mg_levels_pc_type": "icc",
                             # V-cycle of hypre for Poisson
                             "fieldsplit_1_ksp_type": "preonly",
                             "fieldsplit_1_pc_type": DEFAULT_AMG_PC,
                             # ILU(0) for DG mass matrix
                             "fieldsplit_2_ksp_type": "preonly",
                             "fieldsplit_2_pc_type": "bjacobi",
                             "fieldsplit_2_sub_pc_type": "ilu"},
                          {
                              # Newton
                              "snes_type": "newtonls",
                              "snes_linesearch_type": "bt",
                              "ksp_type": "cg",
                              "ksp_rtol": 1e-6,
                              # Same as before, just with a recursive split, so we need an aij matrix
                              "mat_type": "aij",
                              "pc_type": "fieldsplit",
                              "pc_fieldsplit_type": "additive",
                              "pc_fieldsplit_0_fields": "0,1",
                              "pc_fieldsplit_1_fields": "2",
                              # fieldsplit the first field
                              "fieldsplit_0_pc_type": "fieldsplit",
                              "fieldsplit_0_pc_fieldsplit_type": "additive",
                              # V-cycle for the p-laplacian (using GMG)
                              "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
                              "fieldsplit_0_fieldsplit_0_pc_type": "mg",
                              "fieldsplit_0_fieldsplit_0_mg_levels_ksp_type": "chebyshev",
                              "fieldsplit_0_fieldsplit_0_mg_levels_ksp_max_it": 4,
                              "fieldsplit_0_fieldsplit_0_mg_levels_pc_type": "icc",
                              # V-cycle of hypre for Poisson
                              "fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
                              "fieldsplit_0_fieldsplit_1_pc_type": DEFAULT_AMG_PC,
                              # ILU(0) for DG mass matrix
                              "fieldsplit_1_ksp_type": "preonly",
                              "fieldsplit_1_pc_type": "bjacobi",
                              "fieldsplit_1_sub_pc_type": "ilu"}])
@pytest.mark.skipcomplex
@pytest.mark.skipcomplexnoslate
def test_nested_split_multigrid(parameters):
    mesh = UnitSquareMesh(10, 10)

    nlevel = 2

    mh = MeshHierarchy(mesh, nlevel)

    # we will solve p-laplace in V, Poisson in Q, Mass in R
    V = FunctionSpace(mh[-1], "CG", 1)
    Q = FunctionSpace(mh[-1], "CG", 2)
    R = FunctionSpace(mh[-1], "DG", 1)

    W = V*Q*R
    w = Function(W)
    u, p, s = split(w)
    v, q, r = TestFunctions(W)

    epsilon = Constant(1e-4)
    nu = (epsilon**2 + 0.5 * inner(grad(u), grad(u)))

    x = SpatialCoordinate(mh[-1])[0]
    y = SpatialCoordinate(mh[-1])[1]

    plaplace_forcing = 16*(x**2 + y**2)

    poisson_forcing = 0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)

    F = inner(nu*grad(u), grad(v))*dx(degree=4) + inner(plaplace_forcing, v)*dx
    F += inner(grad(p), grad(q))*dx + inner(poisson_forcing, q)*dx
    F += inner(s, r)*dx - inner(x, r)*dx

    expect = Function(W)
    u_expect, p_expect, s_expect = expect.subfunctions

    u_expect.interpolate(x**2 + y**2)
    p_expect.interpolate(sin(pi*x)*tan(pi*x*0.25)*sin(pi*y))
    s_expect.interpolate(x)

    bcs = [DirichletBC(W.sub(0), u_expect, (1, 2, 3, 4)),
           DirichletBC(W.sub(1), Constant(0), (1, 2, 3, 4))]

    problem = NonlinearVariationalProblem(F, w, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, options_prefix="",
                                        solver_parameters=parameters)
    solver.solve()
    u, p, s = w.subfunctions

    assert norm(assemble(u_expect - u)) < 5e-5
    assert norm(assemble(p_expect - p)) < 1e-6
    assert norm(assemble(s_expect - s)) < 1e-10
