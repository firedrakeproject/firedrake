from firedrake import *
import pytest


@pytest.mark.parametrize('quadrilateral', [False, True])
@pytest.mark.parametrize('degree', [1, 2, 3])
def test_multiple_poisson_Pn(quadrilateral, degree):
    m = UnitSquareMesh(4, 4, quadrilateral=quadrilateral)
    mesh = ExtrudedMesh(m, 4)

    V = FunctionSpace(mesh, 'CG', degree)

    W = V*V

    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)

    # Solve 2 independent Poisson problems with strong boundary
    # conditions applied to the top and bottom for the first and on x
    # == 0 and x == 1 for the second.
    a = inner(grad(u), grad(v))*dx + inner(grad(p), grad(q))*dx

    # BCs for first problem
    bc0 = [DirichletBC(W[0], 10.0, "top"),
           DirichletBC(W[0], 1.0, "bottom")]
    # BCs for second problem
    bc1 = [DirichletBC(W[1], 8.0, 1),
           DirichletBC(W[1], 6.0, 2)]

    bcs = bc0 + bc1
    solve(a == 0, w, bcs=bcs,
          # Operator is block diagonal, so we can just do block jacobi
          # with lu on each block
          solver_parameters={'ksp_type': 'cg',
                             'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'additive',
                             'fieldsplit_ksp_type': 'preonly',
                             'fieldsplit_0_pc_type': 'lu',
                             'fieldsplit_1_pc_type': 'lu'})

    wexact = Function(W)

    u, p = wexact.subfunctions

    xs = SpatialCoordinate(mesh)
    u.interpolate(1 + 9*xs[2])
    p.interpolate(8 - 2*xs[0])

    assert assemble(inner(w - wexact, w - wexact)*dx) < 1e-8


@pytest.mark.parametrize('quadrilateral', [False, True])
@pytest.mark.parametrize('degree', [1, 2, 3])
def test_multiple_poisson_strong_weak_Pn(quadrilateral, degree):
    m = UnitSquareMesh(4, 4, quadrilateral=quadrilateral)
    mesh = ExtrudedMesh(m, 4)

    V = FunctionSpace(mesh, 'CG', degree)

    W = V*V

    w = Function(W)
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Solve two independent Poisson problems with a strong boundary
    # condition on the top and a weak condition on the bottom, and
    # vice versa.
    a = inner(grad(u), grad(v))*dx + inner(grad(p), grad(q))*dx
    L = inner(Constant(1), v)*ds_b + inner(Constant(4), q)*ds_t

    # BCs for first problem
    bc0 = [DirichletBC(W[0], 10.0, "top")]
    # BCs for second problem
    bc1 = [DirichletBC(W[1], 2.0, "bottom")]

    bcs = bc0 + bc1
    solve(a == L, w, bcs=bcs,
          # Operator is block diagonal, so we can just do block jacobi
          # with lu on each block
          solver_parameters={'ksp_type': 'cg',
                             'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'additive',
                             'fieldsplit_ksp_type': 'preonly',
                             'fieldsplit_0_pc_type': 'lu',
                             'fieldsplit_1_pc_type': 'lu'})

    wexact = Function(W)

    u, p = wexact.subfunctions

    xs = SpatialCoordinate(mesh)
    u.interpolate(11 - xs[2])
    p.interpolate(2 + 4*xs[2])

    assert assemble(inner(w - wexact, w - wexact)*dx) < 1e-8


@pytest.mark.parametrize("mat_type", ["nest", "aij"])
def test_stokes_taylor_hood(mat_type):
    length = 10
    m = IntervalMesh(40, length)
    mesh = ExtrudedMesh(m, 20)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    P = FunctionSpace(mesh, 'CG', 1)

    W = V*P

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a = inner(grad(u), grad(v))*dx - inner(p, div(v))*dx + inner(div(u), q)*dx

    f = Constant((0, 0))
    L = inner(f, v)*dx

    # No-slip velocity boundary condition on top and bottom,
    # y == 0 and y == 1
    noslip = Constant((0, 0))
    bc0 = [DirichletBC(W[0], noslip, "top"),
           DirichletBC(W[0], noslip, "bottom")]

    # Parabolic inflow y(1-y) at x = 0 in positive x direction
    xs = SpatialCoordinate(mesh)
    inflow = as_vector([xs[1]*(1 - xs[1]), 0.0])
    bc1 = DirichletBC(W[0], inflow, 1)

    # Zero pressure at outlow at x = 1
    bc2 = DirichletBC(W[1], 0.0, 2)

    bcs = bc0 + [bc1, bc2]

    w = Function(W)

    u, p = w.subfunctions
    solve(a == L, w, bcs=bcs,
          solver_parameters={'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'schur',
                             'fieldsplit_schur_fact_type': 'diag',
                             'fieldsplit_0_ksp_rtol': 1e-8,
                             'fieldsplit_0_pc_type': 'bjacobi',
                             'fieldsplit_0_sub_pc_type': 'lu',
                             'fieldsplit_1_pc_type': 'none',
                             'mat_type': mat_type})

    # We've set up Poiseuille flow, so we expect a parabolic velocity
    # field and a linearly decreasing pressure.
    uexact = Function(V).interpolate(as_vector([xs[1]*(1 - xs[1]), Constant(0.0)]))
    pexact = Function(P).interpolate(2*(length - xs[0]))

    assert errornorm(u, uexact, degree_rise=0) < 1e-7
    assert errornorm(p, pexact, degree_rise=0) < 1e-7


@pytest.mark.parallel
def test_stokes_taylor_hood_parallel():
    test_stokes_taylor_hood("nest")


@pytest.mark.parallel
def test_stokes_taylor_hood_parallel_monolithic():
    test_stokes_taylor_hood("aij")
