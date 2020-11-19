from firedrake import *


def test_coarse_nullspace():
    base = UnitSquareMesh(10, 10)
    mh = MeshHierarchy(base, 1)
    V = FunctionSpace(mh[-1], "CG", 1)
    w = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(w), grad(v))*dx
    f = Constant(0)
    F = inner(f, v)*dx
    one = Function(V)
    one.interpolate(Constant(1))
    nsp = VectorSpaceBasis([one])
    nsp.orthonormalize()

    sp = {"ksp_type": "cg",
          "ksp_monitor_true_residual": None,
          "pc_type": "mg",
          "mg_coarse_ksp_type": "richardson",
          "mg_coarse_pc_type": "gamg"}

    u = Function(V)
    with u.dat.vec_wo as x:
        x.setRandom()

    problem = LinearVariationalProblem(a, F, u)
    solver = LinearVariationalSolver(problem, solver_parameters=sp, nullspace=nsp)
    solver.solve()

    coarseksp = solver.snes.ksp.pc.getMGCoarseSolve()
    (cA, cP) = coarseksp.getOperators()
    nsp = cA.getNullSpace()
    assert nsp is not None
    vecs = nsp.getVecs()
    assert len(vecs) == 1
    assert abs(vecs[0].dot(vecs[0]) - 1) < 1.0e-12
