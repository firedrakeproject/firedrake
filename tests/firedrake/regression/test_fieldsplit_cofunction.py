import firedrake as fd


def test_fieldsplit_cofunction():
    """
    Test that fieldsplit preconditioners can be used
    with a cofunction on the right hand side.
    """
    mesh = fd.UnitSquareMesh(4, 4)
    BDM = fd.FunctionSpace(mesh, "BDM", 1)
    DG = fd.FunctionSpace(mesh, "DG", 0)
    W = BDM*DG

    u, p = fd.TrialFunctions(W)
    v, q = fd.TestFunctions(W)

    # simple wave equation scheme
    a = (fd.dot(u, v) + fd.div(v)*p
         - fd.div(u)*q + p*q)*fd.dx

    x, y = fd.SpatialCoordinate(mesh)

    f = fd.Function(W)

    f.subfunctions[0].project(
        fd.as_vector([0.01*y, 0]))
    f.subfunctions[1].interpolate(
        -10*fd.exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

    # compare to plain 1-form
    L_check = fd.inner(f, fd.TestFunction(W))*fd.dx
    L_cofun = f.riesz_representation()

    # brute force schur complement solver
    params = {
        'ksp_converged_reason': None,
        'ksp_type': 'preonly',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full',
        'pc_fieldsplit_schur_precondition': 'full',
        'fieldsplit': {
            'ksp_type': 'preonly',
            'pc_type': 'lu'
        }
    }

    w_check = fd.Function(W)
    problem_check = fd.LinearVariationalProblem(a, L_check, w_check)
    solver_check = fd.LinearVariationalSolver(problem_check,
                                              solver_parameters=params)
    solver_check.solve()

    w_cofun = fd.Function(W)
    problem_cofun = fd.LinearVariationalProblem(a, L_cofun, w_cofun)
    solver_cofun = fd.LinearVariationalSolver(problem_cofun,
                                              solver_parameters=params)
    solver_cofun.solve()

    assert fd.errornorm(w_check, w_cofun) < 1e-14
