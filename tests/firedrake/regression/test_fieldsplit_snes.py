from firedrake import *


def test_fieldsplit_snes():
    re = Constant(100)
    nu = Constant(1/re)

    nx = 50
    dt = Constant(0.1)  # CFL = dt*nx

    mesh = PeriodicUnitIntervalMesh(nx)
    x, = SpatialCoordinate(mesh)

    Vu = VectorFunctionSpace(mesh, "CG", 2)
    Vq = FunctionSpace(mesh, "DG", 1)
    W = Vu*Vq

    w0 = Function(W)
    u0, q0 = w0.subfunctions
    u0.project(as_vector([0.5 + 1.0*sin(2*pi*x)]))
    q0.interpolate(cos(2*pi*x))

    def M(u, v):
        return inner(u, v)*dx

    def Aburgers(u, v, nu):
        return (
            inner(dot(u, nabla_grad(u)), v)*dx
            + nu*inner(grad(u), grad(v))*dx
        )

    def Ascalar(q, p, u):
        n = FacetNormal(mesh)
        un = 0.5*(dot(u, n) + abs(dot(u, n)))
        return (- q*div(u*p)*dx
                + jump(un*q)*jump(p)*dS)

    # current and next timestep
    w = Function(W)
    wn = Function(W)

    u, q = split(w)
    un, qn = split(wn)

    v, p = TestFunctions(W)

    # Trapezium rule
    F = (
        M(un - u, v) + 0.5*dt*(Aburgers(un, v, nu) + Aburgers(u, v, nu))
        + M(qn - q, p) + 0.5*dt*(Ascalar(qn, p, un) + Ascalar(q, p, u))
    )

    common_params = {
        'snes_converged_reason': None,
        'snes_monitor': None,
        'snes_rtol': 1e-8,
        'snes_atol': 1e-12,
        'ksp_converged_reason': None,
        'ksp_monitor': None,
    }

    newton_params = {
        'snes_type': 'newtonls',
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }

    uparams = common_params | newton_params
    qparams = common_params | newton_params | {'snes_type': 'ksponly'}

    python_params = {
        'snes_type': 'nrichardson',
        'npc_snes_type': 'python',
        'npc_snes_python_type': 'firedrake.FieldsplitSNES',
        'npc_snes_fieldsplit_type': 'additive',
        'npc_fieldsplit_0': uparams,
        'npc_fieldsplit_1': qparams,
    }

    params = common_params | python_params

    w.assign(w0)
    wn.assign(w0)
    u, q = w.subfunctions
    un, qn = wn.subfunctions
    solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, wn),
        solver_parameters=params,
        options_prefix="")

    nsteps = 2
    for i in range(nsteps):
        w.assign(wn)
        solver.solve()
