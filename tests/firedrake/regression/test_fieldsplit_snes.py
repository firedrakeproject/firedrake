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


def M(u, v):
    return inner(u, v)*dx


def A(u, v, nu):
    return (
        inner(dot(u, nabla_grad(u)), v)*dx
        + nu*inner(grad(u), grad(v))*dx
    )

class AuxiliaryBurgersSNES(AuxiliaryOperatorSNES):
    def form(self, snes, u, v):
        appctx = self.get_appctx(snes)
        nu = appctx["nu"]
        dt = appctx["dt"]
        un = appctx["un"]
        un1 = appctx["un1"]
        uh = (u + un)/2
        F = M(u - un, v) + dt*A(uh, v, nu)
        self.un = un
        self.un1 = un1
        return F, None, u


def test_auxiliary_snes():
    re = Constant(100)
    re_aux = Constant(50)

    nu = Constant(1/re)
    nu_aux = Constant(1/re_aux)

    nx = 50
    dt = Constant(0.1)  # CFL = dt*nx

    mesh = PeriodicUnitIntervalMesh(nx)
    x, = SpatialCoordinate(mesh)

    V = VectorFunctionSpace(mesh, "CG", 2)

    # current and next timestep
    ic = as_vector([1.0 + 0.5*sin(2*pi*x)])
    un = Function(V).project(ic)
    un1 = Function(V).project(ic)

    v = TestFunction(V)

    # Implicit midpoint rule
    uh = (un + un1)/2
    F = M(un1 - un, v) + dt*A(uh, v, nu)

    solver_parameters = {
        'snes': {
            'view': ':snes_view.log',
            'converged_reason': None,
            'monitor': None,
            'rtol': 1e-8,
            'atol': 0,
            'max_it': 3,
            'convergence_test': 'skip',
            'linesearch_type': 'l2',
            'linesearch_damping': 1.0,
            'linesearch_monitor': None,
        },
        'snes_type': 'nrichardson',
        'npc_snes_type': 'python',
        'npc_snes_python_type': f'{__name__}.AuxiliaryBurgersSNES',
        'npc_aux': {
            'snes': {
                'converged_reason': None,
                'monitor': None,
                'rtol': 1e-4,
                'atol': 0,
                'max_it': 2,
                'convergence_test': 'skip',
            },
            'snes_type': 'newtonls',
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'petsc',
        },
    }

    appctx = {
        "nu": nu_aux,
        "dt": dt,
        "un": un,
        "un1": un1,
    }

    solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, un1),
        solver_parameters=solver_parameters,
        options_prefix="fd", appctx=appctx)

    nsteps = 1
    for i in range(nsteps):
        solver.solve()
        un.assign(un1)


if __name__ == "__main__":
    test_auxiliary_snes()
