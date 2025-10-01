from firedrake import *

__all__ = (
    "CorrelationOperatorBase",
    "ExplicitFormCorrelationBase",
    "ImplicitFormCorrelationBase",
    "ExplicitMassCorrelation",
    "ImplicitMassCorrelation",
    "ExplicitDiffusionCorrelation",
    "ImplicitDiffusionCorrelation",
    "CorrelationOperatorPC",
    "CorrelationOperatorMat",
)


def _make_rhs(b):
    if isinstance(b, Function):
        v = TestFunction(b.function_space())
        return inner(b, v)*dx
    elif isinstance(b, Cofunction):
        v = TestFunction(b.function_space().dual())
        return inner(b.riesz_representation(), v)*dx
    else:
        return b


class CorrelationOperatorBase:
    """Correlation weighted norm x^{T}B^{-1}x
    B: V* -> V
    B^{-1}: V -> V*
    """
    def __init__(self, V, generator=None, seed=None):
        self.V = V
        self.generator = generator or Generator(PCG64(seed=seed or 13))

    def norm(self, x):
        """Return x^{T}B^{-1}x

        Inheriting classes may provide more efficient specialisations.
        """
        return self.solve(x)(x)

    def apply(self, y, x=None):
        """Return x = By
        B: V* -> V
        """
        raise NotImplementedError

    def solve(self, x, y=None):
        """Return y = B^{-1}x
        B^{-1}: V -> V*
        """
        raise NotImplementedError

    def uncorrelated_noise(self, x=None):
        """
        Gaussian random variable with zero mean and identity variance.
        """
        x = x or Function(self.V)
        x.assign(self.generator.standard_normal(self.V))
        return x

    def correlated_noise(self, x=None, v=None):
        """
        Correlate white noise: v = B^{1/2}x

        v = l(M^{-1}G)^{m/2}M^{-1/2}x
        """
        raise NotImplementedError

    def new_variable(self, dual=False):
        return Function(self.V.dual() if dual else self.V)

    def new_primal_variable(self):
        return self.new_variable(dual=False)

    def new_dual_variable(self):
        return self.new_variable(dual=True)


class FormCorrelationOperatorBase(CorrelationOperatorBase):
    """Correlation operator is the action or inverse of a finite element form m times.
    x^{T}B^{-1}x = ||x||_{B^{-1}}
    B: V* -> V
    B^{-1}: V -> V*
    """
    def __init__(self, V, m=2, solver_parameters=None, bcs=None, generator=None, seed=None, Msqrt_inv=None):
        super().__init__(V, generator=generator, seed=seed)

        if not isinstance(m, int):
            raise TypeError(
                "m must be an integer number of form applications")
        if (m == 0) or ((m % 2) != 0):
            raise ValueError(
                "number of form applications must be even and >0")

        self.m = m
        self.bcs = bcs or []
        self.solver_parameters = solver_parameters or {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }

        _x = Function(V)
        self._xaction = _x

        u = TrialFunction(V)
        v = TestFunction(V)

        self.Maction = inner(_x, v)*dx
        self.Msolve = inner(u, v)*dx

        self.Gaction = self.form(_x, v)
        self.Gsolve = self.form(u, v)

        # diagonal approximation of inverse square root of mass matrix
        if Msqrt_inv is None:
            Msqrt_inv = Function(V)
            Msqrt_inv.dat.data[:] = np.sqrt(1/assemble(self.Msolve, diagonal=True).dat.data)
        self.Msqrt_inv = Msqrt_inv

    def _MinvG(self, x, w=None):
        """Return w = M^{-1}Gx
        M^{-1}G: V -> V
        """
        w = w or Function(self.V)
        self._xaction.assign(x)
        solve(self.Msolve == self.Gaction, w,
              solver_parameters=self.solver_parameters)
        return w

    def _GinvM(self, x, w=None):
        """Return w = G^{-1}Mx
        G^{-1}M: V -> V
        """
        w = w or Function(self.V)
        self._xaction.assign(x)
        solve(self.Gsolve == self.Maction, w,
              solver_parameters=self.solver_parameters)
        return w

    def riesz(self, x):
        return x.riesz_representation()

    def form(self, u, v):
        """Return the form defining the correlation operator.

        Inheriting classes must implement this.
        """
        raise NotImplementedError


class ExplicitFormCorrelationBase(FormCorrelationOperatorBase):
    """Correlation operator is the action of a finite element form m times.
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||((G^{-1}M)^(m/2)x)||
    M: V -> V* = <u,v>
    G: V -> V*
    B: V* -> V = (Minv*G)^(m/2)*Minv*(G*Minv)^(m/2)
    B^{-1}: V -> V* = (Ginv*M)^{T}M(Ginv*M) = M*Ginv*M*Ginv*M (if m=2)
    """

    def apply(self, y, x=None):
        """Return x = By
        B: V* -> V = l((M^{-1}G)^m)M^{-1}l
        """
        x = x or Function(self.V)
        primal = Function(self.V)
        primal = Function(self.V).assign(self.riesz(y))

        primal.interpolate(self.lamda*primal)
        for _ in range(self.m):
            primal = self._MinvG(primal)
        primal.interpolate(self.lamda*primal)

        return x.assign(primal)

    def solve(self, x, y=None):
        """Return y = B^{-1}x
        B^{-1}: V -> V* = linv(((MG^{-1})**m)M)linv
        """
        y = y or Function(self.V.dual())
        primal = Function(self.V).assign(x)

        primal.interpolate(self.lamda_inv*primal)
        for _ in range(self.m):
            primal = self._GinvM(primal)
        primal.interpolate(self.lamda_inv*primal)

        return y.assign(self.riesz(primal))

    def norm(self, x):
        """Return x^{T}B^{-1}x = ||x||^2_{B^{-1}}


        x^{T}B^{-1}x = x^{T}B^{-T/2}MB^{-1/2}x
                     = || B^{-1/2}x ||^2_{M}
                     = || ((G^{-1}M)^{m/2}linv)x ||^2_{M}
        """
        primal = Function(self.V)
        primal.interpolate(self.lamda_inv*x)
        for _ in range(self.m//2):
            primal = self._GinvM(primal)
        return assemble(inner(primal, primal)*dx)

    def correlated_noise(self, x=None, v=None):
        """
        Correlate white noise: v = B^{1/2}x

        v = l(M^{-1}G)^{m/2}M^{-1/2}x
        """
        x = x or self.uncorrelated_noise()
        v = v or Function(self.V)
        w = Function(self.V)
        w.interpolate(self.Msqrt_inv*x)
        for _ in range(self.m//2):
            w = self._MinvG(w)
        v.interpolate(self.lamda*w)
        return v


class ImplicitFormCorrelationBase(FormCorrelationOperatorBase):
    """Correlation operator is the inverse of a finite element form
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||((M^{-1}G)^(m/2)x)|| with:
    M: V -> V* = <u,v>
    G: V -> V*
    i.e.
    B^{-1} = (G*Minv)^{T}M(Minv*G) = (G*Minv)*M*(Minv*G)
    B = (Ginv*M)^(m/2)*Minv*(M*Ginv)^(m/2)
    """
    def apply(self, y, x=None):
        """Return x = By
        B: V* -> V = l((G^{-1}M)^m)M{-1}l
        """
        x = x or Function(self.V)
        primal = Function(self.V).assign(self.riesz(y))

        primal.interpolate(self.lamda*primal)
        for _ in range(self.m):
            primal = self._GinvM(primal)
        primal.interpolate(self.lamda*primal)

        return x.assign(primal)

    def solve(self, x, y=None):
        """Return y = B^{-1}x
        B^{-1}: V -> V* = linvM(M^{-1}G)^{m}linv
        """
        y = y or Function(self.V.dual())
        primal = Function(self.V).assign(x)

        primal.interpolate(self.lamda_inv*primal)
        for _ in range(self.m):
            primal = self._MinvG(primal)
        primal.interpolate(self.lamda_inv*primal)

        return y.assign(self.riesz(primal))

    def norm(self, x):
        """Return x^{T}B^{-1}x = ||x||^2_{B^{-1}}

        x^{T}B^{-1}x = x^{T}B^{-T/2}MB^{-1/2}x
                     = || B^{-1/2}x ||^2_{M}
                     = || ((M^{-1}G)^{m/2}linv)x ||^2_{M}
        """
        primal = Function(self.V).assign(x)
        primal.interpolate(self.lamda_inv*primal)
        for _ in range(self.m//2):
            primal = self._MinvG(primal)
        return assemble(inner(primal, primal)*dx)

    def correlated_noise(self, x=None, v=None):
        """
        Correlate white noise: v = B^{1/2}x

        v = l(G^{-1}M)^{m/2}M^{-1/2}x
        """
        x = x or self.uncorrelated_noise()
        v = v or Function(self.V)
        w = Function(self.V).assign(x)

        w.interpolate(self.Msqrt_inv*w)
        for _ in range(self.m//2):
            w = self._GinvM(w)
        v.interpolate(self.lamda*w)
        return v


class ExplicitMassCorrelation(ExplicitFormCorrelationBase):
    """Correlation operator is the action of a weighted mass matrix
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(G^{-1}Mx)|| with:
    G = sigma^2*(<u, v>)^2
    i.e.
    B^{-1} = (Ginv*M)^{T}M(Ginv*M) = M*Ginv*M*Ginv*M
    B = Minv*G*Minv*G*Minv
    """
    def __init__(self, V, sigma, m=2, solver_parameters=None,
                 bcs=None, generator=None, seed=None, Msqrt_inv=None):
        self.sigma = sigma
        self.lamda = self.sigma
        self.lamda_inv = (1/self.lamda)

        super().__init__(V, m=m, solver_parameters=solver_parameters, bcs=bcs,
                         generator=generator, seed=seed, Msqrt_inv=Msqrt_inv)

    def form(self, u, v):
        return inner(u, v)*dx


class ImplicitMassCorrelation(ImplicitFormCorrelationBase):
    """Correlation operator is the inverse of a finite element form
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(M^{-1}Gx)|| with:
    G = sigma^2*(<u, v)^{-2}
    i.e.
    B^{-1} = (Minv*G)^{T}M(Minv*G) = (G*Minv)*M*(Minv*G)
    B = Ginv*M*Ginv
    """
    def __init__(self, V, sigma, m=2, solver_parameters=None,
                 bcs=None, generator=None, seed=None, Msqrt_inv=None):
        self.sigma = sigma
        self.lamda = self.sigma
        self.lamda_inv = (1/self.lamda)

        super().__init__(V, m=m, solver_parameters=solver_parameters, bcs=bcs,
                         generator=generator, seed=seed, Msqrt_inv=Msqrt_inv)

    def form(self, u, v):
        return inner(u, v)*dx


class DiffusionCorrelationMixin:
    def _generate_diffusion_parameters(self, sigma, L, m):
        kappa = Constant(L*L/(2*m))
        lamda_g = Constant(sqrt(2*pi)*L)
        lamda = Constant(sigma*sqrt(lamda_g))
        lamda_inv = Constant(1/lamda)
        return kappa, lamda, lamda_inv


class ExplicitDiffusionCorrelation(ExplicitFormCorrelationBase, DiffusionCorrelationMixin):
    """Correlation operator is the action of a diffusion operator
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(G^{-1}Mx)|| with:
    G = scale*sigma^2*(<u, v> - <kappa*grad(u), grad(v)>)^2
    i.e.
    B^{-1} = (Ginv*M)^{T}M(Ginv*M) = M*Ginv*M*Ginv*M
    B = Minv*G*Minv*G*Minv
    """
    def __init__(self, V, sigma, L, m=2, solver_parameters=None,
                 bcs=None, generator=None, seed=None, Msqrt_inv=None):
        kappa, lamda, lamda_inv = self._generate_diffusion_parameters(sigma, L, m)

        self.L = L
        self.sigma = sigma
        self.kappa = kappa
        self.lamda = lamda
        self.lamda_inv = lamda_inv

        nx = V.mesh().num_cells()
        cfl_nu = float(kappa*nx*nx)
        PETSc.Sys.Print(f"{float(kappa) = :.3e} | {cfl_nu = :.3e} | {float(lamda) = :.3e}")

        super().__init__(V, m=m, solver_parameters=solver_parameters, bcs=bcs,
                         generator=generator, seed=seed, Msqrt_inv=Msqrt_inv)

    def form(self, u, v):
        return (inner(u, v)*dx - inner(self.kappa*grad(u), grad(v))*dx)


class ImplicitDiffusionCorrelation(ImplicitFormCorrelationBase, DiffusionCorrelationMixin):
    """Correlation operator is the inverse of a diffusion operator
    x^{T}B^{-1}x = ||x||_{B^{-1}} = ||(M^{-1}Gx)|| with:
    G = scale*sigma^2*(<u, v> + <kappa*grad(u), grad(v)>)^{-2}
    i.e.
    B^{-1} = (Minv*G)^{T}M(Minv*G) = (G*Minv)*M*(Minv*G)
    B = Ginv*M*Ginv
    """
    def __init__(self, V, sigma, L, m=2, solver_parameters=None,
                 bcs=None, generator=None, seed=None, Msqrt_inv=None):
        kappa, lamda, lamda_inv = self._generate_diffusion_parameters(sigma, L, m)

        self.L = L
        self.kappa = kappa
        self.sigma = sigma
        self.lamda = lamda
        self.lamda_inv = lamda_inv

        nx = V.mesh().num_cells()
        cfl_nu = float(kappa*nx*nx)
        PETSc.Sys.Print(f"{float(kappa)=:.2e} | {cfl_nu=:.2e} | {float(lamda)=:.2e}")

        super().__init__(V, m=m, solver_parameters=solver_parameters, bcs=bcs,
                         generator=generator, seed=seed, Msqrt_inv=Msqrt_inv)

    def form(self, u, v):
        return (inner(u, v)*dx + inner(self.kappa*grad(u), grad(v))*dx)


class CorrelationOperatorPC:
    """
    Precondition the inverse correlation operator:
    P = B : V* -> V
    """
    def __init__(self):
        self.initialized = False

    def setUp(self, pc):
        if self.initialized:
            self.update(pc)
        else:
            self.initialize(pc)
            self.initialized = True

    def initialize(self, pc):
        _, P = pc.getOperators()
        correlation_mat = P.getPythonContext()
        if not isinstance(correlation_mat, CorrelationOperatorMatCtx):
            raise TypeError(
                "CorrelationOperatorPC needs a CorrelationOperatorMatCtx")
        correlation = correlation_mat.correlation

        self.correlation = correlation
        self.correlation_mat = correlation_mat

        V = correlation.V
        primal = Function(V)
        dual = Function(V.dual())

        # PC does the opposite of the Mat
        if correlation_mat.action == 'apply':
            self.x = primal
            self.y = dual
            self._apply_op = correlation.solve
        elif correlation_mat.action == 'solve':
            self.x = dual
            self.y = primal
            self._apply_op = correlation.apply

        self.update(pc)

    def apply(self, pc, x, y):
        with self.x.dat.vec_wo as xvec:
            x.copy(result=xvec)

        self._apply_op(self.x, self.y)

        with self.y.dat.vec_ro as yvec:
            yvec.copy(result=y)

    def update(self, pc):
        pass


class CorrelationOperatorMatCtx:
    def __init__(self, correlation, action='solve'):
        self.comm = correlation.V.mesh().comm
        self.correlation = correlation
        self.action = action
        self.V = correlation.V

        primal = Function(self.V)
        dual = Function(self.V.dual())

        if action == 'apply':
            self.x = dual
            self.y = primal
            self._mult_op = correlation.apply
        elif action == 'solve':
            self.x = primal
            self.y = dual
            self._mult_op = correlation.solve
        else:
            raise ValueError(
                f"CorrelationOperatorMatCtx action must be 'solve' or 'apply', not {action}.")

    def mult(self, A, x, y):
        with self.x.dat.vec_wo as v:
            x.copy(result=v)

        self._mult_op(self.x, self.y)

        with self.y.dat.vec_ro as v:
            v.copy(result=y)


def CorrelationOperatorMat(correlation, action='solve'):
    ctx = CorrelationOperatorMatCtx(
        correlation, action=action)

    sizes = correlation.V.dof_dset.layout_vec.getSizes()

    mat = PETSc.Mat().createPython(
        (sizes, sizes), ctx, comm=ctx.comm)

    mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    mat.setUp()
    mat.assemble()
    return mat
