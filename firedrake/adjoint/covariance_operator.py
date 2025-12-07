from enum import Enum
from functools import cached_property
from textwrap import dedent
from scipy.special import factorial
from petsctools import get_petscvariables, PCBase
from loopy import generate_code_v2
from pyop2 import op2
from firedrake.tsfc_interface import compile_form
from firedrake import (
    grad, inner, avg, action, outer, replace,
    assemble, CellSize, FacetNormal,
    dx, ds, dS, sqrt, Constant,
    Function, Cofunction, RieszMap,
    TrialFunction, TestFunction,
    FunctionSpace, VectorFunctionSpace,
    BrokenElement, VectorElement,
    RandomGenerator, PCG64,
    LinearVariationalProblem,
    LinearVariationalSolver,
    LinearSolver,
    PETSc
)


class CholeskyFactorisation:
    def __init__(self, V, form=None):
        self._V = V

        if form is None:
            self.form = inner(TrialFunction(V),
                              TestFunction(V))*dx
        else:
            self.form = form

        self._wrk = Function(V)

    @property
    def function_space(self):
        return self._V

    @cached_property
    def _assemble_action(self):
        from firedrake.assemble import get_assembler
        return get_assembler(action(self.form, self._wrk)).assemble

    def assemble_action(self, u, tensor=None):
        self._wrk.assign(u)
        return self._assemble_action(tensor=tensor)

    @cached_property
    def solver(self):
        return LinearSolver(
            assemble(self.form, mat_type='aij'),
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "cholesky",
                "pc_factor_mat_ordering_type": "nd"})

    @cached_property
    def pc(self):
        return self.solver.ksp.getPC()

    def apply(self, u):
        u = self.assemble_action(u)
        v = Cofunction(self.space.dual())
        with u.dat.vec_ro as u_v, v.dat.vec_wo as v_v:
            self.pc.applySymmetricLeft(u_v, v_v)
        return v

    def apply_transpose(self, u):
        v = Function(self.function_space)
        with u.dat.vec_ro as u_v, v.dat.vec_wo as v_v:
            self.pc.applySymmetricRight(u_v, v_v)
        v = self.assemble_action(v)
        return v


class NoiseBackendBase:
    def __init__(self, V, rng=None):
        self._V = V
        self._rng = rng or RandomGenerator(PCG64())

    def sample(self, *, rng=None, tensor=None):
        raise NotImplementedError

    @cached_property
    def broken_space(self):
        element = self.function_space.ufl_element()
        mesh = self.function_space.mesh().unique()
        if isinstance(element, VectorElement):
            dim = element.num_sub_elements
            scalar_element = element.sub_elements[0]
            broken_element = BrokenElement(scalar_element)
            Vbroken = VectorFunctionSpace(
                mesh, broken_element, dim=dim)
        else:
            Vbroken = FunctionSpace(
                mesh, BrokenElement(element))
        return Vbroken

    @property
    def function_space(self):
        return self._V

    @property
    def rng(self):
        return self._rng

    @cached_property
    def riesz_map(self):
        return RieszMap(self.function_space, constant_jacobian=True)


class PetscNoiseBackend(NoiseBackendBase):
    def __init__(self, V, rng=None):
        super().__init__(V, rng=rng)
        self.cholesky = CholeskyFactorisation(self.broken_space)

    def sample(self, *, rng=None, tensor=None, apply_riesz=False):
        V = self.function_space
        rng = rng or self.rng

        # z
        z = rng.standard_normal(self.broken_space)
        # C z
        Cz = self.cholesky.apply_transpose(z)
        # L C z
        b = Cofunction(V.dual()).interpolate(Cz)

        if apply_riesz:
            b = b.riesz_representation(self.riesz_map)

        if tensor:
            tensor.assign(b)
        else:
            tensor = b

        return tensor


class PyOP2NoiseBackend(NoiseBackendBase):
    def __init__(self, V, rng=None):
        super().__init__(V, rng=rng)

        u = TrialFunction(V)
        v = TestFunction(V)
        mass = inner(u, v)*dx

        # Create mass expression, assemble and extract kernel
        mass_ker, *stuff = compile_form(mass, "mass")
        mass_code = generate_code_v2(mass_ker.kinfo.kernel.code).device_code()
        mass_code = mass_code.replace(
            "void " + mass_ker.kinfo.kernel.name,
            "static void " + mass_ker.kinfo.kernel.name)

        # Add custom code for doing Cholesky
        # decomposition and applying to broken vector
        name = mass_ker.kinfo.kernel.name
        blocksize = mass_ker.kinfo.kernel.code[name].args[0].shape[0]

        cholesky_code = dedent(
            f"""\
            extern void dpotrf_(char *UPLO,
                                int *N,
                                double *A,
                                int *LDA,
                                int *INFO);

            extern void dgemv_(char *TRANS,
                               int *M,
                               int *N,
                               double *ALPHA,
                               double *A,
                               int *LDA,
                               double *X,
                               int *INCX,
                               double *BETA,
                               double *Y,
                               int *INCY);

            {mass_code}

            void apply_cholesky(double *__restrict__ z,
                                double *__restrict__ b,
                                double const *__restrict__ coords)
            {{
                char uplo[1];
                int32_t N = {blocksize}, LDA = {blocksize}, INFO = 0;
                int32_t i=0, j=0;
                uplo[0] = 'u';
                double H[{blocksize}*{blocksize}] = {{{{ 0.0 }}}};

                char trans[1];
                int32_t stride = 1;
                double scale = 1.0;
                double zero = 0.0;

                {mass_ker.kinfo.kernel.name}(H, coords);

                uplo[0] = 'u';
                dpotrf_(uplo, &N, H, &LDA, &INFO);
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        if (j>i)
                            H[i*N + j] = 0.0;

                trans[0] = 'T';
                dgemv_(trans, &N, &N, &scale, H, &LDA, z, &stride, &zero, b, &stride);
            }}
            """
        )

        # Get the BLAS and LAPACK compiler parameters to compile the kernel
        # TODO: Ask CW if this is the right comm to use.
        comm = V.mesh()._comm
        if comm.rank == 0:
            petsc_variables = get_petscvariables()
            BLASLAPACK_LIB = petsc_variables.get("BLASLAPACK_LIB", "")
            BLASLAPACK_LIB = comm.bcast(BLASLAPACK_LIB, root=0)
            BLASLAPACK_INCLUDE = petsc_variables.get("BLASLAPACK_INCLUDE", "")
            BLASLAPACK_INCLUDE = comm.bcast(BLASLAPACK_INCLUDE, root=0)
        else:
            BLASLAPACK_LIB = comm.bcast(None, root=0)
            BLASLAPACK_INCLUDE = comm.bcast(None, root=0)

        self.cholesky_kernel = op2.Kernel(
            cholesky_code, "apply_cholesky",
            include_dirs=BLASLAPACK_INCLUDE.split(),
            ldargs=BLASLAPACK_LIB.split())

    def sample(self, *, rng=None, tensor=None, apply_riesz=False):
        rng = rng or self.rng

        z = rng.standard_normal(self.broken_space)
        b = Cofunction(self.function_space.dual())

        z_arg = z.dat(op2.READ, self.broken_space.cell_node_map())
        b_arg = b.dat(op2.INC, self.function_space.cell_node_map())

        mesh = self.function_space.mesh()
        coords = mesh.coordinates
        c_arg = coords.dat(op2.READ, coords.cell_node_map())

        op2.par_loop(
            self.cholesky_kernel,
            mesh.cell_set,
            z_arg, b_arg, c_arg
        )

        if apply_riesz:
            b = b.riesz_representation(self.riesz_map)

        if tensor:
            tensor.assign(b)
        else:
            tensor = b

        return tensor


class WhiteNoiseGenerator:
    r""" Generates a white noise sample

    :arg V: The :class:`firedrake.FunctionSpace` to construct a
        white noise sample on
    :arg backend: The :enum:`WhiteNoiseGenerator.Backend` specifying how to calculate
        and apply the mass matrix square root.
    :arg rng: Initialised random number generator to use for obtaining
        random numbers. Defaults to PCG64.

    Returns a :firedrake.Function: with
    b ~ Normal(0, M)
    where b is the dat.data of the function returned
    and M is the mass matrix.

    For details see [Croci et al 2018]:
    https://epubs.siam.org/doi/10.1137/18M1175239
    """
    # TODO: Add Croci to citations manager

    class Backend(Enum):
        PYOP2 = 'pyop2'
        PETSC = 'petsc'

    def __init__(self, V, backend=None, rng=None):
        backend = backend or self.Backend.PYOP2
        if backend == self.Backend.PYOP2:
            self.backend = PyOP2NoiseBackend(V, rng=rng)
        elif backend == self.Backend.PETSC:
            self.backend = PetscNoiseBackend(V, rng=rng)
        else:
            raise ValueError(
                f"Unrecognised white noise generation backend {backend}")

        self.function_space = self.backend.function_space
        self.rng = self.backend.rng

    def sample(self, *, rng=None, tensor=None, apply_riesz=False):
        return self.backend.sample(
            rng=rng, tensor=tensor, apply_riesz=apply_riesz)


# Auto-regressive function parameters

def lengthscale_m(Lar: float, M: int):
    """Daley-equivalent lengthscale of M-th order autoregressive function.

    Parameters
    ----------
        Lar :
            Target Daley correlation lengthscale.
        M :
            Order of autoregressive function.

    Returns
    -------
        L :
            Lengthscale parameter for autoregressive function.
    """
    return Lar/sqrt(2*M - 3)


def lambda_m(Lar: float, M: int):
    """Normalisation factor for autoregressive function.

    Parameters
    ----------
        Lar :
            Target Daley correlation lengthscale.
        M :
            Order of autoregressive function.

    Returns
    -------
        lambda :
            Normalisation coefficient for autoregressive correlation operator.
    """
    L = lengthscale_m(Lar, M)
    num = (2**(2*M - 1))*factorial(M - 1)**2
    den = factorial(2*M - 2)
    return L*num/den


def kappa_m(Lar: float, M: int):
    """Diffusion coefficient for autoregressive function.

    Parameters
    ----------
        Lar :
            Target Daley correlation lengthscale.
        M :
            Order of autoregressive function.

    Returns
    -------
        kappa :
            Diffusion coefficient for autoregressive covariance operator.
    """
    return lengthscale_m(Lar, M)**2


class GaussianCovariance:
    class DiffusionForm(Enum):
        CG = 'CG'
        IP = 'IP'

    def __init__(self, V, L, sigma=1, m=2, rng=None,
                 bcs=None, form=None, function_space=None,
                 solver_parameters=None, options_prefix=None,
                 mass_parameters=None, mass_prefix=None):

        form = form or self.DiffusionForm.CG

        self.rng = rng or WhiteNoiseGenerator(V)
        self.function_space = function_space or self.rng.function_space

        if sigma <= 0:
            raise ValueError("Variance must be positive.")
        if L < 0:
            raise ValueError("Correlation lengthscale must be positive.")
        if m < 0:
            raise ValueError("Number of iterations must be positive.")
        if (m % 2) != 0:
            raise ValueError("Number of iterations must be even.")

        self.stddev = sigma
        self.lengthscale = L
        self.iterations = m

        if self.iterations > 0:
            # Calculate diffusion operator parameters
            self.kappa = Constant(kappa_m(L, m))
            self.lambda_m = Constant(lambda_m(L, m))
            self._weight = Constant(sigma*sqrt(self.lambda_m))

            # setup diffusion solver
            u, v = TrialFunction(V), TestFunction(V)
            if isinstance(form, self.DiffusionForm):
                K = diffusion_form(u, v, self.kappa, formulation=form)
            else:
                K = form

            M = inner(u, v)*dx

            self._u = Function(V)
            self._b = Cofunction(V.dual())

            self._Mrhs = replace(M, {u: self._u})
            self._Krhs = replace(K, {u: self._u})

            self.solver = LinearVariationalSolver(
                LinearVariationalProblem(K, self._b, self._u, bcs=bcs,
                                         constant_jacobian=True),
                solver_parameters=solver_parameters,
                options_prefix=options_prefix)

            self.mass_solver = LinearVariationalSolver(
                LinearVariationalProblem(M, self._b, self._u, bcs=bcs,
                                         constant_jacobian=True),
                solver_parameters=mass_parameters,
                options_prefix=mass_prefix)

    def sample(self, *, rng=None, tensor=None):
        tensor = tensor or Function(self.function_space)
        rng = rng or self.rng

        if self.iterations == 0:
            w = rng.sample(apply_riesz=True)
            return tensor.assign(self.stddev*w)

        w = rng.sample(apply_riesz=False)

        for i in range(self.iterations//2):
            if i == 0:
                self._b.assign(w)
            else:
                assemble(self._Mrhs, tensor=self._b)
            self.solver.solve()

        return tensor.assign(self._weight*self._u)

    def norm(self, x):
        if self.iterations == 0:
            sigma_x = self.stddev*x
            return assemble(inner(sigma_x, sigma_x)*dx)

        lamda1 = 1/self._weight
        self._u.assign(lamda1*x)

        for i in range(self.iterations//2):
            assemble(self._Krhs, tensor=self._b)
            self.mass_solver.solve()

        return assemble(inner(self._u, self._u)*dx)

    def apply_inverse(self, x, *, tensor=None):
        """B^{-1} : V -> V*
        """
        tensor = tensor or Cofunction(self.function_space.dual())

        if self.iterations == 0:
            riesz_map = self.rng.backend.riesz_map
            Cx = x.riesz_representation(riesz_map)
            variance1 = 1/(self.stddev*self.stddev)
            return tensor.assign(variance1*Cx)

        lamda1 = Constant(1/self._weight)
        self._u.assign(lamda1*x)

        for i in range(self.iterations):
            assemble(self._Krhs, tensor=self._b)
            if i != self.iterations - 1:
                self.mass_solver.solve()

        return tensor.assign(lamda1*self._b)

    def apply_action(self, x, *, tensor=None):
        """B : V* -> V
        """
        tensor = tensor or Function(self.function_space)

        if self.iterations == 0:
            riesz_map = self.rng.backend.riesz_map
            Cx = x.riesz_representation(riesz_map)
            variance = self.stddev*self.stddev
            return tensor.assign(variance*Cx)

        for i in range(self.iterations):
            if i == 0:
                self._b.assign(self._weight*x)
            else:
                assemble(self._Mrhs, tensor=self._b)
            self.solver.solve()

        return tensor.assign(self._weight*self._u)


def diffusion_form(u, v, kappa, formulation):
    if formulation == GaussianCovariance.DiffusionForm.CG:
        return inner(u, v)*dx + inner(kappa*grad(u), grad(v))*dx

    elif formulation == GaussianCovariance.DiffusionForm.IP:
        mesh = v.function_space().mesh()
        n = FacetNormal(mesh)
        h = CellSize(mesh)
        h_avg = 0.5*(h('+') + h('-'))
        alpha_h = Constant(4.0)/h_avg
        gamma_h = Constant(8.0)/h
        return (
            inner(u, v)*dx + kappa*(
                inner(grad(u), grad(v))*dx
                - inner(avg(2*outer(u, n)), avg(grad(v)))*dS
                - inner(avg(grad(u)), avg(2*outer(v, n)))*dS
                + alpha_h*inner(avg(2*outer(u, n)), avg(2*outer(v, n)))*dS
                - inner(outer(u, n), grad(v))*ds
                - inner(grad(u), outer(v, n))*ds
                + gamma_h*inner(u, v)*ds
            )
        )

    else:
        raise ValueError("Unknown GaussianCovariance.DiffusionForm {formulation}")


class CovarianceMatCtx:
    class Operation(Enum):
        ACTION = 'action'
        INVERSE = 'inverse'

    def __init__(self, covariance, operation=None):
        operation = operation or self.Operation.ACTION

        V = covariance.function_space
        self.function_space = V
        self.comm = V.mesh().comm
        self.covariance = covariance
        self.operation = operation

        primal = Function(V)
        dual = Function(V.dual())

        if operation == self.Operation.ACTION:
            self.x = dual
            self.y = primal
            self._mult_op = covariance.apply_action
        elif operation == self.Operation.INVERSE:
            self.x = primal
            self.y = dual
            self._mult_op = covariance.apply_inverse
        else:
            raise ValueError(
                f"Unrecognised CovarianceMat operation {operation}")

    def mult(self, mat, x, y):
        with self.x.dat.vec_wo as v:
            x.copy(result=v)

        self._mult_op(self.x, tensor=self.y)

        with self.y.dat.vec_ro as v:
            v.copy(result=y)

    def view(self, mat, viewer=None):
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return

        viewer.printfASCII(f"  firedrake covariance operator matrix: {type(self).__name__}\n")
        viewer.printfASCII(f"  Applying the {str(self.operation)} of the covariance operator {type(self.covariance).__name__}\n")

        if (type(self.covariance) is GaussianCovariance) and (self.covariance.iterations > 0):
            viewer.printfASCII("  Autoregressive covariance operator with:\n")
            viewer.printfASCII(f"    order: {self.covariance.iterations}\n")
            viewer.printfASCII(f"    correlation lengthscale: {self.covariance.lengthscale}\n")
            viewer.printfASCII(f"    standard deviation: {self.covariance.stddev}\n")

            if self.operation == self.Operation.ACTION:
                viewer.printfASCII("  Information for the diffusion solver for applying the action:\n")
                ksp = self.covariance.solver.snes.ksp
            elif self.operation == self.Operation.INVERSE:
                viewer.printfASCII("  Information for the mass solver for applying the inverse:\n")
                ksp = self.covariance.mass_solver.snes.ksp
            level = ksp.getTabLevel()
            ksp.setTabLevel(mat.getTabLevel() + 1)
            ksp.view(viewer)
            ksp.setTabLevel(level)


def CovarianceMat(covariance, operation=None):
    ctx = CovarianceMatCtx(covariance, operation=operation)

    sizes = covariance.function_space.dof_dset.layout_vec.getSizes()

    mat = PETSc.Mat().createPython(
        (sizes, sizes), ctx, comm=ctx.comm)
    mat.setUp()
    mat.assemble()
    return mat


class CovariancePC(PCBase):
    """
    Precondition the inverse covariance operator:
    P = B : V* -> V
    """
    needs_python_pmat = True
    prefix = "covariance"

    def initialize(self, pc):
        A, P = pc.getOperators()

        use_amat_prefix = self.parent_prefix + "pc_use_amat"
        self.use_amat = PETSc.Options().getBool(use_amat_prefix, False)
        mat = (A if self.use_amat else P).getPythonContext()

        if not isinstance(mat, CovarianceMatCtx):
            raise TypeError(
                "CovariancePC needs a CovarianceMatCtx")
        covariance = mat.covariance

        self.covariance = covariance
        self.mat = mat

        V = covariance.function_space
        primal = Function(V)
        dual = Function(V.dual())

        # PC does the opposite of the Mat
        if mat.operation == CovarianceMatCtx.Operation.ACTION:
            self.operation = CovarianceMatCtx.Operation.INVERSE
            self.x = primal
            self.y = dual
            self._apply_op = covariance.apply_inverse
        elif mat.operation == CovarianceMatCtx.Operation.INVERSE:
            self.operation = CovarianceMatCtx.Operation.ACTION
            self.x = dual
            self.y = primal
            self._apply_op = covariance.apply_action

    def apply(self, pc, x, y):
        with self.x.dat.vec_wo as xvec:
            x.copy(result=xvec)

        self._apply_op(self.x, tensor=self.y)

        with self.y.dat.vec_ro as yvec:
            yvec.copy(result=y)

    def update(self, pc):
        pass

    def view(self, pc, viewer=None):
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return

        viewer.printfASCII(f"  firedrake covariance operator preconditioner: {type(self).__name__}\n")
        viewer.printfASCII(f"  Applying the {str(self.operation)} of the covariance operator {type(self.covariance).__name__}\n")

        if self.use_amat:
            viewer.printfASCII("  using Amat matrix\n")

        if (type(self.covariance) is GaussianCovariance) and (self.covariance.iterations > 0):
            if self.operation == CovarianceMatCtx.Operation.ACTION:
                viewer.printfASCII("  Information for the diffusion solver for applying the action:\n")
                self.covariance.solver.snes.ksp.view(viewer)
            elif self.operation == CovarianceMatCtx.Operation.INVERSE:
                viewer.printfASCII("  Information for the mass solver for applying the inverse:\n")
                self.covariance.mass_solver.snes.ksp.view(viewer)
