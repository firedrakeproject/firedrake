import abc
from enum import Enum
from functools import cached_property
from typing import Iterable
from textwrap import dedent
from scipy.special import factorial
import petsctools
from loopy import generate_code_v2
from pyop2 import op2
from firedrake.tsfc_interface import compile_form
from firedrake.adjoint.transformed_functional import L2Cholesky
from firedrake.functionspaceimpl import WithGeometry
from firedrake.bcs import BCBase
from firedrake import (
    grad, inner, avg, action, outer,
    assemble, CellSize, FacetNormal,
    dx, ds, dS, sqrt, Constant,
    Function, Cofunction, RieszMap,
    TrialFunction, TestFunction,
    FunctionSpace, VectorFunctionSpace,
    BrokenElement, VectorElement,
    RandomGenerator, PCG64,
    LinearVariationalProblem,
    LinearVariationalSolver,
    VertexOnlyMeshTopology,
    PETSc
)


class NoiseBackendBase:
    r"""
    A base class for implementations of a mass matrix square root action
    for generating white noise samples.

    Inheriting classes implement the method from [Croci et al 2018](https://epubs.siam.org/doi/10.1137/18M1175239)

    Generating the samples on the function space :math:`V` requires the following steps:

    1. On each element generate a white noise sample :math:`z_{e}\sim\mathcal{N}(0, I)`
       over all DoFs in the element. Equivalantly, generate the sample on the
       discontinuous superspace :math:`V_{d}^{*}` containing :math:`V^{*}`.
       (i.e. ``Vd.ufl_element() = BrokenElement(V.ufl_element``).

    2. Apply the Cholesky factor :math:`C_{e}` of the element-wise mass matrix :math:`M_{e}`
       to the element-wise sample (:math:`M_{e}=C_{e}C_{e}^{T}`).

    3. Assemble the element-wise samples :math:`z_{e}\in V_{d}^{*}` into the global
       sample vector :math:`z\in V^{*}`. If :math:`L` is the interpolation operator
       then :math:`z=Lz_{e}=LC_{e}z_{e}`.

    4. Optionally apply a Riesz map to :math:`z` to return a sample in :math:`V`.

    Parameters
    ----------
    V :
        The :func:`~.firedrake.functionspace.FunctionSpace` to generate the samples in.
    rng :
        The :mod:`RandomGenerator <firedrake.randomfunctiongen>` to generate the samples
        on the discontinuous superspace.
    seed :
        Seed for the :mod:`RandomGenerator <firedrake.randomfunctiongen>`.
        Ignored if ``rng`` is given.

    See Also
    --------
    PyOP2NoiseBackend
    PetscNoiseBackend
    WhiteNoiseGenerator
    """

    def __init__(self, V: WithGeometry, rng=None,
                 seed: int | None = None):
        self._V = V
        self._rng = rng or RandomGenerator(PCG64(seed=seed))

    @abc.abstractmethod
    def sample(self, *, rng=None,
               tensor: Function | Cofunction | None = None,
               apply_riesz: bool = False):
        """
        Generate a white noise sample.

        Parameters
        ----------
        rng :
            A :mod:`RandomGenerator <firedrake.randomfunctiongen>` to use for
            sampling IID vectors.  If ``None`` then ``self.rng`` is used.

        tensor :
            Optional location to place the result into.

        apply_riesz :
            Whether to apply the L2 Riesz map to return a sample in :math:`V`.

        Returns
        -------
        Function | Cofunction :
            The white noise sample in :math:`V`
        """
        raise NotImplementedError

    @cached_property
    def broken_space(self):
        """
        The discontinuous superspace containing :math:`V`, ``self.function_space``.
        """
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
        """The function space that the noise will be generated on.
        """
        return self._V

    @property
    def rng(self):
        """The :mod:`RandomGenerator <firedrake.randomfunctiongen>` to generate the
        IID sample on the broken function space.
        """
        return self._rng

    @cached_property
    def riesz_map(self):
        """A :class:`~firedrake.cofunction.RieszMap` to cache the solver
        for :meth:`~firedrake.cofunction.Cofunction.riesz_representation`.
        """
        return RieszMap(self.function_space, constant_jacobian=True)
        """
        Generate a white noise sample.

        Parameters
        ----------
        rng :
            A :mod:`RandomGenerator <firedrake.randomfunctiongen>` to use for
            sampling IID vectors. If ``None`` then ``self.rng`` is used.

        tensor :
            Optional location to place the result into.

        apply_riesz :
            Whether to apply an L2 Riesz map to the result to return
            a sample in the primal space.
        """


class PyOP2NoiseBackend(NoiseBackendBase):
    """
    A PyOP2 based implementation of a mass matrix square root
    for generating white noise.
    """
    def __init__(self, V: WithGeometry, rng=None,
                 seed: int | None = None):
        super().__init__(V, rng=rng, seed=seed)

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
            petsc_variables = petsctools.get_petscvariables()
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

    def sample(self, *, rng=None,
               tensor: Function | Cofunction | None = None,
               apply_riesz: bool = False):
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


class PetscNoiseBackend(NoiseBackendBase):
    """
    A PETSc based implementation of a mass matrix square root action
    for generating white noise.
    """
    def __init__(self, V: WithGeometry, rng=None,
                 seed: int | None = None):
        super().__init__(V, rng=rng, seed=seed)
        self.cholesky = L2Cholesky(self.broken_space)
        self._zb = Function(self.broken_space)
        self.M = inner(self._zb, TestFunction(self.broken_space))*dx

    def sample(self, *, rng=None,
               tensor: Function | Cofunction | None = None,
               apply_riesz: bool = False):
        V = self.function_space
        rng = rng or self.rng

        # z
        z = rng.standard_normal(self.broken_space)
        # C z
        self._zb.assign(self.cholesky.C_T_inv_action(z))
        Cz = assemble(self.M)
        # L C z
        b = Cofunction(V.dual()).interpolate(Cz)

        if apply_riesz:
            b = b.riesz_representation(self.riesz_map)

        if tensor:
            tensor.assign(b)
        else:
            tensor = b

        return tensor


class VOMNoiseBackend(NoiseBackendBase):
    """
    A PETSc based implementation of a mass matrix square root action
    for generating white noise on a vertex only mesh.
    """
    def __init__(self, V: WithGeometry, rng=None,
                 seed: int | None = None):
        super().__init__(V, rng=rng, seed=seed)
        self.cholesky = L2Cholesky(V)
        self._zb = Function(V)
        self.M = inner(self._zb, TestFunction(V))*dx

    def sample(self, *, rng=None,
               tensor: Function | Cofunction | None = None,
               apply_riesz: bool = False):
        rng = rng or self.rng

        # z
        z = rng.standard_normal(self.broken_space)
        # C z
        self._zb.assign(self.cholesky.C_T_inv_action(z))
        Cz = assemble(self.M)

        # Usually we would interpolate to the unbroken space,
        # but here we're on a VOM so everything is broken.
        # L C z
        # b = Cofunction(V.dual()).interpolate(Cz)
        b = Cz

        if apply_riesz:
            b = b.riesz_representation(self.riesz_map)

        if tensor:
            tensor.assign(b)
        else:
            tensor = b

        return tensor


class WhiteNoiseGenerator:
    r"""Generate white noise samples.

    Generates samples :math:`w\in V^{*}` with
    :math:`w\sim\mathcal{N}(0, M)`, where :math:`M` is
    the mass matrix, or its Riesz representer in :math:`V`.

    Parameters
    ----------
    V :
        The :class:`~firedrake.functionspace.FunctionSpace` to construct a
        white noise sample on.
    backend :
        The backend to calculate and apply the mass matrix square root.
    rng :
        Initialised random number generator to use for sampling IID vectors.
    seed :
        Seed for the :mod:`RandomGenerator <firedrake.randomfunctiongen>`.
        Ignored if ``rng`` is given.

    References
    ----------
    Croci, M. and Giles, M. B and Rognes, M. E. and Farrell, P. E., 2018:
    "Efficient White Noise Sampling and Coupling for Multilevel Monte Carlo
    with Nonnested Meshes". SIAM/ASA J. Uncertainty Quantification, Vol. 6,
    No. 4, pp. 1630--1655.
    https://doi.org/10.1137/18M1175239

    See Also
    --------
    NoiseBackendBase
    PyOP2NoiseBackend
    PetscNoiseBackend
    VOMNoiseBackend
    CovarianceOperatorBase
    """

    def __init__(self, V: WithGeometry,
                 backend: NoiseBackendBase | None = None,
                 rng=None, seed: int | None = None):

        # Not all backends are valid for VOM.
        if isinstance(V.mesh().topology, VertexOnlyMeshTopology):
            backend = backend or VOMNoiseBackend(V, rng=rng, seed=seed)
            if not isinstance(backend, VOMNoiseBackend):
                raise ValueError(
                    f"Cannot use white noise backend {type(backend).__name__}"
                    " with a VertexOnlyMesh. Please use a VOMNoiseBackend.")
        else:
            backend = backend or PyOP2NoiseBackend(V, rng=rng, seed=seed)

        self.backend = backend
        self.function_space = backend.function_space
        self.rng = backend.rng

        petsctools.cite("Croci2018")

    def sample(self, *, rng=None,
               tensor: Function | Cofunction | None = None,
               apply_riesz: bool = False):
        """
        Generate a white noise sample.

        Parameters
        ----------
        rng :
            A :mod:`RandomGenerator <firedrake.randomfunctiongen>` to use for
            sampling IID vectors.  If ``None`` then ``self.rng`` is used.

        tensor :
            Optional location to place the result into.

        apply_riesz :
            Whether to apply the L2 Riesz map to return a sample in :math:`V`.

        Returns
        -------
        Function | Cofunction :
            The white noise sample
        """
        return self.backend.sample(
            rng=rng, tensor=tensor, apply_riesz=apply_riesz)


# Auto-regressive function parameters

def lengthscale_m(Lar: float, m: int):
    """Daley-equivalent lengthscale of m-th order autoregressive function.

    Parameters
    ----------
    Lar :
        Target Daley correlation lengthscale.
    m :
        Order of autoregressive function.

    Returns
    -------
    L : float
        Lengthscale parameter for autoregressive function.
    """
    return Lar/sqrt(2*m - 3)


def lambda_m(Lar: float, m: int):
    """Normalisation factor for autoregressive function.

    Parameters
    ----------
    Lar :
        Target Daley correlation lengthscale.
    m :
        Order of autoregressive function.

    Returns
    -------
    lambda : float
        Normalisation coefficient for autoregressive correlation operator.
    """
    L = lengthscale_m(Lar, m)
    num = (2**(2*m - 1))*factorial(m - 1)**2
    den = factorial(2*m - 2)
    return L*num/den


def kappa_m(Lar: float, m: int):
    """Diffusion coefficient for autoregressive function.

    Parameters
    ----------
    Lar :
        Target Daley correlation lengthscale.
    m :
        Order of autoregressive function.

    Returns
    -------
    kappa : float
        Diffusion coefficient for autoregressive covariance operator.
    """
    return lengthscale_m(Lar, m)**2


class CovarianceOperatorBase:
    r"""
    Abstract base class for a covariance operator B where

    .. math::

        B: V^{*} \to V \quad \text{and} \quad B^{-1}: V \to V^{*}

    The covariance operators can be used to:

    - calculate weighted norms :math:`\|x\|_{B^{-1}} = x^{T}B^{-1}x`
      to account for uncertainty in optimisation methods.

    - generate samples from the normal distribution :math:`\mathcal{N}(0, B)`
      using :math:`w = B^{1/2}z` where :math:`z\sim\mathcal{N}(0, I)`.

    Inheriting classes must implement the following methods:

    - ``sample``

    - ``apply_inverse``

    - ``apply_action``

    - ``rng``

    - ``function_space``

    They may optionally implement ``norm`` to provide a more
    efficient implementation.

    See Also
    --------
    WhiteNoiseGenerator
    AutoregressiveCovariance
    CovarianceMatCtx
    CovarianceMat
    CovariancePC
    """

    @abc.abstractmethod
    def rng(self):
        """:class:`~.WhiteNoiseGenerator` for generating samples.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def function_space(self):
        """The function space V that the covariance operator maps to.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, *, rng: WhiteNoiseGenerator | None = None,
               tensor: Function | None = None):
        r"""
        Sample from :math:`\mathcal{N}(0, B)` by correlating a
        white noise sample: :math:`w = B^{1/2}z`.

        Parameters
        ----------
        rng :
            Generator for the white noise sample.
            If not provided then ``self.rng`` will be used.
        tensor :
            Optional location to place the result into.

        Returns
        -------
        firedrake.function.Function :
            The sample.
        """
        raise NotImplementedError

    def norm(self, x: Function):
        r"""Return the weighted norm :math:`\|x\|_{B^{-1}} = x^{T}B^{-1}x`.

        Default implementation uses ``apply_inverse`` to first calculate
        the :class:`~firedrake.cofunction.Cofunction` :math:`y = B^{-1}x`,
        then returns :math:`y(x)`.

        Inheriting classes may provide more efficient specialisations.

        Parameters
        ----------
        x :
            The :class:`~firedrake.function.Function` to take the norm of.

        Returns
        -------
        pyadjoint.AdjFloat :
            The norm of ``x``.
        """
        return self.apply_inverse(x)(x)

    @abc.abstractmethod
    def apply_inverse(self, x: Function, *,
                      tensor: Cofunction | None = None):
        r"""Return :math:`y = B^{-1}x` where B is the covariance operator.
        :math:`B^{-1}: V \to V^{*}`.

        Parameters
        ----------
        x :
            The :class:`~firedrake.function.Function` to apply the inverse to.
        tensor :
            Optional location to place the result into.

        Returns
        -------
        firedrake.cofunction.Cofunction :
            The result of :math:`B^{-1}x`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def apply_action(self, x: Cofunction, *,
                     tensor: Function | None = None):
        r"""Return :math:`y = Bx` where B is the covariance operator.
        :math:`B: V^{*} \to V`.

        Parameters
        ----------
        x :
            The :class:`~firedrake.cofunction.Cofunction` to apply
            the action to.
        tensor :
            Optional location to place the result into.

        Returns
        -------
        firedrake.function.Function :
            The result of :math:`B^{-1}x`
        """
        raise NotImplementedError


class AutoregressiveCovariance(CovarianceOperatorBase):
    r"""
    An m-th order autoregressive covariance operator using an implicit diffusion operator.

    Covariance operator B with a kernel that is the ``m``-th autoregressive
    function can be calculated using ``m`` Backward Euler steps of a
    diffusion operator, where the diffusion coefficient is specified by
    the desired correlation lengthscale.

    If :math:`M` is the mass matrix, :math:`K` is the matrix for a single
    Backward Euler step, and :math:`\lambda` is a normalisation factor, then the
    m-th order correlation operator (unit variance) is:

    .. math::

        B: V^{*} \to V = \lambda((K^{-1}M)^{m}M^{-1})\lambda

        B^{-1}: V \to V^{*} = (1/\lambda)M(M^{-1}K)^{m}(1/\lambda)

    This formulation leads to an efficient implementations for :math:`B^{1/2}`
    by taking only m/2 steps of the diffusion operator. This can be used
    to calculate weighted norms and sample from :math:`\mathcal{N}(0,B)`.

    .. math::

        \|x\|_{B^{-1}} = \|(M^{-1}K)^{m/2}(1/\lambda)x\|_{M}

        w = B^{1/2}z = \lambda M^{-1}(MK^{-1})^{m/2}(M^{1/2}z)

    The white noise sample :math:`M^{1/2}z` is generated by a
    :class:`.WhiteNoiseGenerator`.

    Parameters
    ----------
    V :
        The function space that the covariance operator maps into.
    L :
        The correlation lengthscale.
    sigma :
        The standard deviation.
    m :
        The number of diffusion operator steps.
        Equal to the order of the autoregressive function kernel.
    rng :
        White noise generator to seed generating correlated samples.
    seed :
        Seed for the :mod:`RandomGenerator <firedrake.randomfunctiongen>`.
        Ignored if ``rng`` is given.
    form : AutoregressiveCovariance.DiffusionForm | ufl.Form | None
        The diffusion formulation or form. If a ``DiffusionForm`` then
        :func:`.diffusion_form` will be used to generate the diffusion
        form. Otherwise assumed to be a ufl.Form on ``V``.
        Defaults to ``AutoregressiveCovariance.DiffusionForm.CG``.
    bcs :
        Boundary conditions for the diffusion operator.
    solver_parameters :
        The PETSc options for the diffusion operator solver.
    options_prefix :
        The options prefix for the diffusion operator solver.
    mass_parameters :
        The PETSc options for the mass matrix solver.
    mass_prefix :
        The options prefix for the matrix matrix solver.

    References
    ----------
    Mirouze, I. and Weaver, A. T., 2010: "Representation of correlation
    functions in variational assimilation using an implicit diffusion
    operator". Q. J. R. Meteorol. Soc. 136: 1421–1443, July 2010 Part B.
    https://doi.org/10.1002/qj.643

    See Also
    --------
    WhiteNoiseGenerator
    CovarianceOperatorBase
    CovarianceMat
    CovariancePC
    diffusion_form
    """

    class DiffusionForm(Enum):
        """
        The diffusion operator formulation.

        See Also
        --------
        diffusion_form
        """
        CG = 'CG'
        IP = 'IP'

    def __init__(self, V: WithGeometry, L: float | Constant,
                 sigma: float | Constant = 1., m: int = 2,
                 rng: WhiteNoiseGenerator | None = None,
                 seed: int | None = None, form=None,
                 bcs: BCBase | Iterable[BCBase] | None = None,
                 solver_parameters: dict | None = None,
                 options_prefix: str | None = None,
                 mass_parameters: dict | None = None,
                 mass_prefix: str | None = None):

        form = form or self.DiffusionForm.CG
        if isinstance(form, str):
            form = self.DiffusionForm(form)

        self._rng = rng or WhiteNoiseGenerator(V, seed=seed)
        self._function_space = self.rng().function_space

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
            self._urhs = Function(V)

            self._Mrhs = action(M, self._urhs)
            self._Krhs = action(K, self._urhs)

            self.solver = LinearVariationalSolver(
                LinearVariationalProblem(K, self._Mrhs, self._u, bcs=bcs,
                                         constant_jacobian=True),
                solver_parameters=solver_parameters,
                options_prefix=options_prefix)

            self.mass_solver = LinearVariationalSolver(
                LinearVariationalProblem(M, self._Krhs, self._u, bcs=bcs,
                                         constant_jacobian=True),
                solver_parameters=mass_parameters,
                options_prefix=mass_prefix)

    def function_space(self):
        return self._function_space

    def rng(self):
        return self._rng

    def sample(self, *, rng: WhiteNoiseGenerator | None = None,
               tensor: Function | None = None):
        tensor = tensor or Function(self.function_space())
        rng = rng or self.rng()

        if self.iterations == 0:
            w = rng.sample(apply_riesz=True)
            return tensor.assign(self.stddev*w)

        w = rng.sample(apply_riesz=True)
        self._u.assign(w)

        for i in range(self.iterations//2):
            self._urhs.assign(self._u)
            self.solver.solve()

        return tensor.assign(self._weight*self._u)

    def norm(self, x: Function):
        if self.iterations == 0:
            sigma_x = (1/self.stddev)*x
            return assemble(inner(sigma_x, sigma_x)*dx)

        lamda1 = 1/self._weight
        self._u.assign(lamda1*x)

        for i in range(self.iterations//2):
            self._urhs.assign(self._u)
            self.mass_solver.solve()

        return assemble(inner(self._u, self._u)*dx)

    def apply_inverse(self, x: Function, *,
                      tensor: Cofunction | None = None):
        tensor = tensor or Cofunction(self.function_space().dual())

        if self.iterations == 0:
            riesz_map = self.rng().backend.riesz_map
            Cx = x.riesz_representation(riesz_map)
            variance1 = 1/(self.stddev*self.stddev)
            return tensor.assign(variance1*Cx)

        lamda1 = Constant(1/self._weight)
        self._u.assign(lamda1*x)

        for i in range(self.iterations):
            self._urhs.assign(self._u)
            if i != self.iterations - 1:
                self.mass_solver.solve()
        b = assemble(self._Krhs)

        return tensor.assign(lamda1*b)

    def apply_action(self, x: Cofunction, *,
                     tensor: Function | None = None):
        tensor = tensor or Function(self.function_space())

        riesz_map = self.rng().backend.riesz_map
        Cx = x.riesz_representation(riesz_map)

        if self.iterations == 0:
            variance = self.stddev*self.stddev
            return tensor.assign(variance*Cx)

        self._u.assign(self._weight*Cx)

        for i in range(self.iterations):
            self._urhs.assign(self._u)
            self.solver.solve()

        return tensor.assign(self._weight*self._u)


def diffusion_form(u, v, kappa: Constant | Function,
                   formulation: AutoregressiveCovariance.DiffusionForm):
    """
    Convenience function for common diffusion forms.

    Currently provides:

    - Standard continuous Galerkin form.

    - Interior penalty method for discontinuous spaces.


    Parameters
    ----------
    u :
        :func:`~firedrake.ufl_expr.TrialFunction` to construct diffusion form with.
    v :
        :func:`~firedrake.ufl_expr.TestFunction` to construct diffusion form with.
    kappa :
        The diffusion coefficient.
    formulation :
        The type of diffusion form.

    Returns
    -------
    ufl.Form :
        The diffusion form over u and v.

    Raises
    ------
    ValueError
        Unrecognised formulation.

    See Also
    --------
    AutoregressiveCovariance.DiffusionForm
    """
    if formulation == AutoregressiveCovariance.DiffusionForm.CG:
        return inner(u, v)*dx + inner(kappa*grad(u), grad(v))*dx

    elif formulation == AutoregressiveCovariance.DiffusionForm.IP:
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
        raise ValueError("Unknown AutoregressiveCovariance.DiffusionForm {formulation}")


class CovarianceMatCtx:
    r"""
    A python Mat context for a covariance operator.
    Can apply either the action or inverse of the covariance.

    .. math::

        B: V^{*} \to V

        B^{-1}: V \to V^{*}

    Parameters
    ----------
    covariance :
        The covariance operator.
    operation : CovarianceMatCtx.Operation
        Whether the matrix applies the action or inverse of the covariance operator.
        Defaults to ``Operation.ACTION``.

    See Also
    --------
    CovarianceOperatorBase
    AutoregressiveCovariance
    CovarianceMat
    CovariancePC
    """
    class Operation(Enum):
        """
        The covariance operation to apply with this Mat.

        See Also
        --------
        CovarianceOperatorBase
        AutoregressiveCovariance
        CovarianceMat
        CovariancePC
        """
        ACTION = 'action'
        INVERSE = 'inverse'

    def __init__(self, covariance: CovarianceOperatorBase, operation=None):
        operation = self.Operation(operation or self.Operation.ACTION)

        V = covariance.function_space()
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
        """Apply the action or inverse of the covariance operator
        to x, putting the result in y.

        y is not guaranteed to be zero on entry.

        Parameters
        ----------
        A : PETSc.Mat
            The PETSc matrix that self is the python context of.
        x : PETSc.Vec
            The vector acted on by the matrix.
        y : PETSc.Vec
            The result of the matrix action.
        """
        with self.x.dat.vec_wo as v:
            x.copy(result=v)

        self._mult_op(self.x, tensor=self.y)

        with self.y.dat.vec_ro as v:
            v.copy(result=y)

    def view(self, mat, viewer=None):
        """View object. Method usually called by PETSc with e.g. -ksp_view.
        """
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return

        viewer.printfASCII(f"  firedrake covariance operator matrix: {type(self).__name__}\n")
        viewer.printfASCII(f"  Applying the {str(self.operation)} of the covariance operator {type(self.covariance).__name__}\n")

        if (type(self.covariance) is AutoregressiveCovariance) and (self.covariance.iterations > 0):
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


def CovarianceMat(covariance: CovarianceOperatorBase,
                  operation: CovarianceMatCtx.Operation | None = None):
    r"""
    A Mat for a covariance operator.
    Can apply either the action or inverse of the covariance.
    This is a convenience function to create a PETSc.Mat with a :class:`.CovarianceMatCtx` Python context.

    .. math::

        B: V^{*} \to V

        B^{-1}: V \to V^{*}

    Parameters
    ----------
    covariance :
        The covariance operator.
    operation : CovarianceMatCtx.Operation
        Whether the matrix applies the action or inverse of the covariance operator.

    Returns
    -------
    PETSc.Mat :
        The python type Mat with a :class:`CovarianceMatCtx` context.

    See Also
    --------
    CovarianceOperatorBase
    AutoregressiveCovariance
    CovarianceMatCtx
    CovarianceMatCtx.Operation
    CovariancePC
    """
    ctx = CovarianceMatCtx(covariance, operation=operation)

    sizes = covariance.function_space().dof_dset.layout_vec.getSizes()

    mat = PETSc.Mat().createPython(
        (sizes, sizes), ctx, comm=ctx.comm)
    mat.setUp()
    mat.assemble()
    return mat


CovarianceMat.Operation = CovarianceMatCtx.Operation


class CovariancePC(petsctools.PCBase):
    r"""
    A python PC context for a covariance operator.
    Will apply either the action or inverse of the covariance,
    whichever is the opposite of the Mat operator.

    .. math::

        B: V^{*} \to V

        B^{-1}: V \to V^{*}

    Available options:

    * ``-pc_use_amat`` - use Amat to apply the covariance operator.

    See Also
    --------
    CovarianceOperatorBase
    AutoregressiveCovariance
    CovarianceMatCtx
    CovarianceMat
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

        V = covariance.function_space()
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
        """Apply the action or inverse of the covariance operator
        to x, putting the result in y.

        y is not guaranteed to be zero on entry.

        Parameters
        ----------
        pc : PETSc.PC
            The PETSc preconditioner that self is the python context of.
        x : PETSc.Vec
            The vector acted on by the pc.
        y : PETSc.Vec
            The result of the pc application.
        """
        with self.x.dat.vec_wo as xvec:
            x.copy(result=xvec)

        self._apply_op(self.x, tensor=self.y)

        with self.y.dat.vec_ro as yvec:
            yvec.copy(result=y)

    def update(self, pc):
        pass

    def view(self, pc, viewer=None):
        """View object. Method usually called by PETSc with e.g. -ksp_view.
        """
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return

        viewer.printfASCII(f"  firedrake covariance operator preconditioner: {type(self).__name__}\n")
        viewer.printfASCII(f"  Applying the {str(self.operation)} of the covariance operator {type(self.covariance).__name__}\n")

        if self.use_amat:
            viewer.printfASCII("  using Amat matrix\n")

        if (type(self.covariance) is AutoregressiveCovariance) and (self.covariance.iterations > 0):
            if self.operation == CovarianceMatCtx.Operation.ACTION:
                viewer.printfASCII("  Information for the diffusion solver for applying the action:\n")
                self.covariance.solver.snes.ksp.view(viewer)
            elif self.operation == CovarianceMatCtx.Operation.INVERSE:
                viewer.printfASCII("  Information for the mass solver for applying the inverse:\n")
                self.covariance.mass_solver.snes.ksp.view(viewer)
