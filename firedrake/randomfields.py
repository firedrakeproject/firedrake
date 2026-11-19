import numpy as np

from firedrake.assemble import assemble
from firedrake.constant import Constant
from firedrake.dmhooks import add_hooks
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.linear_solver import LinearSolver
from firedrake.logging import warning
from firedrake.petsc import get_petsc_variables
from firedrake.randomfunctiongen import PCG64, RandomGenerator
from firedrake.tsfc_interface import compile_form
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.utility_meshes import *
from firedrake.variational_solver import LinearVariationalProblem, LinearVariationalSolver

from loopy import generate_code_v2

from inspect import signature

from math import ceil, gamma

from ufl import as_vector, dx, inner, grad, SpatialCoordinate
from finat.ufl import BrokenElement, MixedElement

from pyop2 import op2
from pyop2.mpi import COMM_WORLD

_default_pcg = PCG64()


def WhiteNoise(V, rng=None, scale=1.0):
    r""" Generates a white noise sample

    :arg V: The :class: `firedrake.FunctionSpace` to construct a
        white noise sample on
    :arg rng: Initialised random number generator to use for obtaining
        random numbers
    :arg scale: Multiplicative scale factor for the white noise

    Returns a :firedrake.Function: with
    b ~ Normal(0, M)
    where b is the dat.data of the function returned
    and M is the mass matrix.
    For details see Paper [Croci et al 2018]:
    https://arxiv.org/abs/1803.04857v2
    """
    # If no random number generator provided make a new one
    if rng is None:
        pcg = _default_pcg
        rng = RandomGenerator(pcg)

    # Create broken space for independent samples
    mesh = V.mesh()
    broken_elements = MixedElement([BrokenElement(Vi.ufl_element()) for Vi in V])
    Vbrok = FunctionSpace(mesh, broken_elements)
    iid_normal = rng.normal(Vbrok, 0.0, 1.0)
    wnoise = Function(V)

    # Create mass expression, assemble and extract kernel
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = inner(u, v)*dx
    mass_ker, *stuff = compile_form(mass, "mass")
    mass_code = generate_code_v2(mass_ker.kinfo.kernel.code).device_code()
    mass_code = mass_code.replace("void " + mass_ker.kinfo.kernel.name,
                                  "static void " + mass_ker.kinfo.kernel.name)

    # Add custom code for doing "Cholesky" decomp and applying to broken vector
    name = mass_ker.kinfo.kernel.name
    blocksize = mass_ker.kinfo.kernel.code[name].args[0].shape[0]

    cholesky_code = f"""
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
    double scale = {scale};
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
    # Get the BLAS and LAPACK compiler parameters to compile the kernel
    if COMM_WORLD.rank == 0:
        petsc_variables = get_petsc_variables()
        BLASLAPACK_LIB = petsc_variables.get("BLASLAPACK_LIB", "")
        BLASLAPACK_LIB = COMM_WORLD.bcast(BLASLAPACK_LIB, root=0)
        BLASLAPACK_INCLUDE = petsc_variables.get("BLASLAPACK_INCLUDE", "")
        BLASLAPACK_INCLUDE = COMM_WORLD.bcast(BLASLAPACK_INCLUDE, root=0)
    else:
        BLASLAPACK_LIB = COMM_WORLD.bcast(None, root=0)
        BLASLAPACK_INCLUDE = COMM_WORLD.bcast(None, root=0)

    cholesky_kernel = op2.Kernel(cholesky_code,
                                 "apply_cholesky",
                                 include_dirs=BLASLAPACK_INCLUDE.split(),
                                 ldargs=BLASLAPACK_LIB.split())

    i, _ = mass_ker.indices

    z_arg = iid_normal.dat(op2.READ, Vbrok.cell_node_map())
    b_arg = wnoise.dat(op2.INC, V.cell_node_map())
    coords = mesh.coordinates

    op2.par_loop(cholesky_kernel,
                 mesh.cell_set,
                 z_arg,
                 b_arg,
                 coords.dat(op2.READ, coords.cell_node_map())
                 )

    return wnoise


def PaddedMesh(constructor, *args, pad=0.2, **kwargs):
    r"""A mesh as specified and an additional mesh with padding on all
    of its boundaries.
    :arg constructor: The utility function used to construct a a mesh
    :arg args: Arguments used to construct the mesh
    :arg pad: Padding to add to all boundaries
    :arg kwargs: Additional keyword arguments used for constructing the mesh
    """
    # Enumerate different types of supported meshes
    supported_1D_constructors = [IntervalMesh, UnitIntervalMesh]
    supported_2D_constructors = [RectangleMesh, SquareMesh, UnitSquareMesh]
    supported_2D_immersed_constructors = [CylinderMesh]
    supported_3D_constructors = [BoxMesh, CubeMesh, UnitCubeMesh]
    supported_periodic_constructors = [PeriodicRectangleMesh, PeriodicSquareMesh, PeriodicUnitSquareMesh]
    xyz = ['x', 'y', 'z']
    # Note that all of:
    # PeriodicIntervalMesh, PeriodicUnitIntervalMesh, CircleManifoldMesh,
    # IcosahedralSphereMesh, UnitIcosahedralSphereMesh,
    # OctahedralSphereMesh, UnitOctahedralSphereMesh,
    # CubedSphereMesh, UnitCubedSphereMesh,
    # TorusMesh,
    # PeriodicBoxMesh, PeriodicUnitCubeMesh
    # have no boundary, so don't need an auxiliary mesh
    supported_constructors = supported_1D_constructors \
        + supported_2D_constructors \
        + supported_2D_immersed_constructors \
        + supported_3D_constructors \
        + supported_periodic_constructors
    assert constructor in supported_constructors

    def _pad(N, L, pad):
        r""" Defines the number of additional cells, additional length
        and coordinate shift in one direction
        :arg N: Number of cells
        :arg L: Length
        :arg pad: Desired amount of padding
        """
        h = L/N
        extra = ceil(pad/h)
        aN = N + 2*extra
        aL = h*aN
        shift = h*extra
        return aN, aL, shift

    # Remove the pad keyword and construct the original mesh
    # ~ pad = kwargs.pop('pad')
    mesh = constructor(*args, **kwargs)

    # Gather the size and number of cells arguments into a dict
    sig = signature(constructor)
    known_args = [
        'ncells', 'nx', 'ny', 'nz', 'nr', 'nl',
        'length_or_left', 'right', 'L', 'Lx', 'Ly', 'Lz', 'depth'
    ]
    constructor_keys = [var for var in sig.parameters if (var in known_args)]
    arg_dict = {k: v for k, v in zip(constructor_keys, args)}
    # To avoid duplicate keys move any known args out of the kwargs dict
    for argument in known_args:
        try:
            arg_dict[argument] = kwargs.pop(argument)
        except KeyError:
            pass

    # Construct the padded mesh
    if constructor in supported_1D_constructors:
        # aux_constructor = IntervalMesh
        aN, aL, shift = _pad(arg_dict.get('ncells'),
                             abs(arg_dict.get('right', 0) - arg_dict.get('length_or_left', 1)),
                             pad)
        aux_mesh = IntervalMesh(aN, aL, **kwargs)
        aux_mesh.coordinates.dat.data[:] = aux_mesh.coordinates.dat.data - shift
    elif constructor in supported_2D_immersed_constructors:
        # aux_constructor = CylinderMesh
        direction = xyz.index(kwargs.get('longitudinal_direction', 'z'))
        shift = [0, 0, 0]
        aN, aL, s = _pad(arg_dict.get('nl'),
                         arg_dict.get('depth', 1),
                         pad)

        aux_mesh = CylinderMesh(arg_dict.get('nr'), aN, depth=aL, **kwargs)
        shift[direction] = s
        aux_mesh.coordinates.dat.data[:, :] = aux_mesh.coordinates.dat.data - shift
    elif constructor in supported_periodic_constructors:
        aux_constructor = PeriodicRectangleMesh
        if kwargs.get('direction') == 'both':
            # Has no boundary!
            return mesh, None
        else:
            c = kwargs.get('direction')
            direction = (xyz.index(c) + 1) % mesh.topological_dimension()
        aN = [arg_dict.get('nx'), arg_dict.get('ny')]
        aL = [
            arg_dict.get('Lx', arg_dict.get('L', 1)),
            arg_dict.get('Ly', arg_dict.get('L', 1))
        ]
        shift = [0, 0]
        n, l, s = _pad(arg_dict.get('n' + c),
                       arg_dict.get('L' + c, arg_dict.get('L', 1)),
                       pad)
        aN[direction] = n
        aL[direction] = l
        shift[direction] = s
        aux_mesh = PeriodicRectangleMesh(*aN, *aL, **kwargs)
        aux_mesh.coordinates.dat.data[:, :] = aux_mesh.coordinates.dat.data - shift
    else:
        # We can handle rectangles and boxes together
        if mesh.topological_dimension() == 2:
            aux_constructor = RectangleMesh
        elif mesh.topological_dimension() == 3:
            aux_constructor = BoxMesh
        aN = []
        aL = []
        shift = []
        for _, c in zip(range(mesh.topological_dimension()), xyz):
            n, l, s = _pad(arg_dict.get('n' + c),
                           arg_dict.get('L' + c, arg_dict.get('L', 1)),
                           pad)
            aN.append(n)
            aL.append(l)
            shift.append(s)
        aux_mesh = aux_constructor(*aN, *aL, **kwargs)
        aux_mesh.coordinates.dat.data[:, :] = aux_mesh.coordinates.dat.data - shift

    return mesh, aux_mesh


def TrimFunction(f, mesh):
    r"""Really don't like using this function, but can't think of better method
    Cuts a function to the provided mesh, doesn't work in parallel
    """
    V = f.ufl_function_space()

    W = FunctionSpace(mesh, V.ufl_element())
    g = Function(W)

    coordspace = VectorFunctionSpace(mesh, V.ufl_element())
    ho_coords = Function(coordspace)
    ho_coords.interpolate(as_vector(SpatialCoordinate(mesh)))

    g.dat.data[:] = f.at(ho_coords.dat.data, tolerance=1e-8)
    return g


class GaussianRF:
    r"""The distribution is a spatially varying Gaussian Random Field
    with Matern covariance, GRF(mu, C):
    mu is the mean and C is the covariance kernel:
                       sigma**2
    C(x, y) = ------------------------- * (kappa * r)**nu * K_nu(kappa r)
               2**(nu - 1) * Gamma(nu)
    where
    Gamma is the gamma function,
    K_nu is the modified Bessel function of the second kind
    r = ||x-y||_2,
             sqrt(8 * nu)
    kappa = --------------,
                lambda
    sigma is the variance,
    nu is the desired smoothness,
    lambda is the correlation length

    A realisation of a GRF is generated by calling the sample method of this class
    """
    def __init__(self, V, mu=0, sigma=1, smoothness=1, correlation_length=0.2,
                 rng=None, solver_parameters=None, V_aux=None):
        r"""Creates a spatially varying Gaussian Random Field with Matern covariance
        :arg V: The :class: `firedrake.FunctionSpace` to construct a
            white noise sample on
        :arg mu: The mean of the distribution
        :arg sigma: The standard deviation of the distribution
        :arg smoothness: The smoothness of the functions from the distribution,
            sampled functions will have *at least* the smoothness specified here
        :arg correlation_length: The length scale over which the sampled
        function will vary
        :arg rng: Initialised random number generator to use for obtaining
            random numbers, used for seeding the distribution
        :arg solver_parameters: Override the solver parameters used in constructing
            the random field. These default to {'ksp_type': 'cg', 'pc_type': 'gamg'}
        """
        # Check for sensible arguments
        assert sigma > 0
        assert smoothness > 0
        assert correlation_length > 0

        self.mesh = V.mesh()
        if self.mesh.coordinates.exterior_facet_node_map().values.size != 0 and V_aux is None:
            warning(
                r"""The mesh that the provided function space is defined on
has a boundary, but no auxiliary function space has been provided. To
prevent boundary pollution effects you should provide a function space V_aux
defined on a mesh that is at least the correlation length larger on all
boundary edges to avoid boundary pollution effects. The utility function
PaddedMesh is provided to gennerate such meshes for some of Firedrake's
utility meshes.
                """)
        self.trueV = V
        self.V = V_aux if V_aux is not None else V
        self.mu = mu
        self.sigma = sigma

        # Set symbols to match
        self.dim = V.mesh().topological_dimension()
        self.nu = smoothness
        self.lambd = correlation_length
        self.k = ceil((self.nu + self.dim/2)/2)

        # Calculate additional parameters
        self.kappa = np.sqrt(8*self.nu)/self.lambd
        sigma_hat2 = gamma(self.nu)*self.nu**(self.dim/2)
        sigma_hat2 /= gamma(self.nu + self.dim/2)
        sigma_hat2 *= (2/np.pi)**(self.dim/2)
        sigma_hat2 *= self.lambd**(-self.dim)
        self.eta = self.sigma/np.sqrt(sigma_hat2)

        # Setup RNG if provided
        self.rng = rng

        # Setup modified Helmholtz problem
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.wnoise = Function(self.V)
        a = (inner(u, v) + Constant(1/(self.kappa**2))*inner(grad(u), grad(v)))*dx
        self.A = assemble(a)

        # Solve problem once
        self.u_h = Function(self.V)
        if solver_parameters is None:
            self.solver_param = {'ksp_type': 'cg', 'pc_type': 'gamg'}
        else:
            self.solver_param = solver_parameters

        self.base_solver = LinearSolver(
            self.A,
            solver_parameters=self.solver_param
        )
        # We need the appctx if we want to perform geometric multigrid
        lvproblem = LinearVariationalProblem(a, v*dx, Function(self.V))
        lvsolver = LinearVariationalSolver(lvproblem)
        self.lvs_ctx = lvsolver._ctx

        # For smoother solutions we must iterate this solve
        if self.k > 1:
            self.u_j = Function(self.V)
            l_j = inner(self.u_j, v)*dx
            problem = LinearVariationalProblem(a, l_j, self.u_h)
            self.solver = LinearVariationalSolver(
                problem,
                solver_parameters=self.solver_param
            )

    def get_parameters(self):
        r""" Returns a dict of parameters used to specify the distribution
        """
        rf_parameters = {'mu': self.mu,
                         'sigma': self.sigma,
                         'smoothness': self.nu,
                         'correlation_length': self.lambd,
                         'rng': self.rng
                         }
        return rf_parameters

    def sample(self, rng=None):
        r"""Returns a realisation of a GRF with parameters specified
        :arg rng: Initialised random number generator to use for obtaining
            random numbers, used for seeding the distribution, will override
            the generator specified at initialisation for the given sample.
        """
        # If no random number generator provided use default
        if rng is None:
            if self.rng is None:
                pcg = _default_pcg
                rng = RandomGenerator(pcg)
            else:
                rng = self.rng

        # Generate a new white noise sample
        b = WhiteNoise(self.V, rng, scale=self.eta)

        # Solve adding an appctx from the equivalent linear variational problem
        ksp = self.base_solver.ksp
        dm = ksp.getDM()
        with add_hooks(dm, ksp, appctx=self.lvs_ctx, save=False):
            self.base_solver.solve(self.u_h, b)

        # Iterate solve until required smoothness achieved
        if self.k > 1:
            self.u_j.assign(self.u_h)
            for _ in range(self.k - 1):
                self.solver.solve()
                self.u_j.assign(self.u_h)

        # Shift the function by the mean and return
        self.u_h.dat.data[:] = self.u_h.dat.data + self.mu

        # Remove boundary
        if self.V is not self.trueV:
            sample = TrimFunction(self.u_h, self.mesh)
        else:
            sample = self.u_h
        return sample


class LogGaussianRF(GaussianRF):
    r"""The distribution is a spatially varying Log Gaussian Random Field
    with Matern covariance LGRF(mu, C):
    mu is the mean and C is the covariance kernel of the
    _logarithm_ of the distribution:
                       sigma**2
    C(x, y) = ------------------------- * (kappa * r)**nu * K_nu(kappa r)
               2**(nu - 1) * Gamma(nu)
    where
    Gamma is the gamma function,
    K_nu is the modified Bessel function of the second kind
    r = ||x-y||_2,
             sqrt(8 * nu)
    kappa = --------------,
                lambda
    sigma is the variance,
    nu is the desired smoothness,
    lambda is the correlation length

    A realisation of a LGRF is generated by calling the sample method of this class
    """
    def __init__(self, V, mu=0, sigma=1, smoothness=1, correlation_length=0.2,
                 rng=None, solver_parameters=None, mean=None, std_dev=None, V_aux=None):
        r'''Creates a spatially varying Log Gaussian Random Field with Matern covariance
        :arg V: The :class: `firedrake.FunctionSpace` to construct a
            white noise sample on
        :arg mu: The mean of the _logarithm_ of the distribution
        :arg sigma: The standard deviation of the _logarithm_ of the distribution
        :arg smoothness: The smoothness of the functions from the distribution,
            sampled functions will have *at least* the smoothness specified here
        :arg correlation_length: The length scale over which the sampled
        function will vary
        :arg rng: Initialised random number generator to use for obtaining
            random numbers, used for seeding the distribution
        :arg solver_parameters: Override the solver parameters used in constructing
            the random field. These default to {'ksp_type': 'cg', 'pc_type': 'gamg'}
        It is also possible to specify the mean and standard deviation of
        the distribution directly by setting:
        :arg mean: The mean of the distribution
        :arg std_dev: The standard deviation of the distribution
        Both of the above parameters must be set to completely specify the
        distribution. If set the parameters mu and sigma will be ignored.
        '''
        if (mean is not None) and (std_dev is not None):
            mu = np.log(mean**2/np.sqrt(std_dev**2 + mean**2))
            sigma = np.sqrt(np.log(1 + (std_dev/mean)**2))
        elif (mean is not None) and (std_dev is not None):
            raise ValueError(r"""Specify either:
    - Both `mu` and `sigma`, which the expected value and variance of the logarithm of the random field respectively.
    - Both `mean` and `std_dev`, which specify the distribution of the random field respectively.
    """)
        super().__init__(V, mu, sigma, smoothness, correlation_length,
                         rng=rng, solver_parameters=solver_parameters, V_aux=V_aux)

    def sample(self, rng=None):
        r"""Returns a realisation of a LGRF with parameters specified
        :arg rng: Initialised random number generator to use for obtaining
            random numbers, used for seeding the distribution, will override
            the generator specified at initialisation for the given sample.
        """
        sample = super().sample(rng)
        sample.dat.data[:] = np.exp(sample.dat.data)
        return sample
