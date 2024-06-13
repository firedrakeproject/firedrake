"""Specify and solve finite element eigenproblems."""
from firedrake.assemble import assemble
from firedrake.bcs import DirichletBC
from firedrake.function import Function
from firedrake.functionspace import RestrictedFunctionSpace
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake import utils
from firedrake.petsc import OptionsManager, flatten_parameters
from firedrake.exceptions import ConvergenceError
from ufl import replace, inner, dx
try:
    from slepc4py import SLEPc
except ImportError:
    SLEPc = None
__all__ = ["LinearEigenproblem",
           "LinearEigensolver"]


class LinearEigenproblem():
    """Generalised linear eigenvalue problem.

    The problem has the form, find `u`, `λ` such that::

        A(u, v) = λM(u, v)    ∀ v ∈ V

    Parameters
    ----------
    A : ufl.Form
        The bilinear form A(u, v).
    M : ufl.Form
        The mass form M(u, v),  defaults to inner(u, v) * dx.
    bcs : DirichletBC or list of DirichletBC
        The boundary conditions.
    bc_shift: float
        The value to shift the boundary condition eigenvalues by. This value
        will be ignored if restrict==True.
    restrict: bool
        If True, replace the function spaces of u and v with their restricted
        version. The output space remains unchanged.

    Notes
    -----
    If restrict==True, the arguments of A and M will be replaced, such that
    their function space is replaced by the equivalent RestrictedFunctionSpace
    class. This avoids the computation of eigenvalues associated with the
    Dirichlet boundary conditions. This in turn prevents convergence failures,
    and allows only the non-boundary eigenvalues to be returned. The
    eigenvectors will be in the original, non-restricted space.

    If restrict==False and Dirichlet boundary conditions are supplied, then
    these conditions will result in the eigenproblem having a nullspace spanned
    by the basis functions with support on the boundary. To facilitate
    solution, this is shifted by the specified amount. It is the user's
    responsibility to ensure that the shift is not close to an actual
    eigenvalue of the system.
    """
    def __init__(self, A, M=None, bcs=None, bc_shift=0.0, restrict=True):
        if not SLEPc:
            raise ImportError(
                "Unable to import SLEPc, eigenvalue computation not possible "
                "(try firedrake-update --slepc)"
            )

        args = A.arguments()
        v, u = args
        self.output_space = u.function_space()
        self.bc_shift = bc_shift
        self.restrict = restrict

        if not M:
            M = inner(u, v) * dx

        if restrict and bcs:  # assumed u and v are in the same space here
            V_res = RestrictedFunctionSpace(self.output_space, boundary_set=set([bc.sub_domain for bc in bcs]))
            u_res = TrialFunction(V_res)
            v_res = TestFunction(V_res)
            self.M = replace(M, {u: u_res, v: v_res})
            self.A = replace(A, {u: u_res, v: v_res})
            self.bcs = [DirichletBC(V_res, bc.function_arg, bc.sub_domain) for bc in bcs]
            self.restricted_space = V_res
        else:
            self.A = A  # LHS
            self.M = M
            self.bcs = bcs

    def dirichlet_bcs(self):
        """Return an iterator over the Dirichlet boundary conditions."""
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    @utils.cached_property
    def dm(self):
        r"""Return the dm associated with the output space."""
        if self.restrict:
            return self.restricted_space.dm
        else:
            return self.output_space.dm


class LinearEigensolver(OptionsManager):
    r"""Solve a LinearEigenproblem.

    Parameters
    ----------
    problem : LinearEigenproblem
        The eigenproblem to solve.
    n_evals : int
        The number of eigenvalues to compute.
    options_prefix : str
        The options prefix to use for the eigensolver.
    solver_parameters : dict
        PETSc options for the eigenvalue problem.
    ncv : int
        Maximum dimension of the subspace to be used by the solver. See
        `SLEPc.EPS.setDimensions`.
    mpd : int
        Maximum dimension allowed for the projected problem. See
        `SLEPc.EPS.setDimensions`.

    Notes
    -----

    Users will typically wish to set solver parameters specifying the symmetry
    of the eigenproblem and which eigenvalues to search for first.

    The former is set using the options available for `EPSSetProblemType
    <https://slepc.upv.es/documentation/current/docs/manualpages/EPS/EPSSetProblemType.html>`__.

    For example if the bilinear form is symmetric (Hermitian in complex mode),
    one would add this entry to `solver_options`::

        "eps_gen_hermitian": None

    As always when specifying PETSc options, `None` indicates that the option
    in question is a flag and hence doesn't take an argument.

    The eigenvalues to search for first are specified using the options for
    `EPSSetWhichEigenPairs <https://slepc.upv.es/documentation/current/docs/manualpages/EPS/EPSSetWhichEigenpairs.html>`__.

    For example, to look for the eigenvalues with largest real part, one
    would add this entry to `solver_options`::

        "eps_largest_real": None
    """

    DEFAULT_EPS_PARAMETERS = {"eps_type": "krylovschur",
                              "eps_tol": 1e-10,
                              "eps_target": 0.0}

    def __init__(self, problem, n_evals, *, options_prefix=None,
                 solver_parameters=None, ncv=None, mpd=None):

        self.es = SLEPc.EPS().create(comm=problem.dm.comm)
        self._problem = problem
        self.n_evals = n_evals
        self.ncv = ncv
        self.mpd = mpd
        solver_parameters = flatten_parameters(solver_parameters or {})
        for key in self.DEFAULT_EPS_PARAMETERS:
            value = self.DEFAULT_EPS_PARAMETERS[key]
            solver_parameters.setdefault(key, value)
        if self._problem.bcs:
            solver_parameters.setdefault("st_type", "sinvert")
        super().__init__(solver_parameters, options_prefix)
        self.set_from_options(self.es)

    def check_es_convergence(self):
        r"""Check the convergence of the eigenvalue problem."""
        r = self.es.getConvergedReason()
        try:
            reason = SLEPc.EPS.ConvergedReasons[r]
        except KeyError:
            reason = ("unknown reason (petsc4py enum incomplete?), "
                      "try with -eps_converged_reason")
        if r < 0:
            raise ConvergenceError(
                r"""Eigenproblem failed to converge after %d iterations.
        Reason:
        %s""" % (self.es.getIterationNumber(), reason)
            )

    def solve(self):
        r"""Solve the eigenproblem.

        Returns
        -------
        int
            The number of Eigenvalues found.
        """
        self.A_mat = assemble(self._problem.A, bcs=self._problem.bcs).M.handle
        self.M_mat = assemble(
            self._problem.M, bcs=self._problem.bcs,
            weight=self._problem.bc_shift and 1./self._problem.bc_shift
        ).M.handle

        self.es.setDimensions(nev=self.n_evals, ncv=self.ncv, mpd=self.mpd)
        self.es.setOperators(self.A_mat, self.M_mat)
        with self.inserted_options():
            self.es.solve()
        nconv = self.es.getConverged()
        if nconv == 0:
            raise ConvergenceError("Did not converge any eigenvalues.")
        return nconv

    def eigenvalue(self, i):
        r"""Return the i-th eigenvalue of the solved problem."""
        return self.es.getEigenvalue(i)

    def eigenfunction(self, i):
        r"""Return the i-th eigenfunction of the solved problem.

        Returns
        -------
        (Function, Function)
            The real and imaginary parts of the eigenfunction.
        """
        if self._problem.restrict:
            eigenmodes_real = Function(self._problem.restricted_space)
            eigenmodes_imag = Function(self._problem.restricted_space)
        else:
            eigenmodes_real = Function(self._problem.output_space)  # fn of V
            eigenmodes_imag = Function(self._problem.output_space)
        with eigenmodes_real.dat.vec_wo as vr:
            with eigenmodes_imag.dat.vec_wo as vi:
                self.es.getEigenvector(i, vr, vi)  # gets the i-th eigenvector
        if self._problem.restrict:
            eigenmodes_real = Function(self._problem.output_space).interpolate(eigenmodes_real)
            eigenmodes_imag = Function(self._problem.output_space).interpolate(eigenmodes_imag)
        return eigenmodes_real, eigenmodes_imag  # returns Firedrake fns
