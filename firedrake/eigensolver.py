"""Specify and solve finite element eigenproblems."""
from firedrake.assemble import assemble
from firedrake.function import Function
from firedrake import utils
from firedrake.petsc import OptionsManager, flatten_parameters
from firedrake.exceptions import ConvergenceError
try:
    from slepc4py import SLEPc
except ImportError:
    SLEPc = None
__all__ = ["LinearEigenproblem",
           "LinearEigensolver"]


class LinearEigenproblem():
    """Linear eigenvalue problem

    The problem has the form::

        A(u, v) = λ * M(u, v)

    Parameters
    ----------
    A : ufl.Form
        the bilinear form A(u, v)
    M : ufl.Form
        the mass form M(u, v),  defaults to u * v * dx.
    bcs : DirichletBC or list of DirichletBC
        the boundary conditions
    bc_shift: float
        the value to shift the boundary condition eigenvalues by.

    Notes
    -----

    If Dirichlet boundary conditions are supplied then these will result in the
    eigenproblem having a nullspace spanned by the basis functions with support
    on the boundary. To facilitate solution, this is shifted by the specified
    amount. It is the user's responsibility to ensure that the shift is not
    close to an actual eigenvalue of the system.
    """
    def __init__(self, A, M=None, bcs=None, bc_shift=666.0):
        if not SLEPc:
            raise ImportError(
                "Unable to import SLEPc, eigenvalue computation not possible "
                "(try firedrake-update --slepc)"
            )

        self.A = A  # LHS
        args = A.arguments()
        v, u = args[0], args[1]
        if M:
            self.M = M
        else:
            from ufl import inner, dx
            self.M = inner(u, v) * dx
        self.output_space = u.function_space()
        self.bcs = bcs
        self.bc_shift = bc_shift

    def dirichlet_bcs(self):
        """Return an iterator over the Dirichlet boundary conditions."""
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    @utils.cached_property
    def dm(self):  # cargo cult
        r"""Return the dm associated with the output space."""
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
    """

    DEFAULT_EPS_PARAMETERS = {"eps_gen_non_hermitian": None,
                              "st_pc_factor_shift_type": "NONZERO",
                              "eps_type": "krylovschur",
                              "eps_largest_imaginary": None,
                              "eps_tol": 1e-10}

    def __init__(self, problem, n_evals, *, options_prefix=None,
                 solver_parameters=None):

        self.es = SLEPc.EPS().create(comm=problem.dm.comm)
        self._problem = problem
        self.n_evals = n_evals
        solver_parameters = flatten_parameters(solver_parameters or {})
        for key in self.DEFAULT_EPS_PARAMETERS:
            value = self.DEFAULT_EPS_PARAMETERS[key]
            solver_parameters.setdefault(key, value)
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

        self.es.setOperators(self.A_mat, self.M_mat)
        # SLEPc recommended params
        self.es.setDimensions(nev=self.n_evals, ncv=2*self.n_evals)
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
        eigenmodes_real = Function(self._problem.output_space)  # fn of V
        eigenmodes_imag = Function(self._problem.output_space)
        with eigenmodes_real.dat.vec_wo as vr:
            with eigenmodes_imag.dat.vec_wo as vi:
                self.es.getEigenvector(i, vr, vi)  # gets the i-th eigenvector
        return eigenmodes_real, eigenmodes_imag  # returns Firedrake fns
