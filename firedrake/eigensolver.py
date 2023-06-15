from firedrake.assemble import assemble
from firedrake.function import Function
from firedrake import utils
from firedrake.petsc import OptionsManager, flatten_parameters
from firedrake.exceptions import ConvergenceError
try:
    from slepc4py import SLEPc
except ImportError:
    raise ImportError(
        "Unable to import SLEPc, eigenvalue computation not possible "
        "(try firedrake-update --slepc)"
    )
__all__ = ["LinearEigenproblem",
           "LinearEigensolver"]


class LinearEigenproblem():
    r"""Linear eigenvalue problem A(u; v) = lambda * M (u; v)."""
    def __init__(self, A, M=None, bcs=None):
        r"""
        :param A: the bilinear form A(u, v)
        :param M: the mass form M(u, v) (optional)
        :param bcs: the boundary conditions (optional)
        """
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

    def dirichlet_bcs(self):  # cargo cult
        r"""Return an iterator over the Dirichlet boundary conditions in self.bcs."""
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    @utils.cached_property
    def dm(self):  # cargo cult
        r"""Return the function space's distributed mesh associated with self.output_space (a cached property)."""
        return self.output_space.dm


class LinearEigensolver(OptionsManager):
    r"""Solve a :class:`LinearEigenproblem`."""

    DEFAULT_EPS_PARAMETERS = {"eps_gen_non_hermitian": None,
                              "st_pc_factor_shift_type": "NONZERO",
                              "eps_type": "krylovschur",
                              "eps_largest_imaginary": None,
                              "eps_tol": 1e-10}

    def __init__(self, problem, n_evals, *, options_prefix=None, solver_parameters=None):
        r'''
        :param problem: :class:`LinearEigenproblem` to solve.
        :param n_evals: number of eigenvalues to compute.
        :param options_prefix: options prefix to use for the eigensolver.
        :param solver_parameters: PETSc options for the eigenvalue problem.
        '''

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
        r"""Solve the eigenproblem, return the number of converged eigenvalues."""
        if self._problem.bcs is None:  # Neumann BCs
            self.A_mat = assemble(self._problem.A, bcs=self._problem.bcs).M.handle
            self.M_mat = assemble(self._problem.M, bcs=self._problem.bcs).M.handle
        else:  # Dirichlet BCs - in this case we are solving Mu = lambda Au, so the eigenvalue needs to be inverted
            self.A_mat = assemble(self._problem.M, bcs=self._problem.bcs, weight=0.).M.handle
            self.M_mat = assemble(self._problem.A, bcs=self._problem.bcs, weight=1.).M.handle
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
        r"""Return the `i`th eigenvalue of the solved problem."""
        return self.es.getEigenvalue(i)

    def eigenfunction(self, i):
        r"""Return the `i`th eigenfunction of the solved problem."""
        eigenmodes_real = Function(self._problem.output_space)  # fn of V
        eigenmodes_imag = Function(self._problem.output_space)
        with eigenmodes_real.dat.vec_wo as vr:
            with eigenmodes_imag.dat.vec_wo as vi:
                self.es.getEigenvector(i, vr, vi)  # gets the i-th eigenvector
        return eigenmodes_real, eigenmodes_imag  # returns Firedrake fns
