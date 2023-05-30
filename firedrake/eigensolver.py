from firedrake.assemble import assemble
from firedrake.function import Function
from firedrake import solving_utils
from firedrake import utils
from firedrake.petsc import PETSc, OptionsManager, flatten_parameters
from firedrake.logging import warning
from firedrake.exceptions import ConvergenceError
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)
__all__ = ["LinearEigenproblem",
           "LinearEigensolver",
           "NonlinearEigenproblem",
           "NonlinearEigensolver"]


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
    # DEFAULT_SNES_PARAMETERS = {"snes_type": "newtonls",
    #                            "snes_linesearch_type": "basic"}

    # # Looser default tolerance for KSP inside SNES.
    DEFAULT_EPS_PARAMETERS = solving_utils.DEFAULT_KSP_PARAMETERS.copy() 

    DEFAULT_EPS_PARAMETERS = {"eps_gen_non_hermitian": None, 
                              "st_pc_factor_shift_type": "NONZERO",
                              "eps_type": "krylovschur",
                              "eps_largest_imaginary": None,
                              "eps_tol":1e-10}  # to change

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
            reason = "unknown reason (petsc4py enum incomplete?), try with -eps_converged_reason"
        if r < 0:
            raise ConvergenceError(r"""Eigenproblem failed to converge after %d iterations.
        Reason:
        %s""" % (self.es.getIterationNumber(), reason))

    def solve(self):
        r"""Solve the eigenproblem, return the number of converged eigenvalues."""
        if self._problem.bcs is None: # Neumann BCs 
            print('Neumann BCs')
            self.A_mat = assemble(self._problem.A, bcs=self._problem.bcs).M.handle
            self.M_mat = assemble(self._problem.M, bcs=self._problem.bcs).M.handle 
        else:
            self.A_mat = assemble(self._problem.M, bcs=self._problem.bcs, weight=0.).M.handle
            self.M_mat = assemble(self._problem.A, bcs=self._problem.bcs, weight=1.).M.handle 
        # E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        # E.setProblemType(SLEPc.EPS.ProblemType.GHEP);
        self.es.setOperators(self.A_mat, self.M_mat)
        self.es.setDimensions(nev=self.n_evals,ncv=2*self.n_evals)
        with self.inserted_options():
            self.es.solve()
        nconv = self.es.getConverged()
        if nconv == 0:
            raise ConvergenceError("Did not converge any eigenvalues.")  # solving_utils
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
        return eigenmodes_real, eigenmodes_imag  # firedrake fns
class NonlinearEigenproblem():
    r"""Nonlinear eigenvalue problem A(u; v, lambda) = 0."""

    def __init__(self, A, J, M=None, bcs=None):
        r"""
        :param A: the residual form A(u, v, lambda)
        :param J: the Jacobian form J(u, du, v, lambda)
        :param M: the mass form M(u, v) (optional)
        :param bcs: the boundary conditions (optional)
        """
        self.A = A  # residual
        self.J = J  # Jacobian
        args = A.arguments()
        v, u = args[1], args[0]
        self.output_space = u.function_space()
        self.bcs = bcs

        if M:
            self.M = M
        else:
            from ufl import inner, dx
            self.M = inner(u, v) * dx

    def dirichlet_bcs(self):
        r"""Return an iterator over the Dirichlet boundary conditions in self.bcs."""
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    @utils.cached_property
    def dm(self):
        r"""Return the function space's distributed mesh associated with self.output_space (a cached property)."""
        return self.output_space.dm


class NonlinearEigensolver(OptionsManager):
    r"""Solve a :class:`NonlinearEigenproblem` with an NEP."""
    DEFAULT_SLEPC_PARAMETERS = {"nep_type": "gd",
                                "nep_gd_double_expansion": True,
                                "nep_gd_double_expansion_max": 10}

    def __init__(self, problem, n_evals, *, options_prefix=None, solver_parameters=None):
        r'''
        :param problem: :class:`NonlinearEigenproblem` to solve.
        :param n_evals: number of eigenvalues to compute.
        :param options_prefix: options prefix to use for the eigensolver.
        :param solver_parameters: PETSc options for the nonlinear eigenvalue problem.
        '''

        self.nep = SLEPc.NEP().create(comm=problem.dm.comm)
        self.nep.setProblemType(SLEPc.NEP.ProblemType.NHEP)
        self.nep.setDimensions(n_evals)
        self._problem = problem
        solver_parameters = flatten_parameters(solver_parameters or {})
        for key in self.DEFAULT_SLEPC_PARAMETERS:
            value = self.DEFAULT_SLEPC_PARAMETERS[key]
            solver_parameters.setdefault(key, value)
        super().__init__(solver_parameters, options_prefix)
        self.nep.setFromOptions()

    def solve(self):
        r"""Solve the nonlinear eigenproblem, return the number of converged eigenvalues."""
        A_mat = assemble(self._problem.A, bcs=self._problem.bcs, weight=1.).M.handle
        J_mat = assemble(self._problem.J, bcs=self._problem.bcs, weight=1.).M.handle
        self.nep.setOperators(A_mat, J_mat)
        with self.inserted_options():
            self.nep.solve()
        nconv = self.nep.getConverged()
        if nconv == 0:
            raise ConvergenceError("Did not converge any eigenvalues.")  # solving_utils
        return nconv


