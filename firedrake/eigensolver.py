from firedrake.assemble import assemble
from firedrake.function import Function
from firedrake import solving_utils
from firedrake import utils
from firedrake.petsc import PETSc, OptionsManager, flatten_parameters
from firedrake.logging import warning
import numpy as np
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)
__all__ = ["LinearEigenproblem",
           "LinearEigensolver"]


class LinearEigenproblem():
    def __init__(self, A, M=None, bcs=None):
        self.A = A  # LHS 
        args = A.arguments()
        v, u = args[0], args[1]
        if M:
            self.M = M
        else:
            from ufl import inner, dx
            self.M = inner(u, v) * dx
        self.OutputSpace = u.function_space()
        self.bcs = bcs

    def dirichlet_bcs(self):  # cargo cult
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    @utils.cached_property
    def dm(self):  # cargo cult
        return self.OutputSpace.dm

class LinearEigensolver(OptionsManager): 
    DEFAULT_SNES_PARAMETERS = {"snes_type": "newtonls",
                               "snes_linesearch_type": "basic"}

    # Looser default tolerance for KSP inside SNES.
    DEFAULT_EPS_PARAMETERS = solving_utils.DEFAULT_KSP_PARAMETERS.copy() 

    DEFAULT_EPS_PARAMETERS = {"eps_gen_non_hermitian": None, 
                              "st_pc_factor_shift_type": "NONZERO",
                              "eps_type": "krylovschur",
                              "eps_largest_imaginary": None,
                              "eps_tol":1e-10}

    def __init__(self, problem, n_evals, *, options_prefix=None, solver_parameters=None):
        '''
        param problem: LinearEigenproblem
        param index: int, index of eigenvalue/vector
        '''
        self.es = SLEPc.EPS().create(comm=problem.dm.comm)
        self._problem = problem 
        self.n_evals = n_evals
        solver_parameters = flatten_parameters(solver_parameters or {})
        for key in self.DEFAULT_EPS_PARAMETERS:
            value = self.DEFAULT_EPS_PARAMETERS[key]
            solver_parameters.setdefault(key, value)
        super().__init__(solver_parameters, options_prefix)
        self.es.setFromOptions()

    def solve(self):
        '''Solves the eigenproblem, returns the number of converged eigenvalues'''
        self.A_mat = assemble(self._problem.A, bcs=self._problem.bcs).M.handle
        self.M_mat = assemble(self._problem.M, bcs=self._problem.bcs).M.handle 
        self.es.setOperators(self.A_mat, self.M_mat)
        self.es.setDimensions(self.n_evals)
        with self.inserted_options():
            self.es.solve()
        nconv = self.es.getConverged()
        if nconv == 0:
            warning("Did not converge any eigenvalues") 
            raise Exception("Did not converge any eigenvalues")
        return nconv

    def eigenvalue(self, i): # DO THIS
        '''Return the eigenvalues of the problem'''
        return self.es.getEigenvalue(i)

    def eigenfunction(self, i):
        '''Return the ith eigenfunctions of the problem.'''
        eigenmodes_real = Function(self._problem.OutputSpace)  # fn of V
        eigenmodes_imag = Function(self._problem.OutputSpace)
        vr, vi = eigenmodes_real.dat.vec_wo, eigenmodes_imag.dat.vec_wo
        return vr, vi

