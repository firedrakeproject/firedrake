from firedrake import inner, dx, assemble
from firedrake import solving_utils
from firedrake import utils
import numpy as np
from firedrake.petsc import PETSc, OptionsManager
# from firedrake.logging import warning
# try:
#     from slepc4py import SLEPc
# except ImportError:
#     import sys
#     warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
#     sys.exit(0)

class LinearEigenproblem:
    def __init__(self, A, M=None, bcs=None):
        self.A = A  # LHS 
        args = A.arguments()
        v, u = args[0], args[1]
        if M:
            self.M = M
        else:
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
        self.es = SLEPc.EPS().create(comm=problem.dm().comm)
        self._problem = problem 
        self.n_evals = n_evals
        solver_parameters = flatten_parameters(solver_parameters or {})
        for k, v in self.DEFAULT_EPS_PARAMETERS:
            solver_parameters.setdefault(k, v)
        super().__init__(solver_parameters, options_prefix)
        self.es.set_from_options()

    def solve(self):
        '''Solves the eigenproblem, returns the number of converged eigenvalues'''
        self.A_mat = assemble(self._problem.A).M.handle
        self.M_mat = assemble(self._problem.M, bcs=self._problem.bcs).M.handle 
        self.es.setOperators(self.A_mat, self.M_mat)
        self.es.setDimensions(self.n_evals)
        with self.inserted_options():
            self.es.solve()
        nconv = self.es.getConverged()
        
        if nconv == 0:
            import sys
            warning("Did not converge any eigenvalues")
            sys.exit(0)
        else:
            self.evals = np.zeros(nconv, dtype=complex)
            vr, vi = self.A_mat.getVecs()
            for i in range(nconv):
                #self.evals[i] = self.es.getEigenvalue(i)
                self.evals[i] = self.es.getEigenpair(0, vr, vi)
        return nconv

    def eigenvalues(self):
        '''Return the eigenvalues of the problem'''
        return self.evals

    def eigenfunctions(self, i):
        '''Return the eigenfunctions of the problem'''
        eigenmodes_real = Function(self._problem.OutputSpace)
        eigenmodes_imag = Function(self._problem.OutputSpace)
        vr, vi = eigenmodes_real.dat.vec_wo,  eigenmodes_imag.dat.vec_wo
        lam = self.es.getEigenpair(i, vr, vi)
        print(eigenmodes_real.dat.data_ro[:])
        return lam, eigenmodes_real, eigenmodes_imag  # Firedrake Vectors
    

    # def errors(self, nconv):
    #     '''Returns array of errors of each eigenvector'''
    #     if nconv > 0:
    #         errors = np.zeros(nconv)
    #         #vr, vi = self.petsc_A.getVecs()
    #         for i in range(nconv):
    #             #k = self.es.getEigenpair(i, vr, vi)
    #             errors[i] = self.es.computeError(i)
    #         return errors
    #     else:
    #         raise ValueError('nconv must be positive')



# # change 
# def helmholtz_test():
#     mesh = UnitSquareMesh(10, 10)
#     V = FunctionSpace(mesh, "CG", 1)
#     u = TrialFunction(V)
#     v = TestFunction(V)
#     A = (inner(grad(u), grad(v)) + u*v )* dx
#     bcs = DirichletBC(V, 0.0, "on_boundary")
#     Lin_EP = LinearEigenproblem(A, bcs=bcs)
#     Lin_ES = LinearEigensolver(Lin_EP, index=1)
#     Lin_ES.set_num_eigenvals(5)
#     nconv = Lin_ES.solve()
#     err = Lin_ES.errors(nconv)
#     print(err)

# def tutorial():
#     mesh = UnitSquareMesh(10, 10)
#     Vcg  = FunctionSpace(mesh,'CG',3)
#     bc = DirichletBC(Vcg, 0.0, "on_boundary")
#     beta = Constant('1.0')
#     F    = Constant('1.0')
#     phi, psi = TestFunction(Vcg), TrialFunction(Vcg)
#     a =  beta*phi*psi.dx(0)*dx
#     m = -inner(grad(psi), grad(phi))*dx - F*psi*phi*dx
#     eigenprob = LinearEigenproblem(a, m) # try with no m

#     eigensolver = LinearEigensolver(eigenprob, 1)
#     opts_dict = {"eps_gen_non_hermitian": None, 
#             "st_pc_factor_shift_type": "NONZERO",
#             "eps_type": "krylovschur",
#             "eps_largest_imaginary": None,
#             "eps_tol":1e-10}
#     eigensolver.set_from_options(opts_dict)
#     eigensolver.set_num_eigenvals(1)
#     eigensolver.solve()
#     evals = eigensolver.eigenvalues()
#     evecs = eigensolver.eigenfunctions(2)
#     print(evals)
