from firedrake import *
from slepc4py import SLEPc
import numpy as np




class LinearEigenproblem:
    def __init__(self, mesh, a):
        self.mesh = mesh
        self.a = a  # LHS 
        self.V = FunctionSpace(mesh, "CG", 1)
        self.bc = DirichletBC(self.V, 0.0, "on_boundary")
        
    def get_problem(self):
        # u = TrialFunction(self.V)
        # v = TestFunction(self.V)
        # a = (inner(grad(u), grad(v)) + u*v )* dx
        L = Constant(1) * inner(u, v) * dx
        A = assemble(self.a).M.handle
        b = assemble(L, bcs=self.bc).M.handle 
        return A, b

class LinearEigensolver:
    def __init__(self):
        self.es = SLEPc.EPS().create(comm=COMM_WORLD)

    def set_operators(self, A, b):
        self.es.setOperators(A, b)

    def set_num_eigenvals(self, n):
        '''Sets the number of eigenvalues'''
        self.es.setDimensions(n)
    
    def set_from_options(self, opts=None):
        # setting options, make this more general
        opts = PETSc.Options()
        opts.setValue("eps_gen_non_hermitian", None)
        opts.setValue("st_pc_factor_shift_type", "NONZERO")
        opts.setValue("eps_type", "krylovschur")
        opts.setValue("eps_largest_imaginary", None)
        opts.setValue("eps_tol", 1e-10)
        self.es.setFromOptions()

    def solve(self):
        self.es.solve()
        nconv = self.es.getConverged()

        if nconv > 0:
            self.evals = np.zeros(nconv, dtype=np.complex)
            self.evecs = np.zeros(nconv, dtype=np.complex)
            for i in range(nconv):
                self.evals[i] = self.es.getEigenvalue(i)


    def get_eigenvalues(self):
        '''Returns the eigenvalues of the problem'''
        return self.evals

    def get_eigenvectors(self):
        '''Returns the eigenvectors of the problem'''

        vr, wr = self.A.getVecs()
        vi, wi = self.A.getVecs()
        
        return self.evecs

    def print_err(self):
        '''Prints errors of each eigenvector'''
        pass

    def plot(self):
        '''go from tutorial'''
        pass


m = UnitSquareMesh(10, 10)
a = (inner(grad(u), grad(v)) + u*v )* dx
Lin_EP = LinearEigenproblem(m, a, k=5)
Lin_ES = LinearEigensolver()

A, b = Lin_EP.get_problem()
Lin_ES.set_operators(A, b)
Lin_ES.set_num_eigenvals(5)
#Lin_ES.set_from_options()
Lin_ES.solve()
evals = Lin_ES.get_eigenvalues()
print(evals)
