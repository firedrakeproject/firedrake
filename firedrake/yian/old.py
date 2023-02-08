import ufl
from itertools import chain
from contextlib import ExitStack

from firedrake import dmhooks
from firedrake import slate
from firedrake import solving_utils
from firedrake import ufl_expr
from firedrake import utils
from firedrake import solving
from firedrake.petsc import PETSc, OptionsManager, flatten_parameters
from firedrake.bcs import DirichletBC
from firedrake.adjoint import NonlinearVariationalProblemMixin, NonlinearVariationalSolverMixin
from variational_solver import NonlinearVariationalSolver

# adjustable

class LinearEigenproblem():
    def __init__(self, A, u=None, M=None, bcs=None):
        self.A = A
        #self.u = u
        args = A.arguments()
        v, u = args[0], args[1]
        if M:
            self.M = M
        else:
            args = A.arguments()
            self.M = inner(u, v) * dx

        self.bcs = solving._extract_bcs(bcs)

    def dirichlet_bcs(self):  # cargo cult
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    def dm(self):  # cargo cult
        return self.u.function_space().dm



class LinearEigensolver():
    def __init__(self, problem, appctx=None):
        assert isinstance(problem, NonlinearVariationalProblem)
        self._problem = problem
        super().__init__(solver_parameters, options_prefix)


    def linear_esolve(self, problem):
        eigensolver_parameters={'esp_type': 'GMRES',
                                'esp_atol': 1e-50,
                                'esp_rtol': 1e-7,
                                'esp_divtol': 1e4,
                                'esp_max_it':10000,
                                'pc_type':'ILU'}

        opts = PETSc.Options()
        opts.setValue("eps_gen_non_hermitian", None)
        opts.setValue("st_pc_factor_shift_type", "NONZERO")
        opts.setValue("eps_type", "krylovschur")
        opts.setValue("eps_largest_imaginary", None)
        opts.setValue("eps_tol", 1e-10)

        s = Function(V)
        solve(a == L, s, bcs=[bc1, bc2], eigensolver_parameters)


    def linear_system(self):
        x = Function(V)
        A = assemble(self.F)
        b = assemble(np.zeros(self.F.shape))
        solve(A, x, b)

    def nlinear_esolve(self, problem):
        eigensolver_parameters={'snes_type': 'Newton linesearch',
                                'snes_rtol': 1e-8,
                                'snes_atol': 1e-50,
                                'snes_stol': 1e-8,
                                'snes_max_it': 50,
                                'esp_type': 'GMRES,
                                'esp_atol': 1e-50,
                                'esp_rtol': 1e-5,
                                'esp_divtol': 1e4,
                                'esp_max_it':10000,
                                'pc_type':'ILU'}

        solve(F == 0, u, bcs=[bc1, bc2], eigensolver_parameters)

