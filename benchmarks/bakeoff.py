from math import prod

from FIAT import ufc_simplex
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule as GLL_qlr
from finat.point_set import GaussLobattoLegendrePointSet as GLL_ps
from finat.quadrature import QuadratureRule, TensorProductQuadratureRule
from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI

import numpy as np

parprint = PETSc.Sys.Print

# GLL quadrature rule
def gauss_lobatto_legendre_line_rule(degree):
    fiat_rule = GLL_qlr(ufc_simplex(1), degree + 1)
    points = GLL_ps(fiat_rule.get_points())
    weights = fiat_rule.get_weights()
    return QuadratureRule(points, weights)

def make_tensor_product_rule(rule, dimension, degree):
    result = rule(degree)
    for _ in range(1, dimension):
        line_rule = rule(degree)
        result = TensorProductQuadratureRule([result, line_rule])
    return result

# Setup bakeoff problem range(1,7)
def setup_problem(bp, s, p, tets=False):
    q, r = divmod(s, 3)
    N = [2**(q + 1) if ii < r else 2**q for ii in range(3)]

    if tets:
        mesh = UnitCubeMesh(*N)
        E = 6*prod(N)
    else:
        base_mesh = UnitSquareMesh(N[0], N[1], quadrilateral=True)
        mesh = ExtrudedMesh(base_mesh, N[2])
        E = prod(N)
    parprint(f'Total elements : {E}')

    if bp % 2 == 1:
        # Scalar function space
        V = FunctionSpace(mesh, 'CG', degree=p)
    else:
        # Vector function space
        V = VectorFunctionSpace(mesh, 'CG', degree=p)
    ndofs = V.dim()
    parprint(f'Total DOFs : {ndofs}')

    u = TrialFunction(V)
    v = TestFunction(V)
    if bp in {1, 2}:
        # Mass problem
        a = inner(u, v)*dx
    elif bp in {3, 4}:
        # Primal Poisson
        a = inner(grad(u), grad(v))*dx
    elif bp in {5, 6}:
        # Primal Poisson with GLL quadrature
        GLL_rule = make_tensor_product_rule(
            gauss_lobatto_legendre_line_rule,
            dimension=3,
            degree=p
        )
        a = inner(grad(u), grad(v))*dx(rule=GLL_rule)

    f = Function(V)
    f.assign(1.0)
    L = inner(f, v)*dx

    bcs = DirichletBC(V, zero(), ("on_boundary",))

    u_h = Function(V)
    problem = LinearVariationalProblem(a, L, u_h, bcs=bcs)
    return problem

# Custom KSP monitor
class KSPTimeMonitor(object):
    def __init__(self, solver=None):
        if solver:
            solver.snes.ksp.setMonitor(self)
        self.clock = 0
        self.all_times = np.zeros(1000)

    def __call__(self, ksp, its, rnorm):
        ''' 45 KSP unpreconditioned resid norm 1.709986085748e-07 true resid norm 1.709986085748e-07 ||r(i)||/||b|| 9.649612120486e-07
        '''
        tdiff = MPI.Wtime() - self.clock
        self.all_times[its] = tdiff
        parprint(f'|   {its:2d} | {rnorm:14.12e} | {tdiff:14.12e} |')
        self.clock = MPI.Wtime()

    def __enter__(self):
        ''' Residual norms for firedrake_0_ solve.
        '''
        parprint('Starting')
        parprint('| iter | rnorm              | time(s)            |')
        parprint('|------|--------------------|--------------------|')
        #        '|    0 | 1.023109441696e-01 | 3.637913799980e-02 |'
        self.clock = MPI.Wtime()
        self.total = self.clock

    def __exit__(self,*exc):
        ''' Linear firedrake_0_ solve converged due to CONVERGED_RTOL iterations 45
        '''
        self.total = MPI.Wtime() - self.total
        parprint('Finishing')

# Solve problem
def solve_problem(problem, solver_parameters):
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    time_monitor = KSPTimeMonitor(solver)
    with time_monitor:
        solver.solve()

solver_parameters = {
    'mat_type': 'matfree',
    'ksp_type': 'cg', "ksp_rtol": 1e-6, 'ksp_max_it': 999, 'ksp_norm_type': 'unpreconditioned' ,'ksp_view': None, 'ksp_monitor_true_residual_': None, 'ksp_converged_reason': None,
    'pc_type':  'none'
}

# Jacobi options:
# pc_jacobi_type = diagonal,rowmax,rowsum
# pc_jacobi_abs
# pc_jacobi_fixdiag
alternative = {'pc_type': 'jacobi', 'pc_jacobi_type': 'diagonal'}
solver_parameters.update(alternative)

if __name__ == '__main__':
    # TODO: Output MPI info
    comm = COMM_WORLD
    parprint(f'Total ranks : {comm.size}')

    lvp = setup_problem(5, 12, 3)
    solve_problem(lvp, solver_parameters=solver_parameters)
