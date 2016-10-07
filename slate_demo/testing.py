from firedrake import *
from pyop2.profiling import timed_stage
from firedrake.slate import slate
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt


def test_hybridization_slate(degree, mesh_res=None, schur_pc_type='lu', write=False):

    # Generate UnitSquareMesh and facet normal
    if mesh_res is None:
        mesh = UnitSquareMesh(5, 5)
    elif isinstance(mesh_res, int):
        mesh = UnitSquareMesh(2 ** mesh_res, 2 ** mesh_res)
    #  mesh = UnitSquareMesh(2*mesh_res, 2*mesh_res)
    else:
        raise ValueError("Integers or None are only accepted for mesh_res.")
    n = FacetNormal(mesh)

    # Define relevant finite element spaces
    RT = FiniteElement("RT", triangle, degree + 1)
    BRT = FunctionSpace(mesh, BrokenElement(RT))
    DG = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", degree)
    W = BRT * DG

    # Define trial and test functions of the finite element spaces
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    # lambdar = TrialFunction(T)
    gammar = TestFunction(T)

    # Define finite element forms
    Mass1 = dot(sigma, tau)*dx
    Mass2 = u*v*dx
    Grad = div(tau)*u*dx
    Div = div(sigma)*v*dx
    trace = gammar('+')*dot(sigma, n)*dS
    # trace_jump = jump(tau, n=n)*lambdar('+')*dS

    # Homogeneous Dirichlet boundary conditions on all edges of the UnitSquareMesh
    bc = DirichletBC(T, Constant(0), (1, 2, 3, 4))

    # Creating Schur matrix with SLATE tensors
    A = slate.Matrix(Mass1 + Mass2 + Div - Grad)
    K = slate.Matrix(trace)
    schur = -K * A.inv * K.T

    # Creating left-hand side source function
    f = Function(DG)
    f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    L = f*v*dx

    # Left-hand side vector for the Schur-compliment system  in SLATE
    F = slate.Vector(L)
    RHS = -K * A.inv * F

    S = assemble(schur, bcs=bc)
    b = assemble(RHS)

    parameters = {}
    if schur_pc_type == 'lu':
        parameters.update({'pc_type': 'lu',
                           # 'pc_factor_mat_solver_package': 'mumps',
                           'ksp_type': 'cg'})
    elif schur_pc_type == 'ilu':
        parameters.update({'pc_type': 'ilu',
                           'ksp_type': 'cg'})
    elif schur_pc_type == 'hypre':
        parameters.update({'pc_type': 'hypre',
                           'ksp_type': 'cg'})
    elif schur_pc_type == 'gamg':
        parameters.update({'pc_type': 'gamg',
                           'ksp_type': 'cg'})
    else:
        raise NotImplementedError()


    print parameters
    # parameters={'pc_type': 'fieldsplit',
    #             'pc_fieldsplit_type': 'schur',
    #             'pc_fieldsplit_schur_fact_type': 'full'}
    # Now we solve for the Langrange multiplier
    lambda_sol = Function(T)
    start = timer()
    with timed_stage(schur_pc_type):
        solve(S, lambda_sol, b, solver_parameters=parameters)
    end = timer()
    elapsed_time = end - start

    # Construct right-hand side for reconstruction in UFL
    orig = assemble(L)
    orig -= assemble(action(trace_jump, lambda_sol))

    # Solve via back-substitution
    a = Mass1 + Mass2 + Div - Grad
    A = assemble(a, nest=False)
    # solution = Function(W)
    o_v, o_p = orig.split()
    sol_v, sol_p = solution.split()
    solve(A, solution, orig, solver_parameters={'ksp_type': 'preonly',
                                                'pc_type': 'lu'})
    sigma_h, u_h = solution.split()

    # Compare result with non-hybridized computation
    # RTc = FunctionSpace(mesh, "RT", degree + 1)
    # W2 = RTc * DG
    # sigma, u = TrialFunctions(W2)
    # tau, v = TestFunctions(W2)
    # a = (dot(sigma, tau) - u*div(tau) + u*v + v*div(sigma))*dx
    # L = f*v*dx
    # non_hybrid_sol = Function(W2)
    # solve(a == L, non_hybrid_sol, solver_parameters={'pc_type': 'fieldsplit',
    #                                                  'pc_fieldsplit_type': 'schur',
    #                                                  'ksp_type': 'cg',
    #                                                  'pc_fieldsplit_schur_fact_type': 'FULL',
    #                                                  'fieldsplit_V_ksp_type': 'cg',
    #                                                  'fieldsplit_P_ksp_type': 'cg'})
    # nhsigma, nhu = non_hybrid_sol.split()

    # # Return L2 error (should be identical w.r.t. solver tolerance)
    # uerr = sqrt(assemble((u_h - nhu)*(u_h - nhu)*dx))
    # sigerr = sqrt(assemble(dot(sigma_h - nhsigma, sigma_h - nhsigma)*dx))

    # Analytical solution
    expected = Expression("sin(x[0]*pi*2)*sin(x[1]*pi*2)")

    f.interpolate(expected)
    error = sqrt(assemble((u_h - f)*(u_h - f)*dx))
    p = Function(T)
    p.interpolate(expected)
    multiplier_error = sqrt(assemble(FacetArea(mesh)*(lambda_sol - p)('+')*(lambda_sol - p)('+')*dS))

    if write:
        # Write hybridized solutions to paraview file
        sigma_h = project(sigma_h, FunctionSpace(mesh, RT))
        # File("solution.pvd").write(sigma_h, u_h, nhsigma, nhu)
        File("solution.pvd").write(sigma_h, u_h)

    # return elapsed_time
    return multiplier_error

deg = 0
n = 3
multiplier_error = test_hybridization_slate(deg, write=True)
# uerr, sigerr, err, mult_err = test_hybridization_slate(deg, write=True)
# print "Error in scalar variable: ", uerr
# print "Error in flux variable: ", sigerr
# print "Error between hybrid sol and analytical sol: ", err
# error_diff = []
# comp_time = []
# for i in range(1, n):
#     e, t = test_hybridization_slate(deg, i)
#     error_diff.append(e)
#     comp_time.append(t)

# error_diff = np.array(error_diff)
# comp_time = np.array(comp_time)

# print "L2 error norms: ", error_diff
# print "Computational time to solve Schur-system: ", comp_time

# conv_rate = np.log2(error_diff[:-1] / error_diff[1:])
# print "Convergence order: ", conv_rate

# grid_points = np.array([2**(2*i) for i in range(1, n)])
# print grid_points
# comp_time_lu = []
# comp_time_ilu = []
# comp_time_hypre = []
# comp_time_gamg = []

# for pc_type in ['lu', 'ilu', 'hypre', 'gamg']:
#    for i in range(1, n):
#        comp_time = test_hybridization_slate(deg, i, schur_pc_type=pc_type)
#        if pc_type == 'lu':
#            comp_time_lu.append(comp_time)
#        elif pc_type == 'ilu':
#            comp_time_ilu.append(comp_time)
#        elif pc_type == 'hypre':
#            comp_time_hypre.append(comp_time)
#        elif pc_type == 'gamg':
#            comp_time_gamg.append(comp_time)
#        else:
#            raise ValueError()
# comp_time_lu = np.array(comp_time_lu)
# comp_time_hypre = np.array(comp_time_hypre)
# comp_time_gamg = np.array(comp_time_gamg)
# comp_time_ilu = np.array(comp_time_ilu)

# lu, = plt.loglog(grid_points, comp_time_lu, linewidth=3, linestyle=':', marker='^', color='g', label='LU')
# ilu, = plt.loglog(grid_points, comp_time_ilu, linewidth=3, linestyle='--', marker=None, color='b', label='ILU')
# hypre, = plt.loglog(grid_points, comp_time_hypre, linewidth=2, linestyle='--', marker='s', color='k', label='hypre')
# gamg, = plt.loglog(grid_points, comp_time_gamg, linewidth=2, linestyle='--', marker='s', color='r', label='GAMG')
# plt.legend([lu, ilu, hypre, gamg], ['LU', 'ILU', 'hypre', 'GAMG'], loc="NorthWest")
# plt.xlabel('Grid points')
# plt.ylabel('Time (s), logscale')
# plt.grid(True)
# plt.title('Computational time for solving the Schur-system')
# plt.show()

# xaxis = np.array([2**r for r in range(1, n)])
# plt.loglog(xaxis, error_diff, 'b^', xaxis, error_diff)
# plt.xlabel('Mesh resolution: $2^r, r = \{1, \dots, 10\}$')
# plt.ylabel('Absolute errors in the $L^2$-norm')
# plt.grid(True)
# plt.title('Resolution study for the scalar variable $u_h$: $RT_0$')
# plt.gca().invert_xaxis()
# plt.annotate('Convergence rate for $\parallel u_h - u \parallel_{2}$: %f' % conv_rate[-1], xy = {8e2, 1e-1})

# xaxis = np.array([2**r for r in range(1, n)])
# plt.loglog(xaxis, error_diff, 'gs', xaxis, error_diff, 'g-')
# plt.xlabel('Mesh resolution: $2^r, r = \{1, \dots, 10\}$')
# plt.ylabel('Approximation error of $u_h$ in the $L^2$ norm')
# plt.grid(True)
# plt.title('Log-log error plot of the scalar solution using the hybridized $RT_0$ method')
# plt.gca().invert_xaxis()
# plt.show()

# # assembledRHS = slate.slate_assemble(RHS).dat._data
# assembledRHS = assemble(RHS).dat.data
# print assembledRHS

# Lass = assemble(L)
# with Lass.dat.vec_ro as v:
#     lvec = v.array_r

# Aass = assemble(mass1 + mass2 + div - grad, nest=False).M.values
# KTass = assemble(trace_jump, nest=False).M.values.T
# print np.dot(KTass, np.dot(Aass, lvec))
