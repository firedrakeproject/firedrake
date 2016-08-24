from firedrake import *
from firedrake.slate import slate
import numpy as np


def test_hybridization_slate(degree, mesh_res=None, write=False):

    # Generate UnitSquareMesh and facet normal
    if mesh_res is None:
        mesh = UnitSquareMesh(10, 10)
    elif isinstance(mesh_res, int):
        mesh = UnitSquareMesh(2 ** mesh_res, 2 ** mesh_res)
    else:
        raise ValueError("Integers or None are only accepted for mesh_res.")
    n = FacetNormal(mesh)

    # Define relevant finite element spaces
    RT = FiniteElement("RT", triangle, degree+1)
    BRT = FunctionSpace(mesh, BrokenElement(RT))
    DG = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", degree)
    W = BRT * DG

    # Define trial and test functions of the finite element spaces
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    lambdar = TrialFunction(T)
    gammar = TestFunction(T)

    # Define finite element forms
    Mass1 = dot(sigma, tau)*dx
    Mass2 = u*v*dx
    Grad = div(tau)*u*dx
    Div = div(sigma)*v*dx
    trace = lambdar('+')*dot(tau, n)*dS
    trace_jump = jump(tau, n=n)*lambdar('+')*dS

    # Homogeneous Dirichlet boundary conditions on all edges of the UnitSquareMesh
    bc = DirichletBC(T, Constant(0), (1, 2, 3, 4))

    # Creating Schur matrix with SLATE tensors
    A = slate.Matrix(Mass1 + Mass2 + Div - Grad)
    K = slate.Matrix(trace)
    schur = K.T * A.inv * K

    # Creating left-hand side source function
    f = Function(DG)
    f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    L = f*v*dx

    # Left-hand side vector for the Schur-compliment system  in SLATE
    F = slate.Vector(L)
    RHS = K.T*A.inv*F

    S = assemble(schur, bcs=bc)
    b = assemble(RHS)

    # Now we solve for the Langrange multiplier
    lambda_sol = Function(T)
    solve(S, lambda_sol, b, solver_parameters={'pc_type': 'lu',
                                               'pc_factor_mat_solver_package': 'mumps'},
          options_prefix="")

    # Construct right-hand side for reconstruction in UFL
    orig = assemble(L)
    orig -= assemble(action(trace_jump, lambda_sol))

    # Solve via back-substitution
    a = Mass1 + Mass2 + Div - Grad
    A = assemble(a, nest=False)
    solution = Function(W)
    solve(A, solution, orig, solver_parameters={'ksp_type': 'preonly',
                                                'pc_type': 'lu'})
    sigma_h, u_h = solution.split()

    # Compare result with non-hybridized computation
    RTc = FunctionSpace(mesh, "RT", degree + 1)
    W2 = RTc * DG
    sigma, u = TrialFunctions(W2)
    tau, v = TestFunctions(W2)
    a = (dot(sigma, tau) - u*div(tau) + u*v + v*div(sigma))*dx
    L = f*v*dx
    non_hybrid_sol = Function(W2)
    solve(a == L, non_hybrid_sol, solver_parameters={'pc_type': 'fieldsplit',
                                                     'pc_fieldsplit_type': 'schur',
                                                     'ksp_type': 'cg',
                                                     'pc_fieldsplit_schur_fact_type': 'FULL',
                                                     'fieldsplit_V_ksp_type': 'cg',
                                                     'fieldsplit_P_ksp_type': 'cg'})
    nhsigma, nhu = non_hybrid_sol.split()

    # Return L2 error (should be identical w.r.t. solver tolerance)
    uerr = sqrt(assemble((u_h - nhu)*(u_h - nhu)*dx))
    sigerr = sqrt(assemble(dot(sigma_h - nhsigma, sigma_h - nhsigma)*dx))

    # Analytical solution
    f.interpolate(Expression("sin(x[0]*pi*2)*sin(x[1]*pi*2)"))

    error = sqrt(assemble((u_h - f)*(u_h - f)*dx))

    if write:
        # Write hybridized solutions to paraview file
        sigma_h = project(sigma_h, FunctionSpace(mesh, RT))
        File("solution.pvd").write(sigma_h, u_h, nhsigma, nhu)
    return uerr, sigerr, error

deg = 0
# uerr, sigerr, err = test_hybridization_slate(deg, write=True)
# print "Error in scalar variable: ", uerr
# print "Error in flux variable: ", sigerr
# print "Error between hybrid sol and analytical sol: ", err
error_diff = np.array([test_hybridization_slate(deg, mesh_res=i)[2] for i in range(1, 6)])
print "L2 error norms: ", error_diff

conv_rate = np.log2(error_diff[:-1] / error_diff[1:])
print "Convergence order: ", conv_rate

# # assembledRHS = slate.slate_assemble(RHS).dat._data
# assembledRHS = assemble(RHS).dat.data
# print assembledRHS

# Lass = assemble(L)

# with Lass.dat.vec_ro as v:
#     lvec = v.array_r

# Aass = assemble(mass1 + mass2 + div - grad, nest=False).M.values

# KTass = assemble(trace_jump, nest=False).M.values.T

# print np.dot(KTass, np.dot(Aass, lvec))
