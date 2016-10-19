from firedrake import *
from firedrake.slate import slate


def slate_hybridization(degree, res):

    # Define the mesh and mesh normal
    mesh = UnitSquareMesh(2 ** res, 2 ** res)
    n = FacetNormal(mesh)

    # Define relevant finite element spaces
    RT = FiniteElement("RT", triangle, degree + 1)
    BRT = FunctionSpace(mesh, BrokenElement(RT))
    DG = FunctionSpace(mesh, "DG", degree)
    TraceSpace = FunctionSpace(mesh, "HDiv Trace", degree)

    # Define mixed space for velocity and pressure
    W = BRT * DG

    # Define trial and test functions
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    lambdar = TrialFunction(TraceSpace)
    gammar = TestFunction(TraceSpace)

    # Define finite element forms
    Mass1 = dot(sigma, tau)*dx
    Mass2 = u*v*dx
    Grad = div(tau)*u*dx
    Div = div(sigma)*v*dx
    trace = gammar('+')*dot(sigma, n)*dS
    trace_jump = jump(tau, n=n)*lambdar('+')*dS

    # Homogeneous Dirichlet boundary conditions
    bc = DirichletBC(TraceSpace, Constant(0), (1, 2, 3, 4))

    # Create Schur system with SLATE tensors
    A = slate.Matrix(Mass1 + Mass2 + Div - Grad)
    K = slate.Matrix(trace)
    Schur = -K * A.inv * K.T

    # Create right-hand side source function
    f = Function(DG)
    f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
    L = f*v*dx
    F = slate.Vector(L)
    RHS = -K * A.inv * F

    S = assemble(Schur, bcs=bc)
    E = assemble(RHS)

    lambda_sol = Function(TraceSpace)
    solve(S, lambda_sol, E, solver_parameters={'pc_type': 'gamg',
                                               'ksp_type': 'cg'})

    # Solve by back-substitution
    orig = assemble(L)
    orig -= assemble(action(trace_jump, lambda_sol))
    A = assemble(Mass1 + Mass2 + Div - Grad, nest=False)
    solution = Function(W)
    solve(A, solution, orig, solver_parameters={'ksp_type': 'preonly',
                                                'pc_type': 'lu'})
    sigma_h, u_h = solution.split()

    sigma_h = project(sigma_h, FunctionSpace(mesh, RT))
    File("solution.pvd").write(sigma_h, u_h)

degree = 2
res = 3
slate_hybridization(degree, res)
