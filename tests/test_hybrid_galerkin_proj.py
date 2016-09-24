from firedrake import *
import numpy as np

def hybrid_galerkin_proj(mesh):
    # Define relevant function spaces
    degree = 3
    RaviartThomas = FiniteElement("RT", triangle, degree)
    HRT = FiniteElement("RT", triangle, degree)
    HRTspace = FunctionSpace(mesh, HRT)
    BrokenRT = FunctionSpace(mesh, BrokenElement(RaviartThomas))
    TraceRT = FunctionSpace(mesh, "HDiv Trace", degree-1)

    W = BrokenRT*TraceRT

    # Define trial and test functions
    sigma, lambdar = TrialFunctions(W)
    tau, gammar = TestFunctions(W)

    # Mesh normal
    n = FacetNormal(mesh)

    # Define function we wish to Galerkin project
    f = project(Expression(("cos(x[0])","sin(x[1])")), HRTspace)

    # Define variational forms
    a_dx = (dot(tau, sigma))*dx
    a_dS = (jump(tau, n=n)*lambdar('+'))*dS + (gammar('+')*jump(sigma, n=n))*dS
    A = a_dx + a_dS
    L = dot(tau, f)*dx

    # Compute projection
    w = Function(W)
    solve(A == L, w, solver_parameters={'ksp_rtol': 1e-14,
                                        'ksp_max_it': 30000})
    hSigma, hLambdar = w.split()

    sigmaError = sqrt(assemble(dot(hSigma - f, hSigma - f)*dx))
    lambdarError = sqrt(assemble((2*avg(hLambdar) - jump(f, n=n))*(2*avg(hLambdar) - jump(f, n=n))*dS))

    return sigmaError, lambdarError

sigErr = []
lamErr = []

    # Create a mesh
mesh = UnitSquareMesh(10,10, quadrilateral=True)

e = hybrid_galerkin_proj(mesh)
sigErr.append(e[0])
lamErr.append(e[1])

sigErr = np.array(sigErr)
lamErr = np.array(lamErr)

print np.log(sigErr[1:]/sigErr[:-1])/np.log(0.5)
print np.log(lamErr[1:]/lamErr[:-1])/np.log(0.5)

print "The error in sigma is: ", sigErr
print "The error in the trace solution is: ", lamErr
