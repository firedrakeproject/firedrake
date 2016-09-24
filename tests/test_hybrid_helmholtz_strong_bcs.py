from firedrake import *
import numpy as np

def test_hybrid_helmholtz(mesh):
    degree = 2
    RT = FiniteElement("RT", triangle, degree)

    BrokenRT = FunctionSpace(mesh, BrokenElement(RT))
    DG = FunctionSpace(mesh, "DG", degree - 1)
    TraceRT = FunctionSpace(mesh, "HDiv Trace", degree - 1)

    W = MixedFunctionSpace([BrokenRT, DG, TraceRT])

    # Trial and Test functions
    sigma, u, lambdar = TrialFunctions(W)
    tau, v, gammar = TestFunctions(W)

    # Mesh normal
    n = FacetNormal(mesh)

    # Source function
    f = Function(DG)
    f.interpolate(Expression("(1+8*pi*pi)*sin(2*pi*x[0])*sin(2*pi*x[1])"))

    # Define the variational forms
    a_dx = dot(tau, sigma)*dx + u*div(tau)*dx + u*v*dx + v*div(sigma)*dx
    a_dS = lambdar('+')*jump(tau, n=n)*dS + gammar('+')*jump(sigma, n=n)*dS
    a = a_dx + a_dS
    L = -v*f*dx

    # Strongly enforce BC
    bc = DirichletBC(W.sub(2), Constant(0), (1, 2, 3, 4))

    # Compute solution
    w = Function(W)
    solve(a == L, w, solver_parameters={'ksp_rtol': 1e-14,
                                        'ksp_max_it': 30000}, bcs=bc)
    Hsigma, Hu, Hlambdar = w.split()

    residual_helmholtz = sqrt(assemble(((Hu +  div(Hsigma)) + f)*((Hu + div(Hsigma)) + f)*dx))

    File('Hu.pvd').write(Hu)
    V = VectorFunctionSpace(mesh, "DG", degree - 1)
    Hsigma_out = Function(V).project(Hsigma)
    File('Hsigma.pvd').write(Hsigma_out)

    # Non-hybridized helmholtz for comparison
    RT = FunctionSpace(mesh, "RT", degree)
    W2 = RT * DG
    sigma, u = TrialFunctions(W2)
    tau, v = TestFunctions(W2)
    w2 = Function(W2)
    a = dot(tau, sigma)*dx + u*div(tau)*dx + u*v*dx + v*div(sigma)*dx
    L = -v*f*dx
    bc = DirichletBC(W2.sub(1), Constant(0), (1, 2, 3, 4))
    solve(a == L, w2, solver_parameters={'ksp_rtol': 1e-14,
                                         'ksp_max_it': 30000}, bcs=bc)
    NHsigma, NHu = w2.split()

    File('NHu.pvd').write(NHu)
    File('NHsigma.pvd').write(NHsigma)

    uerr = sqrt(assemble((Hu - NHu)*(Hu - NHu)*dx))
    sigerr = sqrt(assemble(dot(Hsigma - NHsigma, Hsigma - NHsigma)*dx))

    print "The residual for the Helmholtz problem is: ", residual_helmholtz
    print "There error in the computed vector term is: ", uerr
    print "The error in the computed scalar term is: ", sigerr

res = 10
mesh = UnitSquareMesh(res, res)
test_hybrid_helmholtz(mesh)
