from firedrake import *
import numpy as np
import slate


# Defining mesh, finite element spaces and forms
res = 1
degree = 0

mesh = UnitSquareMesh(res, res)
n = FacetNormal(mesh)

RT = FiniteElement("RT", triangle, degree+1)
BRT = FunctionSpace(mesh, BrokenElement(RT))
DG = FunctionSpace(mesh, "DG", degree)
T = FunctionSpace(mesh, "HDiv Trace", degree)

W = BRT * DG

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
lambdar = TrialFunction(T)
gammar = TestFunction(T)

mass1 = dot(sigma, tau)*dx
mass2 = u*v*dx
grad = div(tau)*u*dx
div = div(sigma)*v*dx
trace = lambdar('+')*dot(tau, n)('+')*dS

trace_jump = jump(tau, n=n)*lambdar('+')*dS

bc = DirichletBC(T, Constant(0), (1, 2, 3, 4))

# Assembling with Firedrake and with SLATE
hdiv_cell = assemble(mass1 + mass2 + div - grad, nest=False).M.values
firedrake_trace = assemble(trace_jump, nest=False).M.values
firedrake_schur = np.dot(firedrake_trace.T, np.dot(np.linalg.inv(hdiv_cell),
                                                   firedrake_trace))
firedrake_schur[bc.nodes, :] = 0
firedrake_schur[:, bc.nodes] = 0

D = np.zeros_like(firedrake_schur)
D[bc.nodes, bc.nodes] = 1.0
thunk = D - firedrake_schur

A = slate.Matrix(mass1 + mass2 + div - grad)
K = slate.Matrix(trace)
schur = -K.T * A.inv * K
slate_schur = slate.slate_assemble(schur, bcs=[bc])._M.values

f = Function(DG)
f.interpolate(Expression("(1+8+pi*pi)*sin(2*pi*x[0])*sin(2*pi*x[1])"))
L = f*v*dx
F = slate.Vector(L)
RHS = K.T*A*F
assembledRHS = slate.slate_assemble(RHS).dat._data
print thunk
print slate_schur
print assembledRHS
print np.allclose(thunk, slate_schur)
