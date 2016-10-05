from firedrake import *
from firedrake.slate import slate

mesh = UnitSquareMesh(8, 8)
n = FacetNormal(mesh)

degree = 0
RT = FiniteElement("RT", triangle, degree + 1)
BRT = FunctionSpace(mesh, BrokenElement(RT))
DG = FunctionSpace(mesh, "DG", degree)
T = FunctionSpace(mesh, "HDiv Trace", degree)
W = BRT * DG

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
gammar = TestFunction(T)

f = Function(DG)
f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))

bc = DirichletBC(T, Constant(0), (1, 2, 3, 4))

A = slate.Matrix(dot(sigma, tau)*dx + u*v*dx
                 + div(tau)*u*dx + div(sigma)*v*dx)
K = slate.Matrix(gammar('+')*dot(sigma, n)*dS)
F = slate.Vector(f*v*dx)

S = -K*A.inv*K.T
E = -K*A.inv*F

Smat = assemble(S, bcs=bc).M.values
Evec = assemble(E).dat._data

print Smat
print Evec
