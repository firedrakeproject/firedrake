from firedrake import *

import slate

parameters["pyop2_options"]["debug"] = True
mesh = UnitTriangleMesh()

RT = FiniteElement("RT", triangle, 1)

BRT = FunctionSpace(mesh, BrokenElement(RT))

DG = FunctionSpace(mesh, "DG", 0)

T = FunctionSpace(mesh, "HDiv Trace", 0)

W = MixedFunctionSpace([BRT, DG])

sigma, u = TrialFunctions(W)

tau, v = TestFunctions(W)

lambdar = TrialFunction(T)

gammar = TestFunction(T)

Mass = dot(sigma, tau)*dx + u*v*dx

Div = div(sigma)*v*dx

Grad = div(tau)*u*dx
n = FacetNormal(mesh)

trace = lambdar*dot(tau, n)*dS

A = slate.Matrix(Mass + Div - Grad)

K = slate.Matrix(trace)

S = -K.T*A.inv*K

coords, coeffs, facet_flag, kernel = slate.compile_slate_expression(S)

import ipdb
ipdb.set_trace(context=5)
print kernel


#thunk = slate.slate_assemble(S)

#print thunk.values
