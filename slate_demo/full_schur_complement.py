from firedrake import *
from firedrake.static_condensation import slate
import numpy as np
import ipdb


# Defining mesh, finite element spaces and forms
res = 1
degree = 0

mesh = UnitSquareMesh(res, res)
n = FacetNormal(mesh)

RT = FiniteElement("RT", triangle, degree+1)
BRT = FunctionSpace(mesh, BrokenElement(RT))
DG = FunctionSpace(mesh, "DG", degree)
T = FunctionSpace(mesh, "HDiv Trace", degree)

sigma = TrialFunction(BRT)
tau = TestFunction(BRT)
u = TrialFunction(DG)
v = TestFunction(DG)
lambdar = TrialFunction(T)
gammar = TestFunction(T)

a = dot(sigma, tau)*dx
b = v*div(tau)*dx
c = u*v*dx
d = gammar('+')*dot(sigma, n)('+')*dS

bc = DirichletBC(T, Constant(0), (1, 2, 3, 4))

# Creating SLATE tensors
A = slate.Matrix(a)
B = slate.Matrix(b)
C = slate.Matrix(c)
D = slate.Matrix(d)

M = B*A.inv*B.T + C
K = A - B.T*M.inv*B
S = -D*A.inv*K*A.inv*D.T

coords, coeffs, need_cell_facets, op2kernel = slate.compile_slate_expression(S)
ipdb.set_trace()
