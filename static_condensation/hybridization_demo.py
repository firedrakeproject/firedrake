"""This is a demonstration of using the SLATE language for solving
a hybridized finite element problem."""

from __future__ import absolute_import

import numpy as np
import slate

from firedrake import *

mesh = UnitSquareMesh(1, 1, quadrilateral=False)
degree = 1
RT_element = FiniteElement("RT", triangle, degree)

BrokenRTSpace = FunctionSpace(mesh, BrokenElement(RT_element))
DG = FunctionSpace(mesh, "DG", degree-1)
TraceSpace = FunctionSpace(mesh, "HDiv Trace", degree-1)

W = MixedFunctionSpace([BrokenRTSpace, DG, TraceSpace])

sigma, u, lambdar = TrialFunctions(W)
tau, v, gammar = TestFunctions(W)

n = FacetNormal(mesh)

f = Function(DG)
f.interpolate(Expression("(1+8*pi*pi)*sin(2*pi*x[0])*sin(2*pi*x[1])"))

bcs = DirichletBC(TraceSpace, Constant(0), (1, 2, 3, 4))

Mass = dot(sigma, tau)*dx + u*v*dx
Div = div(sigma)*v*dx
Grad = div(tau)*u*dx

A_f = assemble(Mass + Div - Grad, nest=False).M.values

Trace = jump(tau, n=n)*lambdar('+')*dS + jump(sigma, n=n)*gammar('+')*dS
Positive_trace = dot(tau, n)('+')*lambdar('+')*dS + dot(sigma, n)('+')*gammar('+')*dS

K = slate.Matrix(Positive_trace)
A = slate.Matrix(Mass + Div - Grad)

Schur = -K.T*A.inv*K
slate_schur = slate.slate_assemble(Schur, bcs=[bcs]).values

trace_f = assemble(Trace, nest=False).M.values

schur_f = np.dot(trace_f.T, np.dot(np.linalg.inv(A_f), trace_f))
schur_f[bcs.nodes, :] = 0
schur_f[:, bcs.nodes] = 0

d = np.zeros_like(schur_f)
d[bcs.nodes, bcs.nodes] = 1
thunk = d - schur_f

L = f*v*dx
F = slate.Vector(L)
rhs = K.T*A.inv*F

print thunk
print slate_schur
print np.allclose(thunk, slate_schur)
