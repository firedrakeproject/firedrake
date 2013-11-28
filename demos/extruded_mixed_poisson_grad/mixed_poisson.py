"""This demo solves a mixed formulation of Poisson's equation

  - div grad u(x, y) = 8*pi^2*cos(2*pi*x)*cos(2*pi*y)

on an extruded cube of dimensions 1 x 1 x 0.1 with boundary conditions given by:

  u(x, y, 0) = 10 + cos(2*pi*x)*cos(2*pi*y)
  u(x, y, 0.1) = cos(2*pi*x)*cos(2*pi*y)

Homogeneous Neumann boundary conditions are applied naturally on the
other sides of the domain.

This has the analytical solution

  u(x, y, z) = cos(2*pi*x)*cos(2*pi*y) + 10(1-10*z)

"""
from firedrake import *

# Create mesh and define function space
m = UnitSquareMesh(16, 16)
mesh = ExtrudedMesh(m, layers=4, layer_height=0.025)

V0 = FunctionSpace(mesh, "CG", 2, vfamily="CG", vdegree=1)

V1_a_horiz = FiniteElement("BDM", "triangle", 1)
V1_a_vert = FiniteElement("CG", "interval", 1)
V1_a = HCurl(OuterProductElement(V1_a_horiz, V1_a_vert))

V1_b_horiz = FiniteElement("CG", "triangle", 2)
V1_b_vert = FiniteElement("DG", "interval", 0)
V1_b = HCurl(OuterProductElement(V1_b_horiz, V1_b_vert))

V1_elt = EnrichedElement(V1_a, V1_b)
V1 = FunctionSpace(mesh, V1_elt)

W = V0 * V1

bcs = [DirichletBC(W.sub(0), Expression("10 + cos(2*pi*x[0])*cos(2*pi*x[1])"), "bottom"),
       DirichletBC(W.sub(0), Expression("cos(2*pi*x[0])*cos(2*pi*x[1])"), "top")]

f = Function(V0)
f.interpolate(Expression("8*pi*pi*cos(2*pi*x[0])*cos(2*pi*x[1])"))

u, sigma = TrialFunctions(W)
v, tau = TestFunctions(W)
a = (dot(tau, sigma) - dot(grad(v), sigma) - dot(grad(u), tau))*dx
L = f*v*dx

out = Function(W)
solve(a == L, out, bcs=bcs)
u, sigma = out.split()

exact = Function(V0)
exact.interpolate(Expression("cos(2*pi*x[0])*cos(2*pi*x[1]) + 10*(1-10*x[2])"))

res = sqrt(assemble(dot(u - exact, u - exact) * dx))
print res
