"""This demo doesn't solve a mixed formulation of Poisson's equation

  div grad u(x, y) = (12x^2-12x+2)*[y(1-y)]^2 + (12y^2-12y+2)*[x(1-x)]^2

on an extruded cube of dimensions 1 x 1 x 0.1 with boundary conditions given by:

  u(x, y, 0) = 10 + [x(1-x)]^2*[y(1-y)]^2
  u(x, y, 0.1) = [x(1-x)]^2*[y(1-y)]^2

Homogeneous Neumann boundary conditions are applied naturally on the
other sides of the domain.

This has the analytical solution

  u(x, y, z) = [x(1-x)]^2*[y(1-y)]^2 + 10(1-10z)

"""
from firedrake import *

# Create mesh and define function space
m = UnitSquareMesh(16, 16)
mesh = ExtrudedMesh(m, layers=8, layer_height=0.0125)

V2_a_horiz = FiniteElement("BDM", "triangle", 1)
V2_a_vert = FiniteElement("DG", "interval", 0)
V2_a = HDiv(OuterProductElement(V2_a_horiz, V2_a_vert))

V2_b_horiz = FiniteElement("DG", "triangle", 0)
V2_b_vert = FiniteElement("CG", "interval", 1)
V2_b = HDiv(OuterProductElement(V2_b_horiz, V2_b_vert))

V2_elt = EnrichedElement(V2_a, V2_b)
V2 = FunctionSpace(mesh, V2_elt)

V3 = FunctionSpace(mesh, "DG", 0, vfamily="DG", vdegree=0)

W = V2 * V3

bcs = [DirichletBC(W.sub(1), Expression("10 + x[0]*x[0]*(1-x[0])*(1-x[0])*x[1]*x[1]*(1-x[1])*(1-x[1])"), "bottom"),
       DirichletBC(W.sub(1), Expression("x[0]*x[0]*(1-x[0])*(1-x[0])*x[1]*x[1]*(1-x[1])*(1-x[1])"), "top")]

f = Function(V3)
f.interpolate(Expression("(12*x[0]*x[0] - 12*x[0] + 2)*x[1]*x[1]*(1-x[1])*(1-x[1]) + (12*x[1]*x[1] - 12*x[1] + 2)*x[0]*x[0]*(1-x[0])*(1-x[0])"))

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
a = (v*div(sigma) + dot(tau, sigma) + div(tau)*u)*dx
L = f*v*dx

out = Function(W)
solve(a == L, out, bcs=bcs)
sigma, u = out.split()

exact = Function(V3)
exact.interpolate(Expression("x[0]*x[0]*(1-x[0])*(1-x[0])*x[1]*x[1]*(1-x[1])*(1-x[1]) + 10*(1-10*x[2])"))

res = sqrt(assemble(dot(u - exact, u - exact) * dx))
print res
