"""This demo solves Poisson's equation

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
mesh = ExtrudedMesh(m, layers=5, layer_height=0.02)

V = FunctionSpace(mesh, "CG", 2)
bcs = [DirichletBC(V, Expression("10 + cos(2*pi*x[0])*cos(2*pi*x[1])"), "bottom"),
       DirichletBC(V, Expression("cos(2*pi*x[0])*cos(2*pi*x[1])"), "top")]

u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx
f = Function(V)
f.interpolate(Expression("8*pi*pi*cos(2*pi*x[0])*cos(2*pi*x[1])"))
L = v * f * dx
u = Function(V)
exact = Function(V)
exact.interpolate(Expression("cos(2*pi*x[0])*cos(2*pi*x[1]) + 10*(1-10*x[2])"))
solve(a == L, u, bcs=bcs)
res = sqrt(assemble(dot(u - exact, u - exact) * dx))
print res
