"""This demo solves a mixed formulation of the Helmholtz equation

  - div grad u(x, y) + u = 1 + 38*pi^2*sin(2*pi*x)*sin(3*pi*y)*sin(5*pi*z)

on an extruded cube of dimensions 1 x 1 x 0.2, with boundary conditions chosen so that
the analytical solution is

  u(x, y, z) = sin(2*pi*x)*sin(3*pi*y)*sin(5*pi*z)

"""
from firedrake import *

m = UnitSquareMesh(16, 16)
mesh = ExtrudedMesh(m, layers=8, layer_height=0.025)

V2_a_horiz = FiniteElement("RT", "triangle", 2)
V2_a_vert = FiniteElement("DG", "interval", 1)
V2_a = HDiv(OuterProductElement(V2_a_horiz, V2_a_vert))

V2_b_horiz = FiniteElement("DG", "triangle", 1)
V2_b_vert = FiniteElement("CG", "interval", 2)
V2_b = HDiv(OuterProductElement(V2_b_horiz, V2_b_vert))

V2_elt = EnrichedElement(V2_a, V2_b)
V2 = FunctionSpace(mesh, V2_elt)

V3 = FunctionSpace(mesh, "DG", 1, vfamily="DG", vdegree=1)

f = Function(V3)
exact = Function(V3)
f.interpolate(Expression("(1+38*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5)"))
exact.interpolate(Expression("sin(x[0]*pi*2)*sin(x[1]*pi*3)*sin(x[2]*pi*5)"))

W = V2 * V3
sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
a = (u*v - v*div(sigma) + dot(sigma, tau) + div(tau)*u)*dx
L = f*v*dx

out = Function(W)
solve(a == L, out, solver_parameters={'pc_type': 'fieldsplit',
                                      'pc_fieldsplit_type': 'schur',
                                      'ksp_type': 'cg',
                                      'pc_fieldsplit_schur_fact_type': 'FULL',
                                      'fieldsplit_0_ksp_type': 'cg',
                                      'fieldsplit_1_ksp_type': 'cg'})
sigma, u = out.split()
print "L2 norm: " + str(sqrt(assemble((u-exact)*(u-exact)*dx)))
File("u.pvd") << project(u, FunctionSpace(mesh, "CG", 1))
