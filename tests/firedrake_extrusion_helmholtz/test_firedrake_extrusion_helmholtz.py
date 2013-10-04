"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

# Begin demo
from firedrake import *
import sys

power = int(sys.argv[1])
# Create mesh and define function space
m = UnitSquareMesh(2 ** power, 2 ** power)
layers = 2 ** int(sys.argv[1]) + 1

# Populate the coordinates of the extruded mesh by providing the
# coordinates as a field.
extrusion_kernel = """
void extrusion_kernel(double *xtr[], double *x[], int* j[])
{
    //Only the Z-coord is increased, the others stay the same
    xtr[0][0] = x[0][0];
    xtr[0][1] = x[0][1];
    xtr[0][2] = %(height)s*j[0][0];
}""" % {'height': str(1.0 / 2 ** int(sys.argv[1]))}

mesh = ExtrudedMesh(m, layers, extrusion_kernel)
V = FunctionSpace(mesh, "Lagrange", 1, vfamily="Lagrange", vdegree=1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
f.interpolate(
    Expression("(1+12*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2)*cos(x[2]*pi*2)"))
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx

# Compute solution
A = assemble(a)
b = assemble(L)
x = Function(V)
solve(a == L, x)

# Analytical solution
f.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)*cos(x[2]*pi*2)"))

print sqrt(assemble((x - f) * (x - f) * dx))

# Save solution in VTK format
#file = File("helmholtz.pvd")
#file << x
#file << f
