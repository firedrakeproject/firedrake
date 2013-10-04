
from firedrake import *

output = True

mesh = UnitSquareMesh(100, 100)
# Plumb the space filling curve into UnitSquareMesh after the call to
# gmsh. Doru knows how to do this.

T = 10
# Note to Kaho: Ensure dt<dx for stability.
dt = 0.001
t = 0
fs = FunctionSpace(mesh, 'Lagrange', 1)
p = Function(fs)
phi = Function(fs)

u = TrialFunction(fs)
v = TestFunction(fs)

p.interpolate(Expression("exp(-40*((x[0]-.5)*(x[0]-.5)+(x[1]-.5)*(x[1]-.5)))"))

if output:
    outfile = File("out.pvd")
    phifile = File("phi.pvd")

    outfile << p
    phifile << phi

# Mass matrix
m = u * v * dx

lump_mass = False

step = 0
while t <= T:
    step += 1

    phi -= dt / 2 * p

    if lump_mass:
        p += (assemble(dt * inner(nabla_grad(v), nabla_grad(phi)) * dx)
              / assemble(v * dx))
    else:
        solve(u * v * dx == v * p * dx + dt * inner(
            nabla_grad(v), nabla_grad(phi)) * dx, p)

    phi -= dt / 2 * p

    t += dt

    if output:
        print t
        outfile << p
        phifile << phi
