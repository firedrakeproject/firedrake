import sys
from firedrake import *

output = True

mesh = Mesh("wave_tank.msh")

T = 10.

dt = 0.001
t = 0
fs = FunctionSpace(mesh, 'Lagrange', 2)
p = Function(fs)
phi = Function(fs)

u = TrialFunction(fs)
v = TestFunction(fs)

outfs = FunctionSpace(mesh, 'Lagrange', 1)

bc = [DirichletBC(fs, 0.0, 1)]

if output:
    outfile = File("out.pvd")
    outfile << project(p, outfs)

step = 0

while t <= T:
    timer.start()
    step += 1

    bc[0].set_value(sin(2*pi*5*t))

    phi -= dt / 2 * p

    solve(u * v * dx == v * p * dx + dt * inner(nabla_grad(v), nabla_grad(phi)) * dx,
          p, bcs=bc, solver_parameters={'ksp_type': 'cg',
                                        'pc_type': 'sor',
                                        'pc_sor_symmetric': True})

    phi -= dt / 2 * p

    t += dt

    sys.stdout.write("\r"+str(t)+"   ")
    sys.stdout.flush()
    if output and step % 10 == 0:
        outfile << project(p, outfs)
