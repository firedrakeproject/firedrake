Double slit experiment
======================

Here we solve a linear wave equation using an explicit timestepping
scheme. This example demonstrates the use of an externally generated
mesh, and a time varying boundary condition. The strong form of the
equation we set out to solve is:

.. math::

   \frac{\partial p}{\partial t} + \nu\nabla^2 \phi = 0

   \frac{\partial \phi}{\partial t} = -p

   \nabla p \cdot n = 0 \ \textrm{on}\ \Gamma_N
   
   p = \sin(10\pi t)  \ \textrm{on}\ \Gamma_D




from firedrake import *
mesh = Mesh("wave_tank.msh")

T = 0.01
dt = 0.001
t = 0
fs = FunctionSpace(mesh, 'Lagrange', 2)
p = Function(fs)
phi = Function(fs)

u = TrialFunction(fs)
v = TestFunction(fs)

outfs = FunctionSpace(mesh, 'Lagrange', 1)

bc = DirichletBC(fs, 0.0, 1)

outfile = File("out.pvd")
outfile << project(p, outfs)

step = 0

while t <= T:
    step += 1

    bc.set_value(sin(2*pi*5*t))

    phi -= dt / 2 * p
    solve(u * v * dx == v * p * dx + dt * inner(grad(v), grad(phi)) * dx,
          p, bcs=bc, solver_parameters={'ksp_type': 'cg',
                                        'pc_type': 'sor',
                                        'pc_sor_symmetric': True})
    phi -= dt / 2 * p

    t += dt
    if step % 10 == 0:
        outfile << project(p, outfs)
