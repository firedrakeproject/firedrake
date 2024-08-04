# import finat
# from firedrake import *
# from firedrake.adjoint import *
# from firedrake.__future__ import Interpolator, interpolate
# import gc
# import numpy as np
# from checkpoint_schedules import Revolve
# continue_annotation()

# tape = get_working_tape()
# total_steps = 10
# tape.enable_checkpointing(Revolve(total_steps, 2))

# num_sources = 1
# source_number = 0
# Lx, Lz = 1.0, 1.0
# mesh = UnitSquareMesh(50, 50)
# V = FunctionSpace(mesh, "CG", 1)


# def Dt(u, u_, timestep):
#     return (u - u_)/timestep

# def test_memory_burger():
#     u_ = Function(V)
#     u = Function(V)
#     v = TestFunction(V)
#     timestep = Constant(0.0000001)
#     x,_ = SpatialCoordinate(mesh)
#     ic = project(sin(2.*pi*x), V)
#     u_.assign(ic)
#     nu = Constant(0.001)
#     F = (Dt(u, u_, timestep)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
#     bc = DirichletBC(V, 0.0, "on_boundary")
#     t = 0.0
#     problem = NonlinearVariationalProblem(F, u, bcs=bc)
#     solver = NonlinearVariationalSolver(problem)
#     t += float(timestep)
#     for t in tape.timestepper(iter(range(total_steps))):
#         print("step = ", t, "revolve ")
#         solver.solve()
#         u_.assign(u)
#     J = assemble(u*u*dx)
#     J_hat = ReducedFunctional(J, Control(ic))
#     J_hat(ic)
#     taylor_test(J_hat, ic, Function(V).assign(0.1, annotate=False))
#     assert np.allclose(J_hat(ic), J)
#     print(max(u_.dat.data_ro[:]))
#     VTKFile("burger.pvd").write(u_)


# test_memory_burger()
# # tape.visualise()
# print("done")
from firedrake import *
from pyadjoint import *
from firedrake.adjoint import *

continue_annotation()
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = u*v*dx
L = Constant(1)*v*dx
u0 = Function(V)
solve(a == L, u0)
j = assemble(u0 * u0 * dx)
g = exp(j)
dg_dtheta = compute_gradient(g, Control(u0))
# print(rank, dg_dtheta.dat.data_ro[:])
# pause_annotation()