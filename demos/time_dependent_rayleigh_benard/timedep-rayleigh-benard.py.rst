Rayleigh-Benard Convection
==========================

Contributed by `Robert Kirby <https://sites.baylor.edu/robert_kirby/>`_
and `Pablo Brubeck <https://www.maths.ox.ac.uk/people/pablo.brubeckmartinez/>`_.

This problem involves a variable-temperature incompressible fluid.
Variations in the fluid temperature are assumed to affect the momentum
balance through a buoyant term (the Boussinesq approximation), leading
to a Navier-Stokes equation with a nonlinear coupling to a
convection-diffusion equation for temperature.

We will set up the problem using Taylor-Hood elements for
the Navier-Stokes part, and piecewise linear elements for the
temperature. ::

  from firedrake import *
  from firedrake.pyplot import FunctionPlotter, tripcolor
  import math
  import matplotlib.pyplot as plt
  from matplotlib.animation import FuncAnimation

  try:
      from irksome import Dt, RadauIIA, TimeStepper
  except ImportError:
      import sys
      warning("Unable to import irksome.")
      sys.exit(0)

  Nbase = 8
  ref_levels = 2
  N = Nbase * 2**ref_levels

  base_msh = UnitSquareMesh(Nbase, Nbase)
  mh = MeshHierarchy(base_msh, ref_levels)
  msh = mh[-1]

  V = VectorFunctionSpace(msh, "CG", 2)
  W = FunctionSpace(msh, "CG", 1)
  Q = FunctionSpace(msh, "CG", 1)
  Z = V * W * Q

  upT = Function(Z)
  u, p, T = split(upT)
  v, q, S = TestFunctions(Z)

Two key physical parameters are the Rayleigh number (Ra), which
measures the ratio of energy from buoyant forces to viscous
dissipation and heat conduction and the
Prandtl number (Pr), which measures the ratio of viscosity to heat
conduction. ::

  Ra = Constant(2000.0)
  Pr = Constant(6.8)

Along with gravity, which points down. ::

  g = Constant((0, -1))

Set up variables for time and time-step size. ::

  t = Constant(0.0)
  dt = Constant(1.0 / N)

  F = (
      inner(Dt(u), v)*dx
      + inner(grad(u), grad(v))*dx
      + inner(dot(grad(u), u), v)*dx
      - inner(p, div(v))*dx
      - (Ra/Pr)*inner(T*g, v)*dx
      + inner(div(u), q)*dx
      + inner(Dt(T), S)*dx
      + inner(dot(grad(T), u), S)*dx
      + 1/Pr * inner(grad(T), grad(S))*dx
  )

There are two common versions of this problem.  In one case, heat is
applied from bottom to top so that the temperature gradient is
enforced parallel to the gravitation.  In this case, the temperature
difference is applied horizontally, perpendicular to gravity.  It
tends to make prettier pictures for low Rayleigh numbers, but also
tends to take more Newton iterations since the coupling terms in the
Jacobian are a bit stronger.  Switching to the first case would be a
simple change of bits of the boundary associated with the second and
third boundary conditions below::

  bcs = [
      DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4)),
      DirichletBC(Z.sub(2), Constant(1.0), (3,)),
      DirichletBC(Z.sub(2), Constant(0.0), (4,))
  ]

Like Navier-Stokes, the pressure is only defined up to a constant.::

  nullspace = [(1, VectorSpaceBasis(constant=True))]

Set up the Butcher tableau to use for time-stepping::

  num_stages = 2
  butcher_tableau = RadauIIA(num_stages)
  
We are going to carry out time stepping via Irksome, but we need
to say how to solve the rather interesting stage-coupled system. ::

  exclude_inds = ",".join([str(3*i+j) for i in range(num_stages) for j in (1, 2)])
  params = {
      "mat_type": "aij",
      "snes_type": "newtonls",
      "snes_converged_reason": None,
      "snes_linesearch_type": "l2",
      "snes_monitor": None,
      "ksp_type": "fgmres",
      "ksp_monitor": None,
      "ksp_max_it": 200,
      "ksp_atol": 1.e-12,
      "ksp_gmres_restart": 30,
      "snes_rtol": 1.e-10,
      "snes_atol": 1.e-10,
      "snes_ksp_ew": None,
      "pc_type": "mg",
      "mg_levels": {
          "ksp_type": "gmres",
          "ksp_max_it": 3,
          "ksp_convergence_test": "skip",
          "pc_type": "python",
          "pc_python_type": "firedrake.ASMVankaPC",
          "pc_vanka_construct_dim": 0,
          "pc_vanka_sub_sub_pc_type": "lu",
          "pc_vanka_sub_sub_pc_factor_mat_solver_type": "umfpack",
          "pc_vanka_exclude_subspaces": exclude_inds},
      "mg_coarse": {
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
          "mat_mumps_icntl_14": 200}
  }

  stepper = TimeStepper(F, butcher_tableau, t, dt, upT, bcs=bcs,
                        nullspace=nullspace, solver_parameters=params)

Now that the stepper is set up, let's run over many time steps::


  plot_freq = 10
  Ts = []
  cur_step = 0
  while float(t) < 1.0:
      print(f"t = {float(t)}")
      stepper.advance()

      t.assign(float(t) + float(dt))
      cur_step += 1
      if cur_step % plot_freq == 0:
          Ts.append(upT.subfunctions[2].copy(deepcopy=True))

     
  nsp = 16
  fn_plotter = FunctionPlotter(msh, num_sample_points=nsp)
  fig, axes = plt.subplots()
  axes.set_aspect('equal')
  Tzero = Function(Q)
  colors = tripcolor(Tzero, num_sample_points=nsp, vmin=0, vmax=1, axes=axes)
  fig.colorbar(colors)

  def animate(q):
      colors.set_array(fn_plotter(q))

The last step is to make the animation and save it to a file. ::

  interval = 1e3 * plot_freq * float(dt)
  animation = FuncAnimation(fig, animate, frames=Ts, interval=interval)
  try:
      animation.save("benard_temp.mp4", writer="ffmpeg")
  except:
      print("Failed to write movie! Try installing `ffmpeg`.")
