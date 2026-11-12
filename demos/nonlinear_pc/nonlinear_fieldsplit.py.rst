Nonlinear fieldsplit for coupled reaction-diffusion equations
==============================================================

Contributed by Robert C. Kirby
and `Josh Hope-Collins <https://profiles.imperial.ac.uk/joshua.hope-collins13>`_.

For a linear system with several fields, the ``fieldsplit``
preconditioner solves with one block of the matrix at a time and
combines the results. The nonlinear analogue does the same thing with
the nonlinear equations themselves: it solves the equations for each
field (or group of fields) in turn, with the other fields held fixed,
and then combines the updates. The sweep can apply the updates all at
once (a nonlinear block Jacobi iteration) or one after another, so that
each split sees the latest values of the fields that came before it (a
nonlinear block Gauss-Seidel iteration). Firedrake provides this as the
python SNES type :class:`~firedrake.preconditioners.fieldsplit_snes.FieldsplitSNES`.

This demo shows how to use ``FieldsplitSNES`` as a solver by itself
and as a nonlinear preconditioner for an accelerated nonlinear solver.
For a review of these methods, see :cite:`brune2015composing`. The
demo on :doc:`nonlinear preconditioning for the Allen-Cahn equation
<nonlinear_pc_allen_cahn.py>` gives an introduction to
nonlinear preconditioning.

The problem
-----------

We solve two coupled semilinear diffusion-reaction equations on the
unit square :math:`\Omega`,

.. math::

   -\Delta u_1 + u_1^3 + \beta u_1 u_2^2 &= f_1, \\
   -\Delta u_2 + u_2^3 + \beta u_1^2 u_2 &= f_2,

with :math:`u_1 = u_2 = 0` on :math:`\partial\Omega`. The parameter
:math:`\beta \geq 0` sets how strongly the equations are coupled. When
:math:`\beta = 0` the equations are independent.

These equations are the Euler-Lagrange equations of the energy

.. math::

   E(u_1, u_2) = \int_\Omega \frac{1}{2}|\nabla u_1|^2
   + \frac{1}{2}|\nabla u_2|^2
   + \frac{1}{4}u_1^4 + \frac{1}{4}u_2^4
   + \frac{\beta}{2}u_1^2 u_2^2
   - f_1 u_1 - f_2 u_2 \, \mathrm{d}x.

The Hessian of the reaction part of the integrand has the determinant

.. math::

   3\beta(u_1^4 + u_2^4) + (9 - 3\beta^2)u_1^2 u_2^2.

This determinant is not negative for any :math:`u_1, u_2` when
:math:`0 \leq \beta \leq 3`, so the energy is strictly convex and the
problem has a unique solution. Strict convexity also tells us what a
nonlinear block Gauss-Seidel iteration does for this problem. When
:math:`u_2` is held fixed, the first equation is the condition for
:math:`u_1` to minimize :math:`E` over all choices of :math:`u_1`. A
Gauss-Seidel sweep is then a block coordinate descent method: it
minimizes the energy first in :math:`u_1` and then in :math:`u_2`. Each
of these minimizations can only decrease the energy, so the energy
decreases at every sweep. Each subproblem is a strictly monotone
semilinear equation, for which Newton's method works well.

We first import Firedrake and make the mesh and the mixed function
space.

.. code-block:: python

  from firedrake import *

  mesh = UnitSquareMesh(32, 32)
  V = FunctionSpace(mesh, "CG", 1)
  W = V * V

We use the strongest coupling for which the energy is convex, and two
forcing terms with different shapes so that the solution is not
symmetric in :math:`u_1` and :math:`u_2`.

.. code-block:: python

  x, y = SpatialCoordinate(mesh)
  beta = Constant(3)
  f1 = Constant(200)
  f2 = 800*x*sin(pi*x)*sin(pi*y)

The residual is the weak form of the two equations, and there is a
homogeneous Dirichlet condition on each field. We put the code that
makes the solver into a function because we will use several different
sets of solver parameters.

.. code-block:: python

  w = Function(W)
  u1, u2 = split(w)
  v1, v2 = TestFunctions(W)

  F = (inner(grad(u1), grad(v1))*dx
       + inner(u1**3 + beta*u1*u2**2 - f1, v1)*dx
       + inner(grad(u2), grad(v2))*dx
       + inner(u2**3 + beta*u1**2*u2 - f2, v2)*dx)

  bcs = [DirichletBC(W.sub(i), 0, "on_boundary") for i in range(2)]

  problem = NonlinearVariationalProblem(F, w, bcs)


  def solve_with(parameters, monitor=None):
      w.zero()
      solver = NonlinearVariationalSolver(
          problem, solver_parameters=parameters)
      if monitor is not None:
          solver.snes.setMonitor(monitor)
      solver.solve()
      return solver.snes.getIterationNumber()

For reference, Newton's method on the coupled system converges in a few
iterations.

.. code-block:: python

  newton = {
      "snes_type": "newtonls",
      "snes_rtol": 1e-8,
      "mat_type": "aij",
      "ksp_type": "preonly",
      "pc_type": "lu",
  }

  print(f"Newton: {solve_with(newton)} iterations")

Nonlinear Gauss-Seidel
----------------------

To use nonlinear block Gauss-Seidel, we set the SNES type to
``python`` and the python type to ``firedrake.FieldsplitSNES``, and
choose the ``multiplicative`` fieldsplit type. By default there is one
split for each field of the mixed space, and the splits are solved in
field order. The ``-snes_fieldsplit_%d_fields`` options change how the
fields are grouped into splits and the order of the splits, in the
same way as for the linear ``fieldsplit`` preconditioner.

Each split is solved with its own
:class:`~firedrake.variational_solver.NonlinearVariationalSolver`. The
options for split ``i`` have the prefix ``fieldsplit_i_``. The options
with the prefix ``fieldsplit_`` are defaults for all of the splits, so
here every split uses Newton's method with a direct linear solver. An
option that is given for one split, for example
``"fieldsplit_1": {"snes_rtol": 1e-10}``, takes priority over the
default for that split only.

.. code-block:: python

  def fieldsplit(fieldsplit_type):
      return {
          "snes_type": "python",
          "snes_python_type": "firedrake.FieldsplitSNES",
          "snes_fieldsplit_type": fieldsplit_type,
          "fieldsplit": newton,
      }

  ngs = {
      **fieldsplit("multiplicative"),
      "snes_rtol": 1e-8,
      "snes_max_it": 100,
  }

To see that each sweep decreases the energy, we add a monitor to the
outer SNES that evaluates the energy at the current iterate.

.. code-block:: python

  w_k = Function(W)
  u1_k, u2_k = split(w_k)
  energy = (0.5*inner(grad(u1_k), grad(u1_k))
            + 0.5*inner(grad(u2_k), grad(u2_k))
            + 0.25*u1_k**4 + 0.25*u2_k**4
            + 0.5*beta*u1_k**2*u2_k**2
            - f1*u1_k - f2*u2_k)*dx


  def energy_monitor(snes, its, fnorm):
      with w_k.dat.vec_wo as wvec:
          snes.getSolution().copy(wvec)
      print(f"  {its:2d}  residual {fnorm:.3e}  energy {assemble(energy):.8e}")


  print("Nonlinear Gauss-Seidel:")
  its = solve_with(ngs, monitor=energy_monitor)
  print(f"Nonlinear Gauss-Seidel: {its} iterations")

The residual decreases linearly, and the energy decreases at every
sweep, as the convexity argument predicts.

.. code-block:: console

    Nonlinear Gauss-Seidel:
       0  residual 8.981e+00  energy 0.00000000e+00
       1  residual 2.890e+00  energy -6.45687623e+02
       2  residual 1.633e+00  energy -7.03207311e+02
       3  residual 8.559e-01  energy -7.21415730e+02
       4  residual 3.852e-01  energy -7.26118950e+02
       5  residual 1.625e-01  energy -7.27067166e+02
       ...
      20  residual 2.561e-07  energy -7.27273908e+02
      21  residual 1.050e-07  energy -7.27273908e+02
      22  residual 4.304e-08  energy -7.27273908e+02
    Nonlinear Gauss-Seidel: 22 iterations

Nonlinear Jacobi
----------------

The ``additive`` fieldsplit type gives nonlinear block Jacobi. Every
split is solved with the other fields at their values from the
previous iteration, so the splits are independent of each other.
This lets the splits be solved at the same time, but each sweep uses
less up-to-date information.

.. code-block:: python

  jacobi = {
      **fieldsplit("additive"),
      "snes_rtol": 1e-8,
      "snes_max_it": 100,
  }

  print(f"Nonlinear Jacobi: {solve_with(jacobi)} iterations")

As in the linear case, Jacobi needs about twice as many iterations as
Gauss-Seidel. The energy argument does not apply to Jacobi, because a
Jacobi sweep changes both fields at the same time.

Nonlinear preconditioning
-------------------------

As :math:`\beta` increases, the fields are more strongly coupled and
nonlinear Gauss-Seidel needs more sweeps. We can accelerate it by
using one sweep as a nonlinear preconditioner for an outer method such
as Anderson acceleration or nonlinear GMRES. The parameters for the
nonlinear preconditioner have the prefix ``npc_``. We set
``snes_max_it`` to 1 and skip the convergence test, so that the
preconditioner does exactly one sweep each time it is applied.

.. code-block:: python

  one_sweep = {
      **fieldsplit("multiplicative"),
      "snes_max_it": 1,
      "snes_convergence_test": "skip",
  }

  anderson = {
      "snes_type": "anderson",
      "snes_rtol": 1e-8,
      "snes_max_it": 100,
      "npc": one_sweep,
  }

  ngmres = {
      "snes_type": "ngmres",
      "snes_rtol": 1e-8,
      "snes_max_it": 100,
      "npc": one_sweep,
  }

  print(f"Anderson + Gauss-Seidel: {solve_with(anderson)} iterations")
  print(f"NGMRES + Gauss-Seidel: {solve_with(ngmres)} iterations")

The iteration counts are:

.. code-block:: console

    Newton: 8 iterations
    Nonlinear Gauss-Seidel: 22 iterations
    Nonlinear Jacobi: 44 iterations
    Anderson + Gauss-Seidel: 10 iterations
    NGMRES + Gauss-Seidel: 13 iterations

Both accelerated methods need fewer than two thirds of the sweeps that
nonlinear Gauss-Seidel uses alone. Newton's method on the coupled
system needs fewer iterations still for this small problem. The
fieldsplit methods are useful when the coupled problem is hard to solve
but each field is easy to solve by itself, for example because a good
solver is already available for each separate equation. Because each
split uses a standard ``NonlinearVariationalSolver``, the split solvers
can use any of the solver options that are available in Firedrake,
including another ``FieldsplitSNES`` for a split that contains more
than one field.

Try changing ``beta`` to see how the coupling strength changes the
iteration counts. With ``beta = 0`` the equations are independent, and
one sweep of either fieldsplit method gives the solution.

This demo can be found as a script in
:demo:`nonlinear_fieldsplit.py <nonlinear_fieldsplit.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
