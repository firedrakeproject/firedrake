Nonlinear preconditioning applied to the Allen-Cahn equation
============================================================

Contributed by `Daniel Shapero <https://psc.apl.uw.edu/people/investigators/daniel-shapero/>`_
and `Josh Hope-Collins <https://profiles.imperial.ac.uk/joshua.hope-collins13>`_.

In the other demonstrations on nonlinear PDE, we've used Newton's method to
solve finite- dimensional systems of nonlinear equations. Suppose we want to
solve a system

.. math::

  F(u) = 0.

At each step of Newton's method, we first compute a *search direction*
:math:`v` by solving the linear system

.. math::

  dF(u)v = -F(u).

We can then use a line search along :math:`v` to obtain the next candidate
solution. When we talked about preconditioning, we have always meant applying
a preconditioner to the linear system for the search direction. A linear
preconditioner transforms a linear system into an equivalent one that is
easier to solve.

For some highly nonlinear problems, however, Newton's method can stagnate or
fail to converge, even with globalization strategies such as a line search.
For example, if :math:`dF(u)` behaves badly, we might not be able to compute a
search direction at all.

An alternative strategy is to use *nonlinear preconditioning*, abbreviated as
NPC in the following. Nonlinear preconditioning does the same thing as linear
preconditioning: it transforms the problem into an equivalent one that is
(hopefully) easier to solve. Rather than work on the linear system for the
search direction, however, NPC works directly on the nonlinear system itself.
Typically NPC is applied within a higher-level nonlinear solver. For example,
there is a nonlinear analogue of GMRES. For a review of nonlinear solver and
preconditioning strategies beyond Newton, see :cite:`brune2015composing`.

Here we will demonstrate how to use nonlinear preconditioning for the steady-
state Allen-Cahn equation

.. math::

  F(u) = -\epsilon\Delta u + u^3 - u = 0

on the unit interval with Dirichlet boundary conditions :math:`u(0) = -1` and
:math:`u(1) = +1`. The Jacobian of this residual is indefinite wherever
:math:`|u| < 1/\sqrt{3}`. Newton's method can fail to compute a search
direciton when starting from an initial guess that crosses this region.

This demo is adapted from this `Chebfun example`_.

.. _Chebfun example: https://www.chebfun.org/examples/ode-nonlin/AllenCahn.html

::

  import numpy as np
  from firedrake import *

Here we use a domain of length 10, a small diffusion coefficient of 0.003,
and an initial guess that ramps from +1 at the left-hand boundary to -1 at
the right.

::

  nx = 128
  lx = 10.0
  eps = Constant(3e-3)

  mesh = IntervalMesh(nx, lx)
  Q = FunctionSpace(mesh, "CG", 1)

  x, = SpatialCoordinate(mesh)
  u_1 = Constant(1)
  u_2 = Constant(-1)
  Lx = Constant(lx)
  initial_guess = (1 - x / Lx) * u_1 + x / Lx * u_2

  bcs = [DirichletBC(Q, u_1, [1]), DirichletBC(Q, u_2, [2])]

  u = Function(Q)
  u.interpolate(initial_guess)

In order to have a good baseline to compare against, we want to use as good
a nonlinear solution strategy as possible. Here we use Newton's method with
the *critical point* line search, which is specially adapted for problems
like Allen-Cahn which can be derived from minimizing some free energy. Even
with 10 iterations of a line search, Newton's method will fail. You can try
other line search methods (secant, backtracking, etc.) and find similar
outcomes. If you pump the number of line search iterations way up, you can
make Newton's method converge... to the wrong solution!

::

  v = TestFunction(Q)
  F = (eps * inner(grad(u), grad(v)) + inner(u**3 - u, v)) * dx

  problem = NonlinearVariationalProblem(F, u, bcs)

  newton_parameters = {
      "snes_type": "newtonls",
      "snes_monitor": None,
      "snes_converged_reason": None,
      "snes_linesearch_type": "cp",
      "snes_linesearch_max_it": 10,
      "snes_linesearch_monitor": None,
  }
  solver = NonlinearVariationalSolver(problem, solver_parameters=newton_parameters)
  try:
      solver.solve()
  except ConvergenceError as err:
      print(err)
      print("--------------------------")
      print("Solver failed to converge!")
      print("Resetting `u`")
      u.interpolate(initial_guess)

The residual decreases at first but eventually diverges. Here's some of the PETSc
log output:

.. code-block:: console

    $ python nonlinear_pc_allen_cahn.py
      0 SNES Function norm 2.439229081145e-01
      1 SNES Function norm 5.663027251974e+01
      2 SNES Function norm 1.667065795962e+01
      3 SNES Function norm 4.864679262673e+00
      4 SNES Function norm 1.388964354434e+00
      5 SNES Function norm 3.744562754242e-01
      6 SNES Function norm 8.820230192544e-02
      7 SNES Function norm 3.682817028776e-02
      8 SNES Function norm 5.405858541080e-02
      9 SNES Function norm 5.368629859638e-02
       ...
     26 SNES Function norm 5.280237868253e-02
     27 SNES Function norm 1.098413882561e+00
     28 SNES Function norm 1.082944223664e+00
     29 SNES Function norm 6.520280025999e+03
      Nonlinear firedrake_0_ solve did not converge due to DIVERGED_DTOL iterations 29

In other scenarios, the solver doesn't explode as dramatically but rather
stagnates and exceeds the number of allowable Newton iterations.

To salvage the wreck, we'll use nonlinear preconditioning. Here we use
the simplest solution strategy possible: preconditioned nonlinear
Richardson iterations.
The idea is that if :math:`F(u)=0` is too difficult to solve, we can
instead solve an auxiliary problem :math:`G(u; \ldots) = 0` which is
*nearby* to :math:`F`, and use that solution to iterate towards a solution
for :math:`F(u)=0`. In almost every scenario, the auxiliary problem uses
the previous iterate :math:`u_{k}`.
At each iteration :math:`k`, we solve the following system for the next
iterate :math:`u_{k+1}` using the value of the current iterate :math:`u_k`:

.. math::

   G(u_{k+1}; u_{k}) = G(u_{k}; u_{k}) - F(u_{k}).

We note a few properties of this iteration. First, if
:math:`F(u_{*})=0` then :math:`u_{*}` is a fixed point of the iteration.
Second, we never have to solve :math:`F(u)=0`, we only have to evaluate
its residual at a given state. Third, we can hope that if :math:`G` is
*close enough* to :math:`F` then the iteration will converge,
although proving this is more difficult than in the linear case.

Our approach for defining :math:`G` here is to hold back the linear part of
the reaction term in the Allen-Cahn equation to the value at the previous
iteration. In other words, at each step, we define the PDE

.. math::

  G(u; u_k) = -\epsilon\Delta u + u^3 - u_k = 0.

The Jacobian for this problem w.r.t. :math:`u` is symmetric and positive-
definite. We have good guarantees about the convergence of Newton's method
in that case, so we can reuse the solver parameters that we tried the first
time around for this inner problem.

Here we define a custom nonlinear preconditioner which inherits from
:class:`~firedrake.preconditioners.auxiliary_snes.AuxiliaryOperatorSNES`.
Similar to ``AuxiliaryOperatorPC``, we have to implement the ``form`` method.
This method returns (1) a residual form :math:`G(u; u_{k})\cdot v` where
:math:`v` is the test function and (2) the boundary conditions for this
sub-problem. The arguments that it takes are: the PETSc SNES object; the value
:math:`u_k` of the solution at the previous iteration; the current value
:math:`u` to be solved for; and a test function.

::

  class AllenCahnAuxSNES(firedrake.AuxiliaryOperatorSNES):
      def form(self, snes, u_k, u, v):
          F, bcs = super().form(snes, u_k, u, v)
          return (eps * inner(grad(u), grad(v)) + inner(u**3 - u_k, v)) * dx, bcs

The contract for the ``form`` method requires it to supply the boundary
conditions. We could have obtained the boundary conditions by pulling them out
of the global context. That will work for this particular set of solvers but
can create problems for others, for example when using the multigrid method.
Instead, we've opted to call the parent class's ``form`` method, which will
return the original variational form and boundary conditions. We then discard
the original variational form ``F``, which we aren't using here. For other
nonlinear preconditioners, we might instead be building the preconditioning
form by modifying ``F``.

We now set ``-snes_type nrichardson`` for nonlinear Richardson iterations,
and specify the additional parameters for the nonlinear preconditioner under
the key ``"npc"``. Firstly we need to specify our python SNES type, and then
in the ``"aux"`` key we specify the parameters to actually solve :math:`G` -
here we use Newton iterations.

::

  solver_parameters = {
      "snes_type": "nrichardson",
      "snes_monitor": None,
      "snes_converged_reason": None,
      "npc": {
          "snes_type": "python",
          "snes_python_type": f"{__name__}.AllenCahnAuxSNES",
          "aux": newton_parameters,
      },
  }

  solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
  solver.solve()

Now we actually converge! The convergence on nonlinear Richardson iterations is
usually linear, as opposed to the quadratic convergence of Newton, but with
suitable preconditioning they will usually have a wider basin of convergence than
Newton. From the log output, we can observe that the residual is decreasing by a
factor of 2 or more at each iteration.

.. code-block:: console

      0 SNES Function norm 2.439229081145e-01
      1 SNES Function norm 2.405339859939e-01
      2 SNES Function norm 1.540442351803e-01
      3 SNES Function norm 7.166137071498e-02
      4 SNES Function norm 2.854773301463e-02
      5 SNES Function norm 1.070593887487e-02
      6 SNES Function norm 3.943106335233e-03
      7 SNES Function norm 1.451230558440e-03
       ...
     18 SNES Function norm 3.491990058865e-08
     19 SNES Function norm 1.349411703695e-08
     20 SNES Function norm 5.217958176918e-09
     21 SNES Function norm 2.018693509573e-09
      Nonlinear firedrake_1_ solve converged due to CONVERGED_FNORM_RELATIVE iterations 21

We've chosen to use nonlinear Richardson here because it's the simplest scheme
with the fewest knobs to turn. There are more sophisticated strategies which can
offer a faster convergence rate. For example, you can run this demo again with
`"snes_type": "ngmres"` to use a nonlinear variant of the generalised minimum
residual algorithm. NGMRES with the default options cuts the number of
iterations in half for this problem. But it has more algorithmic knobs to turn
and those can require tweaking depending on the problem.

The Allen-Cahn equation is derivable through minimization of the free energy
functional

.. math::

  E(u) = \int_\Omega\left(\frac{\epsilon}{2}|\nabla u|^2 + \frac{1}{4}(1 - u^2)^2\right)dx.

To close, let's evaluate the free energy at the starting guess and at the
computed solution.

::

  E = (0.5 * eps * inner(grad(u), grad(u)) + 0.25 * (1 - u**2) ** 2) * dx
  E_initial = firedrake.assemble(firedrake.replace(E, {u: initial_guess}))
  E_final = firedrake.assemble(E)
  print(f"Initial free energy: {E_initial.real:0.04f}")
  print(f"Final:               {E_final.real:0.04f}")

.. code-block:: console

    Initial free energy: 1.3339
    Final:               0.0534

This demo can be found as a script in :demo:`nonlinear_pc_allen_cahn.py <nonlinear_pc_allen_cahn.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
