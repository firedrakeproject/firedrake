Solving Poisson's equation by matrix-free methods
-------------------------------------------------

Yes, we're doing the dumbest possible equation, showing
how we can do this by matrix-free unassembled CG

.. math::
   -\Delta u = f
   u = 0 \ \textrm{on} \Gamma

This is just the Python module that drives the simulation.
The real meat is in the other files in this directory,
including a modified nonlinear variational solver that
doesn't call firedrake directly but creates a Python PETSc
matrix that implements multiplication by assembling a 1-form
and also contains pointers to UFL so that user-defined solvers,
implemented typically as preconditioners, can do what they think
is right.

Let's get started::

  from firedrake import *

The following two modules include the new bits that implement our magic::
  
<<<<<<< HEAD
  from firedrake.frankensolve import FrankenSolver
=======
  from uflmat import UFLMatrix
  import nlvs
>>>>>>> Demo showing how matrix-free can be done, with a fallback of assembling matrices inside a custom PC

This is firedrake boilerplate that we all know and love::
  
  M = UnitSquareMesh(16, 16)
  V = FunctionSpace(M, "CG", 1)

  u = TrialFunction(V)
  v = TestFunction(V)

  a = inner(grad(u), grad(v)) * dx

  bcs = [DirichletBC(V, Constant(0.0), (1,2,3,4))] 

  L = v*dx

  u0 = Function(V)

  prob = LinearVariationalProblem(a, L, u0, bcs=bcs)

Now, note that this is the new variational solver in the present
director and not the standard Firedrake one::
  
  solver = FrankenSolver(prob, options_prefix='')

  solver.solve()

  File("poisson.pvd").write(u0)

To see this code in action, one can run python poisson.py.rst -options_file=opts.txt or any of the other .txt files in this directory
