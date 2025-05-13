Shape optimization
==================

Shape optimization is about modifying the shape of a domain so that an
objective function is minimized. In this demo, we consider an objective
function constrained to a boundary value problem and implement a simple
mesh-moving shape optimization strategy using Firedrake and pyadjoint.  This
tutorial was contributed by `Alberto Paganini <mailto:apaganini@le.ac.uk>`__
and was written during the `ccp-dcm hackaton
<https://ccp-dcm.github.io/exeter_hackathon>`__ at Dartington Hall.

Let

.. math::

   J(\Omega) = \int_\Omega \big(u(\mathbf{x}) - u_t(\mathbf{x})\big)^2 \,\mathrm{d}\mathbf{x}\,.

where :math:`u:\mathbb{R}^2\to\mathbb{R}` is the solution to the scalar
boundary value problem

.. math::

    -\Delta u = 4 \quad \text{in }\Omega\,, \qquad u = 0 \quad \text{on } \partial\Omega


and :math:`u_t:\mathbb{R}^2\to\mathbb{R}` is a target function. In particular,
we consider

.. math::

    u_t(x,y) = 1.21 - (x - 0.5)^2 - (y - 0.5)^2\,.

Beside the empty set, the domain that minimizes :math:`J(\Omega)` is a disc of
radius :math:`1.1` centered at :math:`(0.5,0.5)`.

We can now proceed to set up the problem. We import firedrake and pyadjoint and
choose an initial guess (in this case, a unit disc centred at the origin)::

  from firedrake import *
  from firedrake.adjoint import *
  mesh = UnitDiskMesh(refinement_level=3)

Then, we :ref:`start annotating <adjoint-taping>` and turn the mesh coordinates into a control variable::

  continue_annotation()
  Q = mesh.coordinates.function_space()
  dT = Function(Q)
  mesh.coordinates.assign(mesh.coordinates + dT)

We can now implement the target function::

  x, y = SpatialCoordinate(mesh)
  u_t = Constant(1.21) - (x - Constant(0.5))**2 - (y - Constant(0.5))**2

solve the weak form of the boundary value problem::

  V = FunctionSpace(mesh, "CG", 1)
  u = Function(V, name='state')
  v = TestFunction(V)
  F = (dot(grad(u), grad(v)) - 4 * v) * dx
  bcs = DirichletBC(V, Constant(0.), "on_boundary")
  solve(F == 0, u, bcs=bcs)

and evaluate the objective function::

  J = assemble((u - u_t)**2*dx)

We now turn the objective function into a reduced function so that pyadjoint
(and UFL shape differentiation capability) can automatically compute shape
gradients, that is, directions of steepest ascent::

  Jred = ReducedFunctional(J, Control(dT))
  stop_annotating()

We now have all the ingredients to implement a basic steepest descent shape
optimization algorithm with fixed step size.::

  File = VTKFile("shape_iterates.pvd")
  for ii in range(30):
    print("J(ii =", ii, ") =", Jred(dT))
    File.write(mesh.coordinates)

    # compute the gradient (steepest ascent)
    opts = {"riesz_representation": "H1"}
    gradJ = Jred.derivative(options=opts)

    # update domain
    dT -= 0.2*gradJ

  File.write(mesh.coordinates)
  print("J(final) =", Jred(dT))

.. only:: html

  .. container:: youtube

    .. vimeo:: 1083822714?loop=1
       :width: 600px


**Remark:** mesh-moving shape optimization can lead to mesh tangling, which
invalidates finite element computations. For faster and more robust shape
optimization, we recommend using Firedrake's shape optimization toolbox
`Fireshape <https://github.com/fireshape/fireshape>`__.

A python script version of this demo can be found :demo:`here <shape_optimization.py>`.
