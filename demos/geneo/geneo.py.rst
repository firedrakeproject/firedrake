Using the GenEO preconditioner
==============================

.. rst-class:: emphasis

   This demo was developed with help from `Frédéric Nataf
   <https://www.ljll.math.upmc.fr/nataf/>`__.

We demonstrate usage of the GenEO (generalised eigenproblems in the
overlap) preconditioner of :cite:`Spillane:2014`. This relies on the
`geneo4PETSc <https://github.com/geneo4PETSc/geneo4PETSc/>`__
library which you can install with ``firedrake-install
--with-geneo`` or ``firedrake-update --with-geneo``.

The GenEO preconditioner is a two-level Schwarz method for symmetric
positive definite systems. It handles strong coefficient
heterogeneities by automatically constructing an appropriate coarse
space.

As usual, we begin by importing Firedrake, and defining a mesh.

::

   from firedrake import *
   mesh = UnitSquareMesh(50, 50, distribution_parameters={"overlap_type":
                                                          (DistributedMeshOverlapType.NONE, 0)})

.. note::

   Right now, the implementation can only handle meshes distributed
   without any overlap. This restriction means we only support cell
   integrals ``dx`` and exterior facet integrals ``dS``. This
   restriction will be lifted in the future.

If Firedrake was not installed with the geneo4PETSc library, then this
demo will not run, so let's check that here:

::

   from firedrake.preconditioners.geneoimpl import register

   try:
       register()
   except NotImplementedError:
       import sys
       warning("Firedrake not built with geneo4PETSc support (try firedrake-update --with-geneo)")
       sys.exit(0)

We will solve the problem, find :math:`u \in V` such that:

.. math::

   \int \epsilon \nabla u \cdot \nabla v\,\text{d}x &= \int v
   \,\text{d}x \quad \forall v \in V, \\
   u &= 0 \quad \text{on}\ \Gamma_D.

Where :math:`\Gamma_D` is the boundary :math:`y = 1`, and we apply
homogeneous Neumann conditions on the other three sides.

The coefficient :math:`\epsilon` is chosen as:

.. math::

   \epsilon(x, y) = \begin{cases}
                        10^6 &y \in (0.2, 0.4)\\
                        10^5 &y \in (0.6, 0.7)\\
                        1 &\text{otherwise}
                      \end{cases}

We will choose a space of piecewise linear functions for the solution
and test space
::
 
   V = FunctionSpace(mesh, "P", 1)
  
   u = TrialFunction(V)
   v = TestFunction(V)

and create an expression for :math:`\epsilon`, using UFL's :func:`~ufl.conditional`
::
  
   x, y = SpatialCoordinate(mesh)
  
   eps = conditional(And(y > 0.2, y < 0.4), Constant(1e6),
                     conditional(And(y > 0.6, y < 0.8), Constant(1e5),
                                 Constant(1)))
  
   a = (eps*dot(grad(u), grad(v)))*dx
  
   bcs = DirichletBC(V, 0, 4)

Now we have set up the problem, we just need to create a discrete
function for our solution, and to solve the problem.
::

   uh = Function(V)
   solve(a == v*dx, uh, bcs=bcs, options_prefix="",
         solver_parameters={"mat_type": "matfree",
                            "pc_type": "python",
                            "pc_python_type": "firedrake.GenEOPC"})
  
   uh.rename("Solution")
   File("solution.pvd").write(uh)


This demo can be found as a script in `geneo.py <geneo.py>`__.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
