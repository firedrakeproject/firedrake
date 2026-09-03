Adaptive Multigrid Methods
==========================


Contributed by Anurag Rao.

The purpose of this demo is to show how to use Firedrake's multigrid solver on a hierarchy of adaptively refined Netgen meshes.
A :func:`MeshHierarchy <firedrake.MeshHierarchy>` is not restricted to uniform refinement: the same object records the parent child relations between adaptively refined meshes, and grows a level at a time as the solution is resolved.
We will first have a look at how to construct such a hierarchy from Netgen meshes, then we will consider a solution to the Poisson problem on an L-shaped domain, and finally we will use the hierarchy to construct a scalable solver.
We begin by importing the necessary libraries ::

   from firedrake import *
   from netgen.occ import *
   import numpy

Constructing the Mesh Hierarchy
-------------------------------
We first must construct the domain over which we will solve the problem. For a more comprehensive demo on how to use Open Cascade Technology (OCC) and Constructive Solid Geometry (CSG),
see `Netgen integration in Firedrake <netgen_mesh.py>`_. 
We begin with the L-shaped domain, which we build as the union of two rectangles: ::
  
   rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
   rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
   L = rect1 + rect2

   geo = OCCGeometry(L, dim=2)
   ngmsh = geo.GenerateMesh(maxh=0.5)
   mesh = Mesh(ngmsh)

It is important to convert the initial Netgen mesh into a Firedrake mesh before constructing the :func:`MeshHierarchy <firedrake.MeshHierarchy>`. To call the constructor to the hierarchy, we must pass the initial mesh. Our initial mesh looks like this:

.. figure:: initial_mesh.png
   :align: center
   :alt: Initial mesh.

We initialize the :func:`MeshHierarchy <firedrake.MeshHierarchy>` here. The default of zero uniform refinement levels gives a hierarchy holding just the initial mesh, which we will grow adaptively below; passing a positive number instead would start us off with that many uniformly refined levels, and the adaptive levels would stack on top of them just the same: ::
  
   mh = MeshHierarchy(mesh)

Poisson Problem
---------------
Now we can define a simple Poisson problem

.. math::

   - \nabla^2 u = f \text{ in } \Omega, \quad u = 0 \text{ on } \partial \Omega.

Our approach strongly follows the similar problem in this `lecture course <https://github.com/pefarrell/icerm2024>`_. We define the function ``solve_poisson``. The first lines correspond to finding a solution in the CG1 space. The right-hand side is set to be the constant function equal to 1. Since we want Dirichlet boundary conditions, we construct the :class:`DirichletBC <firedrake.DirichletBC>` object and apply it to the entire boundary: ::

   def solve_poisson(mesh, params):
      V = FunctionSpace(mesh, "CG", 1)
      v = TestFunction(V)
      u = TrialFunction(V)
      uh = Function(V, name="solution")
      bcs = [DirichletBC(V, 0, "on_boundary")]
      f = Constant(1)

      a = inner(grad(u), grad(v))*dx
      L = inner(f, v)*dx

      problem = LinearVariationalProblem(a, L, uh, bcs)
      solver = LinearVariationalSolver(problem, solver_parameters=params)
      solver.solve()

      its = solver.snes.getLinearSolveIterations()
      return uh, its

To use the hierarchy in a multigrid solver, we just set the usual multigrid solver parameters. Since we are using linear Lagrange elements, we will employ Jacobi as the multigrid relaxation, which we define with ::

   solver_params = {
      "mat_type": "matfree",
      "ksp_type": "cg",
      "pc_type": "mg",
      "mg_levels": {
          "ksp_type": "chebyshev",
          "ksp_max_it": 1,
          "pc_type": "jacobi",
      },
      "mg_coarse": {
          "mat_type": "aij",
          "pc_type": "lu",
      },
   }

Alternatively for high-order CG elements, it is recommended to use patch relaxation
to achieve degree-independent multigrid convergence.
For more information
see :doc:`Using patch relaxation for multigrid <poisson_mg_patches.py>`. 
The initial solution is shown below.

.. figure:: solution_l1.png
   :align: center
   :alt: Initial Solution from multigrid with initial mesh.


Adaptive Mesh Refinement
------------------------
In this section we will discuss how to adaptively refine select elements and add the newly refined mesh into the hierarchy.
For this problem, we will be using the Babuška-Rheinbolt a-posteriori estimate for an element:

.. math::
   \eta_K^2 = h_K^2 \int_K \| f + \nabla^2 u_h \|^2 \mathrm{d}x + \frac{h_K}{2} \int_{\partial K \setminus \partial \Omega} \left[[ \nabla u_h \cdot \mathbf{n} \right]]^2 \mathrm{d}s,

where :math:`K` is the element, :math:`h_K` is the diameter of the element, :math:`\mathbf{n}` is the outward-facing normal, and :math:`\left[[ \cdot \right]]` is the jump operator. The a-posteriori estimator is computed using the solution at the current level :math:`h`. Integrating over the domain and using the fact that the components of the estimator are piecewise constant on each cell, we can transform the above estimator into the variational problem

.. math::
   \int_\Omega \eta_K^2 q \,\mathrm{d}x = \int_\Omega \sum_K h_K^2 \int_K (f + \text{div} (\text{grad} u_h) )^2 \,\mathrm{d}x q \,\mathrm{d}x + \int_\Omega \sum_K \frac{h_K}{2} \int_{\partial K \setminus \partial \Omega} \left[[ \nabla u_h \cdot \mathbf{n} \right]]^2 \,\mathrm{d}s q \,\mathrm{d}x \quad \forall\, q \in \mathrm{DG}_0

Our approach will be to compute the estimator over all elements and selectively choose to refine only those that contribute most to the error. To compute the error estimator, we use the function below to solve the variational formulation of the error estimator. Since our estimator is a constant per element, we use a DG0 function space.  ::

   def estimate_error(mesh, uh):
       Q = FunctionSpace(mesh, "DG", 0)
       eta_sq = Function(Q)
       p = TrialFunction(Q)
       q = TestFunction(Q)
       f = Constant(1)
       residual = f + div(grad(uh))

       # symbols for mesh quantities
       h = CellDiameter(mesh)
       n = FacetNormal(mesh)
       vol = CellVolume(mesh)
   
       # compute cellwise error estimator
       a = inner(p, q / vol) * dx
       L = (inner(residual**2, q * h**2) * dx
            + inner(jump(grad(uh), n)**2, avg(q * h)) * dS
       )

       sp = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
       solve(a == L, eta_sq, solver_parameters=sp)

       # compute eta from eta^2
       eta = Function(Q).interpolate(sqrt(eta_sq))

       # compute estimate for error in energy norm
       with eta.dat.vec_ro as eta_:  
           error_est = eta_.norm()
       return eta, error_est

The next step is to choose which elements to refine. For this we use a simplified variant of Dörfler marking :cite:`Dorfler1996`:

.. math::
   \eta_K \geq \theta \text{max}_L \eta_L

The logic is to select an element :math:`K` to refine if the estimator is greater than some factor :math:`\theta` of the maximum error estimate of the mesh, where :math:`\theta` ranges from 0 to 1. In our code we choose :math:`\theta=0.5`.
With these helper functions complete, we can solve the system iteratively. In the max_iterations is the number of total levels we want to perform multigrid on. We will solve for 15 levels. At every level :math:`l`, we first compute the solution using multigrid up to level :math:`l`. We then use the current approximation of the solution to estimate the error across the mesh. Finally, we adaptively refine the mesh and repeat. ::

   theta = 0.5
   refinements = 15
   est_errors = []
   sqrt_dofs = []
   mg_iterations = []
   for level in range(refinements):
      print(f"level {level}")

      mesh = mh[-1]
      uh, its = solve_poisson(mesh, solver_params)
      VTKFile(f"output/adaptive_loop_{level}.pvd").write(uh)

      (eta, error_est) = estimate_error(mesh, uh)
      VTKFile(f"output/eta_{level}.pvd").write(eta)

      est_errors.append(error_est)
      sqrt_dofs.append(uh.function_space().dim() ** 0.5)
      mg_iterations.append(its)

      print(f"  ||u - u_h|| <= C * {error_est}")
      if len(est_errors) > 1:
         rates = -numpy.diff(numpy.log(est_errors)) / numpy.diff(numpy.log(sqrt_dofs))
         print(f"  rate = {rates[-1]}")

      if level != refinements - 1:
         mh.adapt(eta, theta)

To perform Dörfler marking, refine the current mesh, and add the mesh to the hierarchy, we use the :meth:`HierarchyBase.adapt <firedrake.HierarchyBase.adapt>` method. In this method the input is the recently computed error estimator ``eta`` and the Dörfler marking parameter ``theta``. The method always performs this on the current fine mesh in the hierarchy.
To mark cells by some other criterion, refine the finest mesh yourself and add the result, which is all that :meth:`HierarchyBase.adapt <firedrake.HierarchyBase.adapt>` does once it has marked:

.. code-block:: python

   mh.add_mesh(mh[-1].refine_marked_elements(markers))

Here ``markers`` is a DG0 function whose value on each cell is the number of times to refine it. If the mesh was instead produced by some procedure Firedrake cannot trace the parent child relations through, pass those cell maps to :meth:`HierarchyBase.add_mesh <firedrake.HierarchyBase.add_mesh>` explicitly.
The meshes now refine according to the error estimator. The error estimators at levels 3,5, and 15 are shown below. Zooming into the vertex of the L-shape at level 15 shows the error indicator remains strongest there. Further refinements will focus on that area.

+-------------------------------+-------------------------------+-------------------------------+
| .. figure:: eta_l3.png        | .. figure:: eta_l6.png        | .. figure:: eta_l15.png       |
|    :align: center             |    :align: center             |    :align: center             |
|    :height: 250px             |    :height: 250px             |    :height: 250px             |
|    :alt: Eta at level 3       |    :alt: Eta at level 6       |    :alt: Eta at level 15      |
|                               |                               |                               |
|    *Level 3*                  |    *Level 6*                  |    *Level 15*                 |
+-------------------------------+-------------------------------+-------------------------------+

The solutions at level 4 and 15 are shown below.

+------------------------------------+------------------------------------+
| .. figure:: solution_l4.png        | .. figure:: solution_l15.png       |
|    :align: center                  |    :align: center                  |
|    :height: 300px                  |    :height: 300px                  |
|    :alt: Solution, level 4         |    :alt: Solution, level 15        |
|                                    |                                    |
|    *MG solution at level 4*        |    *MG solution at level 15*       |
+------------------------------------+------------------------------------+

The convergence follows the expected optimal behavior: ::

   from matplotlib import pyplot as plt

   dofs = numpy.array(sqrt_dofs) ** 2
   opt_errors = est_errors[0] * (sqrt_dofs[0] / numpy.array(sqrt_dofs))
   plt.loglog(dofs, est_errors, '-o', markersize = 3, label="Estimated error")
   plt.loglog(dofs, opt_errors, '--', markersize = 3, label="Optimal convergence")
   plt.ylabel("Error estimate of the energy norm")
   plt.xlabel("Number of degrees of freedom")
   plt.legend()
   plt.savefig("output/adaptive_convergence.png")

.. figure:: adaptive_convergence.png
   :align: center
   :alt: Convergence of the error estimator.

Moreover, the multigrid iteration count is robust to the level of refinement ::

   print(" Level\t | Iterations")
   print("---------------------")
   for level, its in enumerate(mg_iterations):
       print(f"   {level}\t | {its}")

..

======== ================
 Level     Iterations
======== ================
   0	     2
   1	     8
   2	     8
   3	     7
   4	     7
   5	     7
   6	     7
   7	     7
   8	     7
   9	     7
   10	     7
   11	     7
   12	     7
   13	     7
   14	     7
======== ================

A runnable python version of this demo can be found :demo:`here<adaptive_multigrid.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
