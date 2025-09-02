Adaptive Multigrid Methods using the AdaptiveMeshHierarchy
=================================================

The purpose of this demo is to show how to use Firedrake's multigrid solver on a hierarchy of adaptively refined Netgen meshes.
We will first have a look at how to use the *AdaptiveMeshHierarchy* to construct the mesh hierarchy with Netgen meshes, then we will consider a solution to the Poisson problem on an L-shaped domain.
Finally, we will show how to use the *AdaptiveMeshHierarchy* and *AdaptiveTransferManager* as arguments to Firedrake solvers. The *AdaptiveMeshHierarchy* contains information of the mesh hierarchy and the parent child relations between the meshes.
The *AdaptiveTransferManager* deals with the transfer operator logic across any given levels in the hierarchy.
We begin by importing the necessary libraries ::

   from firedrake import *
   from netgen.occ import *
   from ngsPETSc import AdaptiveMeshHierarchy, AdaptiveTransferManager

Constructing the Mesh Hierarchy
---------------------------
We first must construct the domain over which we will solve the problem. For a more comprehensive demo on how to use Open Cascade Technology (OCC) and Constructive Solid Geometry (CSG), see `Netgen integration in Firedrake <https://www.firedrakeproject.org/demos/netgen_mesh.py.html>`_ . 
We begin with the L-shaped domain, which we build as the union of two rectangles: ::
  rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
  rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
  L = rect1 + rect2
  
  geo = OCCGeometry(L, dim=2)
  ngmsh = geo.GenerateMesh(maxh=0.5)
  mesh = Mesh(ngmsh)

It is important to convert the initial Netgen mesh into a Firedrake mesh before constructing the *AdaptiveMeshHierarchy*. To call the constructor to the hierarchy, we must pass the initial mesh. Our initial mesh looks like this:

.. figure:: initial_mesh.png
   :align: center
   :alt: Initial mesh.

We will also initialize the *AdaptiveTransferManager* here: ::
  
  amh = AdaptiveMeshHierarchy(mesh)
  atm = AdaptiveTransferManager()

Poisson Problem
-------------------------
Now we can define a simple Poisson problem

.. math::

   - \nabla^2 u = f \text{ in } \Omega, \quad u = 0 \text{ on } \partial \Omega

Our approach strongly follows the similar problem in this `lecture course <https://github.com/pefarrell/icerm2024>`_. We define the function ``solve_poisson``. The first lines correspond to finding a solution in the CG1 space. The variational problem is formulated with F, where f is the constant function equal to 1. Since we want Dirichlet boundary conditions, we construct the *DirichletBC* object and apply it to the entire boundary: ::

   def solve_poisson(mesh, params):
      V = FunctionSpace(mesh, "CG", 1)
      uh = Function(V, name="Solution")
      v = TestFunction(V)
      bc = DirichletBC(V, Constant(0), "on_boundary")
      f = Constant(1)
      F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx

      problem = NonlinearVariationalProblem(F, uh, bc)
      solver = NonlinearVariationalSolver(problem, solver_parameters=params)
      solver.set_transfer_manager(atm)
      solver.solve()
      return uh

Note the code after the construction of the *NonlinearVariationalProblem()*. To use the *AdaptiveMeshHierarchy* with the existing Firedrake solver, we have to set the *AdaptiveTransferManager* as the transfer manager of the multigrid solver.
For the parameters of the multigrid solver, we will be using patch relaxation, which we define with ::
   solver_params = {
            "mat_type": "matfree",
            "ksp_type": "cg",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                "pc_type": "python",
                "pc_python_type": "firedrake.PatchPC",
                "patch": {
                    "pc_patch": {
                        "construct_type": "star",
                        "construct_dim": 0,
                        "sub_mat_type": "seqdense",
                        "dense_inverse": True,
                        "save_operators": True,
                        "precompute_element_tensors": True,
                    },
                    "sub_ksp_type": "preonly",
                    "sub_pc_type": "lu",
                },
            },
            "mg_coarse": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled": {"ksp_type": "preonly", "pc_type": "lu"},
            },
        }

For more information about patch relaxation, see `Using patch relaxation for multigrid <https://www.firedrakeproject.org/demos/poisson_mg_patches.py.html>`_. The initial solution is shown below.

.. figure:: solution_l1.png
   :align: center
   :alt: Initial Solution from multigrid with initial mesh.


Adaptive Mesh Refinement
-------------------------
In this section we will discuss how to adaptively refine select elements and add the newly refined mesh into the *AdaptiveMeshHierarchy*.
For this problem, we will be using the Babuška-Rheinbolt a-posteriori estimate for an element:

.. math::
   \eta_K^2 = h_K^2 \int_K | f + \nabla^2 u_h |^2 \mathrm{d}x + \frac{h_K}{2} \int_{\partial K \setminus \partial \Omega} \llbracket \nabla u_h \cdot n \rrbracket^2 \mathrm{d}s,

where :math:`K` is the element, :math:`h_K` is the diameter of the element, :math:`n` is the normal, and :math:`\llbracket \cdot \rrbracket` is the jump operator. The a-posteriori estimator is computed using the solution at the current level :math:`h`. Integrating over the domain and using the fact that the components of the estimator are piecewise constant on each cell, we can transform the above estimator into the variational problem 

.. math::
   \int_\Omega \eta_K^2 w \mathrm{d}x = \int_\Omega \sum_K h_K^2 \int_K (f + \text{div} (\text{grad} u_h) )^2 \mathrm{d}x w \mathrm{d}x + \int_\Omega \sum_K \frac{h_K}{2} \int_{\partial K \setminus \partial \Omega} \llbracket \nabla u_h \cdot n \rrbracket^2 \mathrm{d}s w \mathrm{d}x

Our approach will be to compute the estimator over all elements and selectively choose to refine only those that contribute most to the error. To compute the error estimator, we use the function below to solve the variational formulation of the error estimator. Since our estimator is a constant per element, we use a DG0 function space.  ::

   def estimate_error(mesh, uh):
       W = FunctionSpace(mesh, "DG", 0)
       eta_sq = Function(W)
       w = TestFunction(W)
       f = Constant(1)
       h = CellDiameter(mesh)  # symbols for mesh quantities
       n = FacetNormal(mesh)
       v = CellVolume(mesh)
   
       G = (  # compute cellwise error estimator
             inner(eta_sq / v, w)*dx
           - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
           - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS
           - inner(h('-')/2 * jump(grad(uh), n)**2, w('-')) * dS
           )
   
       sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
       solve(G == 0, eta_sq, solver_parameters=sp)
       eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2
   
       with eta.dat.vec_ro as eta_:  # compute estimate for error in energy norm
           error_est = sqrt(eta_.dot(eta_))
       return (eta, error_est)

The next step is to choose which elements to refine. For this we Dörfler marking [Dörfler1996]:  

.. math::
   \eta_K \geq \theta \text{max}_L \eta_L

The logic is to select an element :math:`K` to refine if the estimator is greater than some factor :math:`\theta` of the maximum error estimate of the mesh, where :math:`\theta` ranges from 0 to 1. In our code we choose :math:`theta=0.5`.
With these helper functions complete, we can solve the system iteratively. In the max_iterations is the number of total levels we want to perform multigrid on. We will solve for 15 levels. At every level :math:`l`, we first compute the solution using multigrid with patch relaxation up till level :math:`l`. We then use the current approximation of the solution to estimate the error across the mesh. Finally, we refine the mesh and repeat. ::

   max_iterations = 15
   error_estimators = []
   dofs = []
   for i in range(max_iterations):
       print(f"level {i}")
   
       uh = solve_poisson(mesh, patch_relax)
       VTKFile(f"output/poisson_l/{max_iterations}/adaptive_loop_{i}.pvd").write(uh)
   
       (eta, error_est) = estimate_error(mesh, uh)
       VTKFile(f"output/poisson_l/{max_iterations}/eta_{i}.pvd").write(eta)
   
       print(f"  ||u - u_h|| <= C x {error_est}")
       error_estimators.append(error_est)
       dofs.append(uh.function_space().dim())
   
       if i != max_iterations - 1:
           amh.adapt(eta, theta)

To perform Dörfler marking, refine the current mesh, and add the mesh to the *AdaptiveMeshHierarchy*, we us the ``amh.adapt(eta, theta)`` method. In this method the input is the recently computed error estimator :math:`eta` and the Dörfler marking parameter :math:`theta`. The method always performs this on the current fine mesh in the hierarchy. There is another method for adding a mesh to the hierarchy: ``amh.add_mesh(mesh)``. In this method, refinement on the mesh is performed externally by some custom procedure and the resulting mesh directly gets added to the hierarchy.
The meshes now refine according to the error estimator. The error estimators at levels 3,5, and 15 are shown below. Zooming into the vertex of the L at level 15 shows the error indicator remains strongest there. Further refinements will focus on that area.

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


The convergence follows the expected optimal behavior:

.. figure:: adaptive_convergence_9.png
   :align: center
   :alt: Convergence of the error estimator.

References
-------------------------
[Dörfler1996] W. Dörfler. A convergent adaptive algorithm for poisson’s equation. SIAM Journal on Numerical Analysis, 33(3):1106–1124, 1996.

