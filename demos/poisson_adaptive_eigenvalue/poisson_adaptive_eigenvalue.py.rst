Adaptive eigenvalue problem on an L-shaped domain
=================================================

We consider the Dirichlet eigenvalue problem for the Poisson equation on an L-shaped domain :math:`\Omega`:

.. math::
   -\Delta u &= \lambda u \quad \text{in } \Omega, \\
           u &= 0         \quad \text{on } \partial \Omega.

Since the domain has a re-entrant corner, the eigenfunctions exhibit a singularity. Thus, uniform mesh refinement leads to suboptimal convergence. We demonstrate an adaptive strategy driven by a residual-based a posteriori error estimator. To establish rigorous bounds on the true eigenvalue :math:`\lambda`, we use conforming elements (CG) for an upper bound, and nonconforming elements (CR) with Carstensen-Gedicke postprocessing :cite:`CarstensenGedicke:2014` for a guaranteed lower bound.

We start by importing the necessary libraries: ::

  from firedrake import *
  from netgen.occ import *

We then define the L-shaped domain using Netgen's Open CASCADE technology (OCC) interface, and generate an initial mesh: ::

  rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
  rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
  L = rect1 + rect2

  geo = OCCGeometry(L, dim=2)
  ngmesh = geo.GenerateMesh(maxh=0.5)
  mesh = Mesh(ngmesh)

We create a function to solve the eigenvalue problem for both continuous (CG) and nonconforming Crouzeix-Raviart (CR) :cite:`Crouzeix:1973` elements. By the Rayleigh-Ritz principle :cite:`Boffi:2010,Gander:2012`, the CG solution provides an upper bound :math:`\lambda_{\text{ub}}`. The CR solution gives a discrete eigenvalue :math:`\lambda_{\text{CR}}`, which can be postprocessed to yield a guaranteed lower bound:

.. math::
    \lambda_{\text{lb}} = \frac{\lambda_{\text{CR}}}{1 + \kappa_{\text{CR}}^2 \lambda_{\text{CR}} \int_\Omega h^2 |u_{\text{CR}}|^2 \, \text{d}x}

where :math:`\kappa_{\text{CR}} \approx 0.1893` is the constant established by Carstensen and Gedicke :cite:`CarstensenGedicke:2014`. To ensure the lower bound converges optimally on adaptively refined meshes (where the global maximum mesh size does not vanish), we employ a localized version of the consistency bound (as developed in the frameworks of :cite:`CarstensenGedicke:2014,Hu:2014`), which replaces the global mesh size with the element-wise local mesh size :math:`h`. We will use the difference between these guaranteed upper and lower bounds to terminate the adaptive iteration. ::

  def solve_poisson(mesh):
      h_max = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellDiameter(mesh)).dat.data_ro.max()
      bounds = {}
      eigenfunction = None

      for space in ["CG", "CR"]:
          V = FunctionSpace(mesh, space, 1)

          u = TrialFunction(V)
          v = TestFunction(V)

          a = inner(grad(u), grad(v))*dx
          b = inner(u, v)*dx
          bc = DirichletBC(V, 0, "on_boundary")
          eigenproblem = LinearEigenproblem(a, b, bc)

          sp = {
                "eps_gen_hermitian": None,
                "eps_smallest_real": None,
                "eps_monitor_cancel": None,
                "eps_type": "krylovschur",
                "eps_target": 0,
                "st_type": "sinvert",
                }

          eigensolver = LinearEigensolver(eigenproblem, 1, solver_parameters=sp)
          eigensolver.solve()

          if space == "CR":
              lambda_CR = eigensolver.eigenvalue(0).real
              uh_CR = eigensolver.eigenfunction(0)[0]
              h = CellDiameter(mesh)
              # Localized consistency bound to prevent stall on adaptive meshes
              h_norm_sq = assemble(h**2 * uh_CR**2 * dx)
              kappa_CR = 0.1893
              bounds["lb"] = lambda_CR / (1 + kappa_CR**2 * lambda_CR * h_norm_sq)
          if space == "CG":
              bounds["ub"] = eigensolver.eigenvalue(0).real
              eigenfunction = eigensolver.eigenfunction(0)[0]

      eigenfunction.rename("Eigenfunction")
      return (bounds["lb"], bounds["ub"], eigenfunction)

These bounds do not describe where the mesh should be refined so as to reduce the error. For this purpose we employ a standard residual-based a posteriori error estimator :cite:`Duran:2003,Larson:2000`. ::

  def estimate_error(mesh, uh, lam):
      W = FunctionSpace(mesh, "DG", 0)
      eta_sq = Function(W)
      w = TestFunction(W)
      h = CellDiameter(mesh)
      n = FacetNormal(mesh)
      v = CellVolume(mesh)

      G = (
            inner(eta_sq / v, w)*dx
          - inner(h**2 * (lam*uh + div(grad(uh)))**2, w) * dx
          - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS
          - inner(h('-')/2 * jump(grad(uh), n)**2, w('-')) * dS
          )

      sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
      solve(G == 0, eta_sq, solver_parameters=sp)
      eta = Function(W).interpolate(sqrt(eta_sq))

      with eta.dat.vec_ro as eta_:
          error_est = sqrt(eta_.dot(eta_))
      return (eta, error_est)

We define a function to adapt the mesh by refining elements with large error indicators, using the maximum marking strategy with :math:`\theta = 0.5`: ::

  def adapt(mesh, eta):
      W = FunctionSpace(mesh, "DG", 0)
      markers = Function(W)

      with eta.dat.vec_ro as eta_:
          eta_max = eta_.max()[1]

      theta = 0.5
      should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
      markers.interpolate(should_refine)

      return mesh.refine_marked_elements(markers)

Finally, we run the adaptive loop until the upper and lower bounds agree to within a tolerance. ::

  max_iterations = 100
  error_estimators = []
  dofs = []
  err = 1

  for i in range(max_iterations):
      lam_lb, lam_ub, uh = solve_poisson(mesh)
      err = lam_ub - lam_lb
      error_estimators.append(err)
      dofs.append(uh.function_space().dim())
      print(f"Level {i}: Upper bound {lam_ub:.5f}, Lower bound {lam_lb:.5f}, Gap {err:.5e}")

      VTKFile(f"l_eigenfunction_{i}.pvd").write(uh)

      if err < 1e-2:
          break

      eta, _ = estimate_error(mesh, uh, lam_ub)
      mesh = adapt(mesh, eta)

To demonstrate that adaptivity is necessary to achieve the optimal convergence rate, we can run the same script with :math:`\theta = 0`, which forces uniform refinement (all cells are marked for refinement at every step). ::

  mesh_uniform = Mesh(ngmesh)
  uniform_error_estimators = []
  uniform_dofs = []

  def adapt_uniform(mesh, eta):
      markers = Function(FunctionSpace(mesh, "DG", 0)).assign(1.0)
      return mesh.refine_marked_elements(markers)

  for i in range(max_iterations):
      lam_lb, lam_ub, uh = solve_poisson(mesh_uniform)
      err = lam_ub - lam_lb
      uniform_error_estimators.append(err)
      uniform_dofs.append(uh.function_space().dim())
      if err < 1e-2:
          break
      eta, _ = estimate_error(mesh_uniform, uh, lam_ub)
      mesh_uniform = adapt_uniform(mesh_uniform, eta)

We can plot the convergence of the error bound :math:`\lambda_{\text{ub}} - \lambda_{\text{lb}}` against the number of degrees of freedom. With adaptivity, we achieve the optimal :math:`O(N^{-1})` convergence rate, whereas uniform refinement is suboptimal due to the singularity. ::

  try:
      import matplotlib.pyplot as plt
      import numpy as np

      plt.grid()
      plt.loglog(dofs, error_estimators, '-ok', label=r"Adaptive refinement ($\theta = 0.5$)")
      plt.loglog(uniform_dofs, uniform_error_estimators, '-or', label=r"Uniform refinement ($\theta = 0$)")
      scaling = 1.5 * error_estimators[0] / dofs[0]**-1
      plt.loglog(dofs, np.array(dofs)**(-1.0) * scaling, '--', label="Optimal convergence $N^{-1}$")
      plt.xlabel("Number of degrees of freedom $N$")
      plt.ylabel(r"Error bound $\lambda_{\text{ub}} - \lambda_{\text{lb}}$")
      plt.legend()
      plt.savefig("adaptive_eigenvalue_convergence.png")
  except ImportError:
      warning("Matplotlib not imported")

.. figure:: adaptive_eigenvalue_convergence.png
   :align: center
   :figwidth: 80%

   Convergence of the guaranteed error bound. Note that the adaptive scheme achieves the optimal convergence rate of :math:`O(N^{-1})`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames

