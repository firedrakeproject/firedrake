Coupled equations
=================

This tutorial provides a guide to solving equations coupled on meshes represented by different domain spaces in Firedrake. As an example, we couple a Poisson and Helmholtz equation on meshes connected along one edge with Dirichlet and Neumann boundary conditions.

Coupled Poisson and Helmholtz equations
---------------------------------------

Consider unit squares :math:`\Omega_1 = [0,1] \times [0,1]` and :math:`\Omega_2 = [1,2] \times [0,1]` and let the boundary :math:`\Gamma = {(1, y) : y \in [0,1]}` be the shared edge between each unit square mesh. The Poisson equation is defined on :math:`\Omega_1` as

.. math::

  -\nabla^2 u_1 &= f

  u_1 &= 0 \ \textrm{on}\ \partial \Omega_1 \setminus \Gamma.

The Helmholtz equation is defined on :math:`\Omega_2` as

.. math::
  -\nabla^2 u_2 + u_2 &= g

  \nabla u_2 \cdot n &= 0 \ \textrm{on}\ \partial \Omega_2 \setminus \Gamma,

where :math:`f \ \textrm{and}\ g` are known functions and :math:`u_1, u_2 \in V^1, V^2` are the solutions to these equations in some function spaces :math:`V^1 \ \textrm{and}\ V^2`. These solutions are the trial functions.

The weak forms for the Poisson and Helmholtz equations defined above are derived from multiplying each equation by an arbitrary test function :math:`v \in V` and integrating by parts. Further details on this process can be found in `Mixed formulation for the Poisson equation`_ and `Simple Helmholtz equation`_. From the weak forms, variational problems can be defined. 

For the Poisson equation, the variational problem involves finding :math:`u_1 \in V^1` such that :math:`a_{11}(u_1, v_1) = L_1(v_1) \ \textrm{for all}\ v_1 \in V^1` where

.. math::

  a_{11} (u_1, v_1) &= \int_{\Omega_1}\nabla u_1 \cdot \nabla v_1  \ {\rm d} x - \int_{\Gamma} v_1 \nabla u_1 \cdot n \ {\rm d} s,

  L_1 (v_1) &= \int_{\Omega_1}f v_1  \ {\rm d} x.

Similarly, the variational problem for the Helmholtz equation involves finding :math:`u_2 \in V^2` such that :math:`a_{22} (u_2, v_2) = L_2 (v_2) \ \textrm{for all}\ v_2 \in V^2` where 

.. math::

  a_{22}(u_2, v_2) &= \int_{\Omega_2}\nabla u_2 \cdot \nabla u_2 + u_2 v_2  \ {\rm d} x - \int_{\Gamma}v_2 \nabla u_2 \cdot n  \ {\rm d} s,

  L_2 (v_2) &= \int_{\Omega_2}g v_2  \ {\rm d} x.

Along the shared interface :math:`\Gamma`, we enforce a Neumann boundary condition :math:`\frac{\partial u_1}{\partial n} = \frac{\partial u_2}{\partial n}` on the Poisson equation and a Dirichlet boundary condition :math:`u_1 = u_2` on the Helmholtz equation. This is primarily accomplished through the coupling terms :math:`a_{12}` and :math:`a_{21}`. 

The Dirichlet boundary condition is weakly defined using Nitsche's method, allowing for more accurate approximations of the solution. Thus, a penalty term is also added to :math:`a_{22}`. 

.. math::

  a_{11} (u_1, v_1) &= \int_{\Omega_1}\nabla u_1 \cdot \nabla v_1  \ {\rm d} x - \int_{\Gamma} v_1 \nabla u_1 \cdot n \ {\rm d} s,

  a_{22}(u_2, v_2) &= \int_{\Omega_2}\nabla u_2 \cdot \nabla u_2 + u_2 v_2  \ {\rm d} x - \int_{\Gamma}v_2 \nabla u_2 \cdot n  \ {\rm d} s + w_2 \int_{\Gamma}u_2 v_2 \ {\rm d} s.

Along the shared interface, the two meshes are coupled. These coupling terms are defined as

.. math::

  a_{12}(u_2, v_1) &= - \int_{\Gamma} (\mathcal{I}_{V^1} (\nabla u_2) \cdot n) v_1 \ {\rm d} s,

  a_{21}(u_1, v_2) &= -w_2 \int_{\Gamma}\mathcal{I}_{V^2} (u_1) v_2 \ {\rm d} s,

where :math:`w_2 = \frac{w_0}{h}` is a penalty parameter. :math:`w_0` is a constant and :math:`h` the element spacings for the penalty parameter. The penalty constant :math:`w_0` is typically found through trial and error. Additionally, :math:`\mathcal{I}_{V^2}: V^1 \rightarrow V^2` and :math:`\mathcal{I}_{V^1}: V^2 \rightarrow V^1` are cross-mesh interpolation operators, defining the trial function :math:`u_1` in the domain :math:`V^2` and vice versa. :math:`a_{12}` enforces the Neumann boundary condition whereas :math:`a_{21}` enforces the Dirichlet boundary condition.

Overall, the variational problem for the coupled equations is: find :math:`(u_1, u_2) \in V^1 \times V^2` such that

.. math::

  a_{11}(u_1,v_1) + a_{12}(u_2,v_1) + a_{22}(u_2,v_2) + a_{21}(u_1,v_2) = L_1(v_1) + L_2(v_2) \ \textrm{for all}\ (v_1, v_2) \in V^1 \times V^2.


Method of Manufactured Solutions (MMS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method of manufactured solutions (MMS) is used to verify the accuracy of approximated solutions by explicitly specifying a solution for the problem at hand, ensuring that this solution satisfies all conditions set. The functions :math:`f` and :math:`g` to be passed into the solver are calculated from these exact solutions, resulting in an approximated solution. We can then compare the two solutions, analysing the accuracy of the approximated solution. 

For this demo, we define the exact solutions as

.. math::
  u_1 &= x \sin(\pi y)^2,

  u_2 &= \sin(\pi y)^2 \left(x - \frac{(x - 1)^2}{2} \right),

which satisfies the boundary conditions set on this problem.


Implementation
--------------

We now implement this problem using Firedrake by first initialising the constants and variables required for solving and analysing the problem. ::

  from firedrake import *

  # Constants
  PLOT = False
  VERBOSE = True

  # Variables initialised for convergence analysis
  n1_list = [2,4,8,16,32]
  n2_list = [2,4,8,16,32]
  mesh1_list = []
  mesh2_list = []
  h1_array = []
  h2_array = []
  errors_1 = []
  errors_2 = []

For each index in ``n1_list`` and ``n2_list``, we define two meshes with ``n1 x n1`` and ``n2 x n2`` elements and a shared interface at :math:`x = 1`. ::

  for n1,n2 in zip(n1_list, n2_list):
    mesh1 = UnitSquareMesh(n1, n1, quadrilateral=True)
    mesh2 = UnitSquareMesh(n2, n2, quadrilateral=True)
    mesh2.coordinates.dat.data[:, 0] += 1.0  # Shift to the right by 1

    mesh1_list.append(mesh1)
    mesh2_list.append(mesh2)

The pairs of meshes ``mesh1`` and ``mesh2`` are then passed into ``build_problem()``, which defines the coupled problem onto the meshes. In this function, we first define the exact solutions for this problem in order to calculate the source functions. ::

  def build_problem(mesh1, mesh2):
    p = 3
    p_inner = 2
    w2 = Constant(50.0)/CellDiameter(mesh2)  # Nitsche penalty weight

    x1, y1 = SpatialCoordinate(mesh1)
    x2, y2 = SpatialCoordinate(mesh2)
    u1_exact = x1 * sin(pi * y1) ** 2
    u2_exact = sin(pi * y2) ** 2 * (x2 - (x2 - 1) ** 2 / 2)
    
    # RHS functions
    f1 = -div(grad(u1_exact))
    f2 = -div(grad(u2_exact)) + u2_exact

Measures are then defined where ``n1`` and ``n2`` are the unit normal vectors for each mesh, ``dx`` integrates over the respective meshes and ``ds`` integrates on the edges of the meshes.

.. code-block::
  :dedent: 0

    n1 = FacetNormal(mesh1)
    n2 = FacetNormal(mesh2)
    dx1 = Measure("dx", domain=mesh1)
    dx2 = Measure("dx", domain=mesh2)
    ds1 = Measure("ds", domain=mesh1, subdomain_id=2)
    ds2 = Measure("ds", domain=mesh2, subdomain_id=1)


Function spaces ``V1`` and ``V2`` are combined to create a mixed function space ``W``, with test and trial functions defined on the subspaces of this mixed function space.

.. code-block::
  :dedent: 0
  
    V1 = FunctionSpace(mesh1, "CG", p)
    V2 = FunctionSpace(mesh2, "CG", p)
    W = V1 * V2

    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)


The matrices ``A11`` and ``A22`` are defined directly on the above function spaces. 

.. code-block::
  :dedent: 0

    # Poisson on mesh_1
    A11_form = inner(grad(u1), grad(v1)) * dx1

    # Helmholtz on mesh_2
    A22_form = (inner(grad(u2), grad(v2)) + inner(u2, v2)) * dx2 \
                - inner(dot(grad(u2), n2), v2) * ds2 \
                + w2 * inner(u2, v2) * ds2

Intermediate spaces are used to define the coupling terms in the variational problem. These coupling terms are calculated by finding the product of the cross-mesh interpolation matrices ``B12, B21`` and the mass matrices ``M1, M2``, placing the coupling terms in the dual space of ``W``. Recall that ``A12`` enforces the Neumann boundary condition whilst ``A21`` enforces the Dirichlet boundary condition using Nitsche's method.

.. code-block::
  :dedent: 0

    # Intermediate spaces
    Q1v = VectorFunctionSpace(mesh1, "CG", p_inner)
    Q2 = FunctionSpace(mesh2, "CG", p_inner)
    
    # A12: row v1, column u2
    # W --B12--> Q1v --M1--> W^*
    q1v = TrialFunction(Q1v)
    M1 = -inner(dot(q1v, n1), v1) * ds1  # Q1v -> W^*
    B12 = interpolate(grad(u2), Q1v, allow_missing_dofs=True)  # W -> Q1v
    A12_form = action(M1, B12)

    # A21: row v2, column u1.
    # W --B21--> Q2 --M2--> W^*
    q2 = TrialFunction(Q2)
    M2 = -w2 * inner(q2, v2) * ds2  # Q2 -> W^*
    B21 = interpolate(u1, Q2, allow_missing_dofs=True)  # W -> Q2
    A21_form = action(M2, B21)

    
These definitions are combined to form the overall problem to be solved ``Ax = L``. From this method, we return ``A, L, W`` and the exact solutions which are mapped onto the function space.

.. code-block::
  :dedent: 0

    # RHS
    b1 = inner(f1, v1) * dx1
    b2 = inner(f2, v2) * dx2
    A = A11_form + A12_form + A21_form + A22_form
    L = b1 + b2

    # Exact solutions used for further analysis
    u1_exact_func = Function(V1).interpolate(u1_exact)
    u2_exact_func = Function(V2).interpolate(u2_exact)

    return A, L, W, u1_exact_func, u2_exact_func

The resulting solution can be plotted by calling ``plot()``. Matplotlib is required for plotting with this method and Firedrake's `trisurf`_ is used to produce a three-dimensional surface plot. ::

  import matplotlib.pyplot as plt
  from firedrake.pyplot import trisurf

  def plot(filename, u_1, u_2):
    u1_vals = u_1.dat.data_ro
    u2_vals = u_2.dat.data_ro
    vmin = min(u1_vals.min(), u2_vals.min())
    vmax = max(u1_vals.max(), u2_vals.max())

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")
    trisurf(u_1, axes=ax, vmin=vmin, vmax=vmax, cmap="viridis")
    trisurf(u_2, axes=ax, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.tight_layout()
    plt.savefig(filename)

Utilising both methods mentioned above, the coupled problem can be solved for each specified mesh size as shown below and plotted on the three-dimensional surface plot. We additionally calculate the L2 error norm between the approximated and exact solutions, noting both the error norms and the distance between elements in each mesh ``h`` at each iteration for convergence analysis. ::

  for n1, n2, mesh1, mesh2 in zip(n1_list, n2_list, mesh1_list, mesh2_list):
    A, L, W, u1_exact_func, u2_exact_func = build_problem(mesh1, mesh2)
    u_sol = Function(W)
    
    bc = DirichletBC(W.sub(0), 0, [1, 3, 4]) # Defining a 
    problem = LinearVariationalProblem(A, L, u_sol, bcs=bc)
    params = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    u_1, u_2 = u_sol.subfunctions

    if PLOT:
      plot(f"dirichlet_neumann_example_{n1}_{n2}.png", u_1, u_2)

    e_1 = errornorm(u1_exact_func, u_1, norm_type="L2")
    e_2 = errornorm(u2_exact_func, u_2, norm_type="L2")
    h1 = 1.0/n1
    h2 = 1.0/n2

    h1_array.append(h1)
    h2_array.append(h2)
    errors_1.append(e_1)
    errors_2.append(e_2)


Convergence Analysis
--------------------

Using the L2 error norms calculated above, we can approximate the rate of convergence as

.. math::
  
  q = \frac{\ln \left( \frac{||u_{h_1} - \tilde{u}||_{L^2}}{||u_{h_2} - \tilde{u}||_{L^2}} \right)}{\ln \left( \frac{h_1}{h_2} \right)}
  
where :math:`u_{h_1}` and :math:`u_{h_2}` are approximated solutions on meshes of differing sizes, :math:`\tilde{u}` is the exact solution, :math:`h_1` and :math:`h_2` are the element spacings in each mesh. It is expected for finite element problems to converge to the exact solution at rate :math:`O(h^{p+1})` where :math:`p` is the dimension of the domain. ::

  ratios_1 = []
  ratios_2 = []
  for i in range(len(h1_array) - 1):
    q1_numerator = np.log(errors_1[i]/errors_1[i+1])
    q2_numerator = np.log(errors_2[i]/errors_2[i+1])
    q1_denominator = np.log(h1_array[i]/h1_array[i+1])
    q2_denominator = np.log(h2_array[i]/h2_array[i+1])

    q1 = q1_numerator/q1_denominator
    q2 = q2_numerator/q2_denominator
    ratios_1.append(q1)
    ratios_2.append(q2)

If the ``VERBOSE`` flag is set to True, the following block runs and prints the convergence analysis results, presenting an error graph alongside. :: 

  if VERBOSE:
    print(f"{'h':>10} {'Error 1':>15} {'Rate 1':>10}")
    for i in range(len(errors_1)):
      if i == 0:
        print(f"{h1_array[i]:10.5f} {errors_1[i]:15.6e} {'-':>10}")
      else:
        print(f"{h1_array[i]:10.5f} {errors_1[i]:15.6e} {ratios_1[i-1]:10.4f}")

    print(f"{'h':>10} {'Error 2':>15} {'Rate 2':>10}")
    for i in range(len(errors_2)):
      if i == 0:
        print(f"{h2_array[i]:10.5f} {errors_2[i]:15.6e} {'-':>10}")
      else:
        print(f"{h2_array[i]:10.5f} {errors_2[i]:15.6e} {ratios_2[i-1]:10.4f}")

    plt.figure(figsize=(8,8))
    plt.loglog(h2_array, errors_2, "o-", label="Helmholtz")
    plt.loglog(h1_array, errors_1, "s-", label="Poisson")
    plt.xlabel("h")
    plt.ylabel("L2 error")
    plt.gca().invert_xaxis()
    plt.grid(False)
    plt.legend()
    plt.title("Helmholtz-Poisson Coupling with Dirichlet-Neumann BCs")
    plt.savefig("Logloggraph.png")

Running the above script with ``n1_list = [2,4,8,16,32]``, ``n2_list = [2,4,8,16,32]`` and domain dimensions ``p = 4`` and ``p_inner = 4``, we obtain the following error norm and convergence rate values.

.. list-table::
   :header-rows: 1

   * - h
     - Error 1
     - Rate 1
   * - 0.50000
     - 1.981675e-05
     - -
   * - 0.25000
     - 1.978103e-06
     - 3.3245
   * - 0.12500
     - 3.139620e-08
     - 5.9774
   * - 0.06250
     - 4.924948e-10
     - 5.9943
   * - 0.03125
     - 7.703876e-12
     - 5.9984

.. list-table::
   :header-rows: 1

   * - h
     - Error 2
     - Rate 2
   * - 0.50000
     - 5.559123e-05
     - -
   * - 0.25000
     - 4.598223e-06
     - 3.5957
   * - 0.12500
     - 7.296559e-08
     - 5.9777
   * - 0.06250
     - 1.144489e-09
     - 5.9944
   * - 0.03125
     - 1.791455e-11
     - 5.9974

This corresponds to a problem with conforming coupled meshes, converging with rate :math:`O(h^{p+2})` and small error norm values.

This problem can also be solved on non-conforming meshes. As an example, we run the script with ``n1_list = [8,8,8,8,8]``, ``n2_list = [2,4,8,16,32]`` and the same domain dimensions. We now obtain the following error norm and convergence rate values.

.. list-table::
   :header-rows: 1

   * - h
     - Error 2
     - Rate 2
   * - 0.50000
     - 5.466399e-05
     - -
   * - 0.25000
     - 4.577680e-06
     - 3.5779
   * - 0.12500
     - 7.296559e-08
     - 5.9713
   * - 0.06250
     - 1.571188e-07
     - -1.1066
   * - 0.03125
     - 3.634664e-07
     - -1.2100

Similar results to the conforming mesh case are observed, where the approximated solution decreases in error at a rate of :math:`O(h^{p+2})` until :math:`h = \frac{1}{n_1}` is reached.

A python script version of this demo can be found :demo:`here <coupled_equations.py>`.

.. _DG advection equation with upwinding: https://www.firedrakeproject.org/demos/DG_advection.py.html
.. _Simple Helmholtz equation: https://www.firedrakeproject.org/demos/helmholtz.py.html
.. _Mixed formulation for the Poisson equation: https://www.firedrakeproject.org/demos/poisson_mixed.py.html
.. _Trisurf: https://www.firedrakeproject.org/firedrake.pyplot.html#firedrake.pyplot.trisurf
