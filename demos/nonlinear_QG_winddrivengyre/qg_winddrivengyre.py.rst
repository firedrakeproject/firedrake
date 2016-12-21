Wind-Driven Gyres: Quasi-Geostrophic Limit
==========================================

Building on the previous two demos that used the Quasi-Geostrophic
(QG) model (insert two links), we now consider how to determine a
wind-driven gyre solution that includes bottom bottom drag and
nonlinear advection. This is referred to as the Nonlinear Stommel
Problem.

This is a classical problem going back to Stommel (1949) (link). Even
though it is far too simple to describe the dynamics of the real
oceans quantitatively, it did explain qualitatively why we have
western intensification in the world’s gyres. The curl of the wind
stress adds vorticity into the gyres and the latitudinal variation in
the Coriolis parameter causes a weak equatorward flow away from the
boundaries (Sverdrup flow). It is because of the dissipation that
arises near the boundaries that we must have western intensification.
This was first shown by Stommel (1949) using simple bottom drag but it
was only years later after Munk (1953) did a similar calculation using
lateral viscosity that people took the idea seriously.

After three quarters of a century we are still unable to parametrize
the dissipative effects of the small scales so it is very difficult to
get a good quantiative predictions as to the mean structure of the
gyre that is generated. However, this demo aims to compute the
structure of the oceanic gyre given particular parameters. The
interested reader can read more about this in (Vallis chapter 14). In
this tutorial we will consider the nonlinear Stommel problem.

Governing PDE:Stommel Problem
=============================

The nonlinear, one-layer, QG model equation that is driven by the winds
above (say :math:`Q_{winds}`, which is the vorticity of the winds that
drive the ocean from above) is,

.. math:: \partial_{t}q + \vec{u} \cdot \vec{\nabla} q + \beta v = -rq + Q_{winds}

with the Potential Voritcity and geostrophic velocities defined as

.. math::

   \begin{aligned}
   q = \nabla^2 \psi - F \psi,
   \quad \mbox{ and } \quad
   \vec u = \hat z \times \vec\nabla \psi\end{aligned}

where :math:`\psi` is the stream-function, :math:`\vec{u}=(u, v)` is the
velocity field, :math:`q` is the PV, :math:`\beta`
is the latitudinal gradient of Coriolis parameter, and :math:`F` is the
rotational Froude number.

The non-conservative aspects of this model
occur because of :math:`r`, the strength of the bottom drag, and
:math:`Q_{winds}`, the vorticity of the winds. We pick the wind forcing
as to generate a single gyre,

.. math:: Q_{winds} = \tau \cos\left( \pi \left[\frac{y}{L_y} - \frac{1}{2} \right] \right)

where :math:`L_y` is the length of our domain and :math:`\tau` is the strength of our wind forcing. By putting a :math:`2` in front of the :math:`\pi` we
get a double gyre (Vallis, 2005).

If we only look for steady solutions
in time, we can ignore the time derivative term, and we get

.. math::

   \begin{gathered}
   (\vec{u} \cdot \vec\nabla)\left( \nabla^2 \psi - F \psi\right)
   + \beta \frac{\partial \psi}{\partial x} = - rq + Q_{winds} 
   \end{gathered}

We can write this out in one equation and is the nonlinear Stommel
problem.

.. math::

   \begin{gathered}
   \vec u \cdot \vec\nabla \left( \nabla^2 \psi \right) + r(\nabla^{2} \psi - F\psi) + \beta \frac{\partial \psi}{\partial x} =  Q_{winds} 
   \end{gathered}

Note that we dropped the :math:`-F \psi` term in the nonlinear advection
because the streamfunction does not change following the flow, and
therefore, we can neglect that term entirely.

Weak Formulation
================

To build the weak form of the problem in Firedrake we must find the weak
form of this equation. We begin by multiplying this equation by a Test
Function, :math:`\phi`, which is in the same space as the
streamfunction, and then integrate over the domain :math:`A`,

.. math:: \iint_{A} \phi (\vec u \cdot \vec\nabla) \nabla^2 \psi \,dA  +  r\phi (\nabla^{2} \psi - F\psi)\,dA + \beta\phi\frac{\partial \psi}{\partial x} \,dA =  \iint_{A} \phi \cdot F_{winds} \,dA

The nonlinear term can be rewritten using the fact that the velocity is
divergent free and then integrating by parts,

.. math::

   \begin{aligned}
   \iint_{A} \phi (\vec u \cdot \vec\nabla) \nabla^2 \psi
   =  \int_{A} \phi \vec\nabla \cdot \left(\vec u (\nabla^2 \psi)\right) 
   = - \iint_{A}( \vec\nabla \phi \cdot \vec u){\nabla}^{2}\psi \, dA.\end{aligned}

Note that because we have no normal flow boundary conditions the
boundary contribution is zero. For the term with bottom drag we
integrate by parts and use the fact that the streamfunction is zero on
the walls

.. math::

   \begin{aligned}
   \iint_{A} r \phi \left( \vec{\nabla}^2 \psi - F \psi \right) \, dA & 
   = -r \iint_{A}  \Big(\vec{\nabla}\phi \cdot \vec{\nabla}\psi
   + F \phi \psi \Big)\, dA
   + r \oint_{\partial A} \phi \cdot \frac{\partial \psi}{\partial n} \,dS
  \end{aligned}

The boundary integral above banishes because are are setting the streamfunction to be zero on the boundary.

Finally we can put the equation back together again to produce the weak form of our problem.

.. math:: \iint_{A} \Bigg( - (\vec\nabla \phi \cdot \vec u) \vec{\nabla}^{2}\psi  -r \Big(\vec{\nabla}\phi \cdot \vec{\nabla}\psi + F \phi \psi \Big) + \beta\phi\frac{\partial \psi}{\partial x} \Bigg) \,dA =  \iint_{A} \phi \cdot F_{winds} \,dA

The above problem is the weak form of the nonlinear Stommel problem.  The linear term arises from negelcting the nonlinear advection, and can easily be obtained by neglecting the first term on the left hand side.
	  
Defining the Problem
====================

Now that we know the weak form we are now ready to solve this using Firedrake!

First, we import the Firedrake, PETSc, NumPy and UFL libraries, ::

  from firedrake import *
  from firedrake.petsc import PETSc
  import numpy as np
  import ufl

Next, we can define the geometry of our domain. In this example, we
will be using a square of length one with 50 cells. ::
  
  n0 = 50           #Spatial resolution
  Ly = 1.0          #Meridonal length
  Lx = 1.0          #Zonal length
  mesh = RectangleMesh(n0, n0, Lx, Ly, reorder = None)

We can then define the Function Space within which the
solution of the streamfunction will reside. ::

  Vcg = FunctionSpace(mesh, “CG”, 3) # CG elements for Streamfunction

We will also impose no-normal flow strongly to ensure that the
boundary condition :math:`\psi = 0` will be met, ::
  
  bc = DirichletBC(Vcg, 0.0, “on_boundary”)

Now we will define all the parameters we are using in this tutorial. ::

  beta = Constant(“1.0”)      # Beta parameter
  F = Constant(“1.0”)         # Burger number
  r = Constant(“0.2”)         # Bottom drag
  tau = Constant(“0.001”)     # Wind Forcing
  Qwinds = Function(Vcg).interpolate(Expression(“-tau*cos(pi*( (x[1]/Ly)-0.5))”, tau=tau, Ly=Ly))

We can now define the Test Function and the Trial Function of this problem, both must be in the same function space::

  phi, psi = TestFunction(Vcg), TrialFunction(Vcg)

We must define functions that will store our linear and nonlinear solutions.
In order to solve the nonlinear problem, we use the linear
solution as a guess for the nonlinear problem. ::

  psi_lin = Function(Vcg, name=“Linear Streamfunction”)
  psi_non = Function(Vcg, name=“Nonlinear Streamfunction”)

We can also define an operator for our formulation.
The gradperp() operator is defined as a lambda function which enables us
to compute the gradient cross product in
the z-direction (:math:`\hat{z}\times\nabla`), necessary to find 
the geostrophic velocities. ::

  gradperp = lambda i: as_vector((-i.dx(1),i.dx(0)))

We can finally write down the linear Stommel equation in it’s weak
form. We will use the solution to this as the input for the nonlinear
Stommel equation. ::

  a = - r*inner(grad(psi), grad(phi))*dx - F*psi*phi*dx + beta*psi.dx(0)*phi*dx
  L = Qwinds*phi*dx

We set-up an elliptic inverter for this problem, and solve for the
linear streamfunction, ::

  linear_problem = LinearVariationalProblem(a, L, psi_lin, bcs=bc)
  linear_solver = LinearVariationalSolver(linear_problem,
  solver_parameters= ’ksp_type’:’preonly’, ’pc_type’:’lu’)
  linear_solver.solve()

We will redefine the nonlinear stream-function as it’s guess, the
linear stream-function ::

  psi_non.assign(psi_lin)

And now we can define the weak form of the nonlinear problem. Note
that the input is *not* a TrialFunction. ::

  G = - inner(grad(phi),gradperp(psi_non))*div(grad(psi_non))*dx
  -r*inner(grad(psi_non), grad(phi))*dx - F*psi_non*phi*dx
  + beta*psi_non.dx(0)*phi*dx
  - Fwinds*phi*dx

We solve for the nonlinear streamfunction now by setting up another
elliptic inverter, ::

  nonlinear_problem = NonlinearVariationalProblem(G, psi_non, bcs=bc)
  nonlinear_solver = NonlinearVariationalSolver(nonlinear_problem,
  solver_parameters= ’snes_type’: ’newtonls’, ’ksp_type’:’preonly’,
  ’pc_type’:’lu’) nonlinear_solver.solve()

Now that we have the full solution to the nonlinear Stommel problem,
we can plot it, ::

  p = plot(psi_non)
  p.show()

  file = File(’Nonlinear Streamfunction.pvd’)
  file.write(psi_non)

We can also see the difference between the linear solution and the
nonlinear solution. We do this by defining a weak form.  (Note: this can probably be done differently but it does work.) :: 

  tf, difference = TestFunction(Vcg), TrialFunction(Vcg)

  a = difference*tf*dx L = (psi_lin - psi_non)*tf*dx
  difference = Function(Vcg, name=“Difference”)
  solve(a==L, difference, None)

  p = plot(difference)
  p.show()

  file = File(“Difference between Linear and Nonlinear Streamfunction.pvd”)
  file.write(difference) 
