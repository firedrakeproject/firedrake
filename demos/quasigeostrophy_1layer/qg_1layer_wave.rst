Quasi-Geostrophic Model in Firedrake
====================================

The Quasi-Geostrophic (QG) model is very important in geophysical fluid
dynamics as it describes some aspects of large-scale flows in the oceans
and atmosphere very well. The interested reader can find derivations in
Pedlosky (1987) and Vallis (2006).

In these notes we present the nonlinear equations for the one-layer QG
model with a free-surface. Then, the weak form will be derived as is
needed for the firedrake library.

Governing Equations
-------------------

The Quasi-Geostrophic (QG) model is very similar to the 2D vorticity
equation. Since the leading order geostrophic velocity is incompressible
in the horizontal, the governing equations can be written as

.. math::

   \begin{aligned}
   \partial_t q + \vec \nabla \cdot \left( \vec u q \right)  + \beta v &= 0, \\
   \vec u & = \vec\nabla^\perp \psi, \\
   \nabla^2 \psi - \frac{1}{L_d^2} \psi &= q. \end{aligned}

where the :math:`\psi` and :math:`q` are the streamfunction and
Potential Vorticity (PV). The Laplacian is 2D since we are only in the
horizontal plane and we defined

.. math:: \vec\nabla^\perp =  \hat e_z \times \vec\nabla.

The first equation above states that the PV is conserved following the
flow. The second equation forces the leading order velocity to be
geostrophic and the third equation is the definition for the QG PV for
this barotropic model. To solve this using Finite Elements it is
necessary to establish the weak form of the model, which is done in the
next subsection.

Weak Form
---------

Evolving the nonlinear equations consists of two steps. First, the
elliptic problem must be solved to compute the streamfunction given the
PV. Second, the PV equation must be integrated forward in time. This is
done using a strong stability preserving Runge Kutta 3 (SSPRK3) method.

Elliptic Equation
~~~~~~~~~~~~~~~~~

First, we focus on the elliptic inversion in the case of a flat bottom.
If we compute the inner product of the equation with the test function
:math:`\phi` we obtain,

.. math::

   \begin{aligned}
   \langle \nabla^2 \psi, \phi \rangle - \frac{1}{L_d^2} \langle \psi, \phi \rangle  &= \langle q, \phi \rangle, \\
   \langle \nabla \psi, \nabla \phi \rangle +  \frac{1}{L_d^2} \langle \psi, \phi \rangle &= -\langle q, \phi \rangle,\end{aligned}

where in the second equation we used the divergence theorem and the
homogeneous Dirichlet boundary conditions on the test function.

Evolution Equation
~~~~~~~~~~~~~~~~~~

The SSPRK3 method used as explained in Gottleib (2005) can be written as

.. math::

   \begin{aligned}
   q^{(1)} &= q^n - \Delta t \left[ \vec \nabla \cdot \left( \vec u^n q^n \right) +  \beta v^n \right] , \\
   q^{(2)} &= \frac34 q^n + \frac14 \left[ q^{(1)} - \Delta t  \vec \nabla \cdot \left( \vec u^{(1)} q^{(1)} \right) 
   - \Delta t \beta v^{(1)}\right], \\
   q^{n+1} &= \frac13 q^n + \frac23 \left[ q^{(2)} - \Delta t \vec \nabla \cdot \left( \vec u^{(2)} q^{(2)} \right) - \Delta t \beta v^{(1)} \right].\end{aligned}

To get the weak form we need to introduce a test function, :math:`p`,
and take the inner product of the first equation with :math:`p`.

.. math::

   \begin{aligned}
   \langle q^{(1)}, p \rangle &= \langle q^n, p \rangle  - \Delta t \langle \vec \nabla \cdot \left( \vec u^n q^n \right), p \rangle 
   - \Delta t \langle \beta  v, q \rangle, \\
   \langle q^{(1)}, p \rangle - \Delta t \langle \vec u^n q^n, \vec\nabla p \rangle  +  \Delta t \langle \beta  v, q \rangle
   &= \langle q^n, p \rangle  - \Delta t \langle \vec u^n q^n, p \rangle_{bdry}\end{aligned}

The first and second terms on the left hand side are referred to as
:math:`a_{mass}` and :math:`a_{int}` in the code. The first term on the
right-hand side is referred to as :math:`a_{mass}` in the code. The
second term on the right-hand side is the extra term due to the DG
framework, which does not exist in the CG version of the problem and it
is referred to as :math:`a_{flux}`. This above problem must be solved
for :math:`q^{(1)}` and then :math:`q^{(2)}` and then these are used to
compute the numerical approximation to the PV at the new time
:math:`q^{n+1)}`.

.. code-block:: python

  from firedrake import *

  Lx   = 2.*pi                                     # Zonal length
  Ly   = 2.*pi                                     # Meridonal length
  n0   = 50                                        # Spatial resolution
  mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="x", quadrilateral=True, reorder=None)

  Vdg = FunctionSpace(mesh,"DG",1)               # DG elements for Potential Vorticity (PV)
  Vcg = FunctionSpace(mesh,"CG",1)               # CG elements for Streamfunction
  Vu  = VectorFunctionSpace(mesh,"DG",1)          # DG elements for velocity

  # Intial Conditions for PV
  q0 = Function(Vdg).interpolate(Expression("0.1*sin(x[0])*sin(x[1])"))

  dq1 = Function(Vdg)       # PV fields for different time steps
  qh  = Function(Vdg)
  q1  = Function(Vdg)

  psi0 = Function(Vcg)      # Streamfunctions for different time steps
  psi1 = Function(Vcg)

  # Physical parameters
  F    = Constant(1.0)         # Rotational Froude number
  beta = Constant(0.1)      # beta plane coefficient  
  Dt   = 0.1                  # Time step
  dt   = Constant(Dt)

  # Set up PV inversion
  psi = TrialFunction(Vcg)  # Test function
  phi = TestFunction(Vcg)   # Trial function

  # Build the weak form for the inversion
  Apsi = (inner(grad(psi),grad(phi)) + F*psi*phi)*dx
  Lpsi = -q1*phi*dx

  # Impose Dirichlet Boundary Conditions on the streamfunction
  bc1 = [DirichletBC(Vcg, 0., 1),
         DirichletBC(Vcg, 0., 2)]

  # Set up Elliptic inverter
  psi_problem = LinearVariationalProblem(Apsi,Lpsi,psi0,bcs=bc1)
  psi_solver = LinearVariationalSolver(psi_problem,
                                       solver_parameters={
          'ksp_type':'cg',
          'pc_type':'sor'
          })

  # Make a gradperp operator
  gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

  # Set up Strong Stability Preserving Runge Kutta 3 (SSPRK3) method

  # Mesh-related functions
  n = FacetNormal(mesh)

  # Set up upwinding type method: ( dot(v, n) + |dot(v, n)| )/2.0
  un = 0.5*(dot(gradperp(psi0), n) + abs(dot(gradperp(psi0), n)))

  # advection equation
  q = TrialFunction(Vdg)
  p = TestFunction(Vdg)
  a_mass = p*q*dx
  a_int  = (dot(grad(p), -gradperp(psi0)*q) + beta*p*psi0.dx(0))*dx
  a_flux = (dot(jump(p), un('+')*q('+') - un('-')*q('-')) )*dS
  arhs   = a_mass - dt*(a_int + a_flux)

  q_problem = LinearVariationalProblem(a_mass, action(arhs,q1), dq1)
  q_solver  = LinearVariationalSolver(q_problem, 
                                      solver_parameters={
          'ksp_type':'cg',
          'pc_type':'sor'
          })


  qfile = File("q.pvd")
  qfile << q0
  psifile = File("psi.pvd")
  psifile << psi0
  vfile = File("v.pvd")
  v = Function(Vu).project(gradperp(psi0))
  vfile << v

  t = 0.
  T = 500.
  dumpfreq = 10
  tdump = 0

  v0 = Function(Vu)

  while(t < (T-Dt/2)):

    # Compute the streamfunction for the known value of q0
    q1.assign(q0)
    psi_solver.solve()
    q_solver.solve()

    # Find intermediate solution q^(1)
    q1.assign(dq1)    
    psi_solver.solve()
    q_solver.solve()

    # Find intermediate solution q^(2)
    q1.assign(0.75*q0 + 0.25*dq1)
    psi_solver.solve()
    q_solver.solve()

    # Find new solution q^(n+1)
    q0.assign(q0/3 + 2*dq1/3)
    
    # Store solutions to xml and pvd
    t +=Dt
    print t

    tdump += 1
    if(tdump==dumpfreq):
        tdump -= dumpfreq
        qfile.write(q0)
        psifile.write(psi0)
        v.project(gradperp(psi0))
        vfile.write(v)
