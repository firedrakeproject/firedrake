Benney-Luke equations: a reduced water wave model
=================================================


.. rst-class:: emphasis

    This tutorial was contributed by `Anna Kalogirou <mailto:A.Kalogirou@leeds.ac.uk>`__
    and `Onno Bokhove <mailto:O.Bokhove@leeds.ac.uk>`__.

    The work is based on the article "Variational water wave
    modelling: from continuum to experiment" by Onno Bokhove and Anna
    Kalogirou :cite:`2015:lmscup`. The authors gratefully
    acknowledge funding from EPSRC grant no. EP/L025388/1
    with a link to the Dutch Technology Foundation STW for the project
    "FastFEM: behavior of fast ships in waves".

The Benney-Luke-type equations consist of a reduced potential flow water wave model based on the assumptions of small amplitude parameter :math:`\epsilon` and small dispersion parameter :math:`\mu` (defined by the square of the ratio of the typical depth over a horizontal length scale). They describe the deviation from the still water surface, :math:`\eta(x,y,t)`, and the free surface potential, :math:`\phi(x,y,t)`. A modified version of the Benney-Luke equations can be obtained by the variational principle:

.. math::

  0 &= \delta\int_0^T \int_{\Omega} \eta\phi_t - \frac{\mu}{2}\!\eta\Delta\phi_t + \frac{1}{2}\!\eta^2 + \frac{1}{2}\!\left(1+\epsilon\eta\right)\!\left|\nabla\phi\right|^2 + \frac{\mu}{3}\!\left( \Delta\phi \right)^2 \,dx\,dy\,dt \\
    &= \delta\int_0^T \int_{\Omega} \eta\phi_t + \frac{\mu}{2}\nabla\eta\cdot\nabla\phi_t + \frac{1}{2}\!\eta^2 + \frac{1}{2}\!\left(1+\epsilon\eta\right)\!\left|\nabla\phi\right|^2 + \mu\left( \nabla q\cdot\nabla\phi - \frac{3}{4}q^2 \right) \,dx\,dy\,dt \\
    &= \int_0^T \int_{\Omega} \left( \delta\eta\,\phi_t + \frac{\mu}{2}\nabla\delta\eta\cdot\nabla\phi_t + \eta\,\delta\eta + \frac{\epsilon}{2}\delta\eta\left|\nabla\phi\right|^2 \right) \\
    & \qquad \qquad - \left( \delta\phi\,\eta_t + \frac{\mu}{2}\nabla\eta_t\cdot\nabla\delta\phi - \left(1+\epsilon\eta\right)\!\nabla\phi\cdot\nabla\delta\phi - \mu\nabla q\cdot\nabla\delta\phi \right) \\
    & \qquad \qquad + \mu\left( \nabla\delta q \cdot\nabla\phi - \frac{3}{2}q\,\delta q  \right) \,dx\,dy\,dt,

where the spatial domain is assumed to be :math:`\Omega` with natural boundary conditions, namely Neumann conditions on all the boundaries. In addition, suitable end-point conditions at :math:`t=0` and :math:`t=T` are used. Note that the introduction of the auxiliary function :math:`q` is performed in order to lower the highest derivatives. This is advantageous in a :math:`C^0` finite element formulation and motivated the modification of the "standard" Benney-Luke equations. The partial variations in the last line of the variational principle can be integrated by parts in order to get expressions that only depend on :math:`\delta\eta,\,\delta\phi,\,\delta q` and not their derivatives:

.. math::

  0 = \int_0^T \int_{\Omega} &\left( \phi_t - \frac{\mu}{2}\Delta\phi_t + \eta + \frac{\epsilon}{2}\left|\nabla\phi\right|^2 \right)\delta\eta \\
                              - &\left( \eta_t - \frac{\mu}{2}\Delta\eta_t + \nabla\cdot\bigl(\left(1+\epsilon\eta\right)\!\nabla\phi\bigr)+\mu\Delta q \right)\delta\phi \\
                              - &\mu\left( \Delta\phi + \frac{3}{2}q \right)\delta q \,dx\,dy\,dt.

Since the variations :math:`\delta\eta,\,\delta\phi,\,\delta q` are arbitrary, the modified Benney-Luke equations then arise for functions :math:`\eta,\phi,q\in V` from a suitable function space :math:`V` and are given by:

.. math::

  \phi_t - \frac{\mu}{2}\Delta\phi_t + \eta + \frac{\epsilon}{2}\left|\nabla\phi\right|^2 &= 0 \\
  \eta_t - \frac{\mu}{2}\Delta\eta_t + \nabla\cdot\bigl(\left(1+\epsilon\eta\right)\!\nabla\phi\bigr)+\mu\Delta q &= 0 \\
  q &= - \frac{2}{3}\Delta\phi.

We can either directly use the partial variations in the variational principle above (last line) as the fundamental weak formulation (with :math:`\delta\phi,\, \delta\eta,\, \delta q` playing the role of test functions), or multiply the equations by a test function :math:`v\in V` and integrate over the domain in order to obtain a weak formulation in a classic manner

.. math::

  \int_{\Omega} \phi_t\,v + \frac{\mu}{2}\nabla\phi_t\cdot\nabla v + \eta\,v + \frac{\epsilon}{2}\nabla\phi\cdot\nabla\phi\,v \,dx\,dy &= 0 \\
  \int_{\Omega} \eta_t\,v + \frac{\mu}{2}\nabla\eta_t\cdot\nabla v - \left(1+\epsilon\eta\right)\nabla\phi\cdot\nabla v - \mu\nabla q\cdot\nabla v \,dx\,dy &= 0 \\
  \int_{\Omega} q\,v - \frac{2}{3}\nabla\phi\cdot\nabla v \,dx\,dy &= 0.

Note that the Neumann boundary conditions have been used to remove every surface term that resulted from the integration by parts. Moreover, the variational form of the system requires the use of a symplectic integrator for the time-discretisation. Here we choose the 2nd-order Stormer-Verlet scheme :cite:`2006:SV`, which requires two half-steps to update :math:`\phi` in time (one implicit and one explicit in general) and one (implicit) step for :math:`\eta`:

.. math::

  \int_{\Omega} \frac{\phi^{n+1/2}-\phi^n}{\frac{1}{2}\!dt}\,v + \frac{\mu}{2}\nabla\left(\frac{\phi^{n+1/2}-\phi^n}{\frac{1}{2}\!dt}\right)\cdot\nabla v \hspace{8em}\\
  + \eta^n\,v + \frac{\epsilon}{2}\nabla\phi^{n+1/2}\cdot\nabla\phi^{n+1/2}\,v \,dx\,dy &= 0 \\\\
  %
  \int_{\Omega} q^{n+1/2}\,v - \frac{2}{3}\nabla\phi^{n+1/2}\cdot\nabla v \,dx\,dy &= 0 \\\\
  %
  \int_{\Omega} \frac{\eta^{n+1}-\eta^n}{dt}\,v + \frac{\mu}{2}\nabla\left(\frac{\eta^{n+1}-\eta^n}{dt}\right)\cdot\nabla v \hspace{8em}\\
  - \frac{1}{2}\Bigl( \left(1+\epsilon\eta^{n+1}\right) + \left(1+\epsilon\eta^n\right) \Bigr)\nabla\phi^{n+1/2}\cdot\nabla v \hspace{4em}\\
  - \mu\nabla q^{n+1/2}\cdot\nabla v \,dx\,dy &= 0 \\\\
  %
  \int_{\Omega} \frac{\phi^{n+1}-\phi^{n+1/2}}{\frac{1}{2}\!dt}\,v + \frac{\mu}{2}\nabla\left(\frac{\phi^{n+1}-\phi^{n+1/2}}{\frac{1}{2}\!dt}\right)\cdot\nabla v \hspace{8em}\\
  + \eta^{n+1}\,v + \frac{\epsilon}{2}\nabla\phi^{n+1/2}\cdot\nabla\phi^{n+1/2}\,v \,dx\,dy &= 0 \\\\
  %
  \int_{\Omega} q^{n+1}\,v - \frac{2}{3}\nabla\phi^{n+1}\cdot\nabla v \,dx\,dy &= 0.

Furthermore, we note that the Benney-Luke equations admit asymptotic solutions (correct up to order :math:`\epsilon`). The "exact" solutions can be found by assuming one-dimensional travelling waves of the type

.. math::

  \eta(x,y,t) = \eta(\xi,\tau),\quad \phi(x,y,t) = \Phi(\xi,\tau), \qquad \text{with} \qquad \xi = \sqrt{\frac{\epsilon}{\mu}}(x-t), \quad \tau = \epsilon\sqrt{\frac{\epsilon}{\mu}}t, \quad \Phi = \sqrt{\frac{\epsilon}{\mu}}\phi.

The Benney-Luke equations then become equivalent to a Korteweg-de Vries (KdV) equation for :math:`\eta` at leading order in :math:`\epsilon`. The soliton solution of the KdV :cite:`1989:KdV` travels with speed :math:`c` and is reflected when reaching the solid wall. The initial propagation before reflection matches the asymptotic solution for the surface elevation :math:`\eta` well. The asymptotic solution for the surface potential :math:`\phi` can be found by using :math:`\eta=\phi_{\xi}` (correct at leading order), giving

.. math::

  \eta(x,y,t) &= \frac{c}{3}{\rm sech}^2 \left( \frac{1}{2}\sqrt{\frac{c\epsilon}{\mu}} \left(x-x_0-t-\frac{\epsilon}{6}ct\right) \right), \\
  \phi(x,y,t) &= \frac{2}{3}\sqrt{\frac{c\mu}{\epsilon}}\,\left( {\rm tanh}\left(\frac{1}{2}\sqrt{\frac{c\epsilon}{\mu}} \left(x-x_0-t-\frac{\epsilon}{6}ct\right) \right)+1 \right).

Finally, before implementing the problem in Firedrake, we calculate the total energy defined by the sum of potential and kinetic energy. The system is then stable if the energy is bounded and shows no drift. The expression for total energy is given by:

.. math::

  E(t) = \int_{\Omega} \frac{1}{2}\eta^2 + \frac{1}{2}\!\left(1+\epsilon\eta\right)\left|\nabla\phi\right|^2 + \mu\left( \nabla q\cdot \nabla\phi - \frac{3}{4}q^2 \right) \,dx\,dy.

The implementation of this problem in Firedrake requires solving two nonlinear variational problems and one linear problem. The Benney-Luke equations are solved in a rectangular domain :math:`\Omega=[0,10]\times[0,1]`, with :math:`\mu=\epsilon=0.01`, time step :math:`dt=0.005` and up to the final time :math:`T=2.0`. Additionally, the domain is split into 50 cells in the x-direction using a quadrilateral mesh. In the y-direction only 1 cell is enough since there are no variations in y::

  from firedrake import *

Now we move on to defining parameters::

  T = 2.0
  dt = 0.005
  Lx = 10
  Nx = 50
  Ny = 1
  c = 1.0
  mu = 0.01
  epsilon = 0.01

  m = UnitIntervalMesh(Nx)
  mesh = ExtrudedMesh(m, layers=Ny)
  coords = mesh.coordinates
  coords.dat.data[:,0] = Lx*coords.dat.data[:,0]

The function space chosen consists of degree 2 continuous Lagrange polynomials, and the functions :math:`\eta,\,\phi` are initialised to take the exact soliton solutions for :math:`t=0`, centered around the middle of the domain, i.e. with :math:`x_0=\frac{1}{2}L_x`::

  V = FunctionSpace(mesh,"CG",2)

  eta0 = Function(V, name="eta")
  phi0 = Function(V, name="phi")
  eta1 = Function(V, name="eta_next")
  phi1 = Function(V, name="phi_next")
  q1 = Function(V)
  phi_h = Function(V)
  q_h = Function(V)
  ex_eta = Function(V, name="exact_eta")
  ex_phi = Function(V, name="exact_phi")

  q = TrialFunction(V)
  v = TestFunction(V)

  x = SpatialCoordinate(mesh)
  x0 = 0.5 * Lx
  eta0.interpolate(1/3.0*c*pow(cosh(0.5*sqrt(c*epsilon/mu)*(x[0]-x0)),-2))
  phi0.interpolate(2/3.0*sqrt(c*mu/epsilon)*(tanh(0.5*sqrt(c*epsilon/mu)*(x[0]-x0))+1))

Firstly, :math:`\phi` is updated to a half-step value using a nonlinear variational solver to solve the implicit equation::

  Fphi_h = ( v*(phi_h-phi0)/(0.5*dt) + 0.5*mu*inner(grad(v),grad((phi_h-phi0)/(0.5*dt)))
             + v*eta0 + 0.5*epsilon*inner(grad(phi_h),grad(phi_h))*v )*dx

  phi_problem_h = NonlinearVariationalProblem(Fphi_h,phi_h)
  phi_solver_h = NonlinearVariationalSolver(phi_problem_h)

followed by a calculation of a half-step solution :math:`q`, performed using a linear solver::

  aq = v*q*dx
  Lq_h = 2.0/3.0*inner(grad(v),grad(phi_h))*dx

  q_problem_h = LinearVariationalProblem(aq,Lq_h,q_h)
  q_solver_h = LinearVariationalSolver(q_problem_h)

Then the nonlinear implicit equation for :math:`\eta` is solved::

  Feta = ( v*(eta1-eta0)/dt + 0.5*mu*inner(grad(v),grad((eta1-eta0)/dt))
           - 0.5*((1+epsilon*eta0)+(1+epsilon*eta1))*inner(grad(v),grad(phi_h))
           - mu*inner(grad(v),grad(q_h)) )*dx

  eta_problem = NonlinearVariationalProblem(Feta,eta1)
  eta_solver = NonlinearVariationalSolver(eta_problem)

and finally the second half-step (explicit this time) for the equation of :math:`\phi` is performed and :math:`q` is computed for the updated solution::

  Fphi = ( v*(phi1-phi_h)/(0.5*dt) + 0.5*mu*inner(grad(v),grad((phi1-phi_h)/(0.5*dt)))
           + v*eta1 + 0.5*epsilon*inner(grad(phi_h),grad(phi_h))*v )*dx

  phi_problem = NonlinearVariationalProblem(Fphi,phi1)
  phi_solver = NonlinearVariationalSolver(phi_problem)

  Lq = 2.0/3.0*inner(grad(v),grad(phi1))*dx
  q_problem = LinearVariationalProblem(aq,Lq,q1)
  q_solver = LinearVariationalSolver(q_problem)

What is left before iterating over all time steps, is to find the initial energy :math:`E_0`, used later to evaluate the energy difference :math:`\left|E-E_0\right|/E_0`::

  t = 0
  E0 = assemble( (0.5*eta0**2 + 0.5*(1+epsilon*eta0)*abs(grad(phi0))**2
                  + mu*(inner(grad(q1),grad(phi0)) - 0.75*q1**2))*dx )
  E = E0

and define the exact solutions, which need to be updated at every time-step::

  t_ = Constant(t)
  expr_eta = 1/3.0*c*pow(cosh(0.5*sqrt(c*epsilon/mu)*(x[0]-x0-t_-epsilon*c*t_/6.0)),-2)
  expr_phi = 2/3.0*sqrt(c*mu/epsilon)*(tanh(0.5*sqrt(c*epsilon/mu)*(x[0]-x0-t_-epsilon*c*t_/6.0))+1)

Since we will interpolate these values again and again, we use an
:class:`~.Interpolator` whose :meth:`~.Interpolator.interpolate`
method we can call to perform the interpolation. ::

  eta_interpolator = Interpolator(expr_eta, ex_eta)
  phi_interpolator = Interpolator(expr_phi, ex_phi)
  phi_interpolator.interpolate()
  eta_interpolator.interpolate()

For visualisation, we save the computed and exact solutions to
an output file.  Note that the visualised data will be interpolated
from piecewise quadratic functions to piecewise linears::

  output = VTKFile('output.pvd')
  output.write(phi0, eta0, ex_phi, ex_eta, time=t)

We are now ready to enter the main time iteration loop::

  while t < T:
        print(t, abs((E-E0)/E0))
        t += dt

        t_.assign(t)

        eta_interpolator.interpolate()
        phi_interpolator.interpolate()

        phi_solver_h.solve()
        q_solver_h.solve()
        eta_solver.solve()
        phi_solver.solve()
        q_solver.solve()

        eta0.assign(eta1)
        phi0.assign(phi1)

        output.write(phi0, eta0, ex_phi, ex_eta, time=t)

        E = assemble( (0.5*eta1**2 + 0.5*(1+epsilon*eta1)*abs(grad(phi1))**2
                     + mu*(inner(grad(q1),grad(phi1)) - 0.75*q1**2))*dx )


The output can be visualised using `paraview <http://www.paraview.org/>`__.

A python script version of this demo can be found :demo:`here <benney_luke.py>`.

The Benney-Luke system and weak formulations presented in this demo have also been used to model extreme waves that occur due to Mach reflection through the intersection of two obliquely incident solitary waves. More information can be found in :cite:`Gidel:2017`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
