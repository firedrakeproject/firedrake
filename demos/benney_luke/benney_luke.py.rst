Benney-Luke equations: a reduced water wave model
=================================================

The Benney-Luke-type equations consist a reduced potential flow water wave model based on the assumption of small amplitude parameter :math:`\epsilon` and small dispersion parameter :math:`\mu` (defined by the square of the ratio of the typical depth over a horizontal length scale). They describe the deviation from the still water surface :math:`\eta(x,y,t)` and the free surface potential :math:`\phi(x,y,t)` and are given by:

.. math::

  \phi_t - \frac{\mu}{2}\Delta\phi_t + \eta + \frac{1}{2}\epsilon\left|\nabla\phi\right|^2 = 0

  \eta_t - \frac{\mu}{2}\Delta\eta_t + \nabla\cdot\bigl((1+\epsilon\eta)\nabla\phi\bigr)+\mu\Delta q = 0

  q = - \frac{2}{3}\Delta\phi

where the spatial domain is assumed to be :math:`\Omega` with natural boundary conditions, namely Neumann conditions on all the boundaries. The solutions to these equations are the three functions :math:`\eta,\phi,q\in V` for a suitable function space :math:`V`. Note that the introduction of the auxiliary function :math:`q` is advantageous in a :math:`C^0` finite element formulation. The equations are then multiplied by a test function :math:`v\in V` and integrated over the domain in order to obtain a weak formulation, given by

.. math::

  \int_{\Omega} \phi_t\,v + \frac{\mu}{2}\nabla\phi_t\cdot\nabla v + \eta\,v + \frac{\epsilon}{2}\nabla\phi\cdot\nabla\phi\,v \,dx = 0

  \int_{\Omega} \eta_t\,v + \frac{\mu}{2}\nabla\eta_t\cdot\nabla v - \left(1+\epsilon\eta\right)\nabla\phi\cdot\nabla v - \mu\nabla q\cdot\nabla v \,dx = 0

  \int_{\Omega} q\,v - \frac{2}{3}\nabla\phi\cdot\nabla v \,dx = 0

Note that the Neumann boundary conditions have been used to remove every surface term that resulted from the integration by parts. Moreover, for the time-discretisation a combination of the forward/backward Euler method is used, yielding:

.. math::

  \int_{\Omega} \frac{\phi^{n+1}-\phi^n}{dt}\cdot v + \frac{\mu}{2}\nabla\left(\frac{\phi^{n+1}-\phi^n}{dt}\right)\cdot\nabla v + \eta^n\,v + \frac{\epsilon}{2}\nabla\phi^{n+1}\cdot\nabla\phi^{n+1}\,v \,dx = 0

  \int_{\Omega} q^{n+1}\,v - \frac{2}{3}\nabla\phi^{n+1}\cdot\nabla v \,dx = 0

  \int_{\Omega} \frac{\eta^{n+1}-\eta^n}{dt}\cdot v + \frac{\mu}{2}\nabla\left(\frac{\eta^{n+1}-\eta^n}{dt}\right)\cdot\nabla v - \left(1+\epsilon\eta^n\right)\nabla\phi^{n+1}\cdot\nabla v - \mu\nabla q^{n+1}\cdot\nabla v \,dx = 0

Furthermore, we note that the Benney-Luke equations admit asymptotic solutions (correct up to order :math:`\epsilon`). The "exact" solutions can be found by assumming one-dimensional travelling waves of the type

.. math::

  \eta(x,y,t) = \eta(\xi,\tau),\quad \phi(x,y,t) = \Phi(\xi,\tau), \qquad \text{with} \qquad \xi = \sqrt{\frac{\epsilon}{\mu}}(x-t), \quad \tau = \epsilon\sqrt{\frac{\epsilon}{\mu}}t, \quad \Phi = \sqrt{\frac{\epsilon}{\mu}}\phi

The Benney-Luke equations then become equivalent to a KdV equation for :math:`\eta` at leading order in :math:`\epsilon`. The soliton solution of the KdV travels with speed :math:`c` and gets reflected when reaching  the solid wall. The initial propagation before reflection matches the asymptotic solution for the surface elevation :math:`\eta` well. The asymptotic solution for the surface potential :math:`\phi` can be found by using :math:`\eta=\phi_{\xi}` (correct at leading order), giving

.. math::

  \eta(x,y,t) = \frac{1}{2}c\,{\rm sech}^2 \left( \frac{1}{2}\sqrt{\frac{3c\epsilon}{2\mu}} \left(x-x_0-t-\frac{1}{4}\epsilon ct\right) \right), \quad \phi(x,y,t) = \sqrt{\frac{2c\mu}{3\epsilon}}\,\left( {\rm tanh}\left(\frac{1}{2}\sqrt{\frac{3c\epsilon}{2\mu}} \left(x-x_0-t-\frac{1}{4}\epsilon ct\right) \right)+1 \right)

Finally, before implementing the problem in Firedrake, we calculate the total energy of the system and make sure that it shows no drift. The expression for total energy is given by:

.. math::

  E(t) = \int_{\Omega} \frac{1}{2}\left(1+\epsilon\eta\right)\left|\nabla\phi\right|^2 + \frac{1}{2}\eta^2 + \mu\left( \nabla q\cdot \nabla\phi - \frac{3}{4}q^2 \right) \,dx

The implementation of this problem in Firedrake requires solving two nonlinear variational problems and one linear problem, i.e. a Laplace equation for :math:`q`. The Benney-Luke equations are solved in a rectangular domain :math:`\Omega=[0,10]\times[0,1]`, with :math:`\mu=\epsilon=0.01`, time step :math:`dt=0.005` and up to the final time :math:`T=2.0`. Additionally, the domain is split into 50 cells in the x-direction using a quadrilateral mesh. In the y-direction only 1 cell is enough since there are no variations in y::

  from firedrake import *

  T = 2.0
  dt = 0.005
  mu = 0.01
  epsilon = 0.01
  Lx = 10

  m = UnitIntervalMesh(50)
  mesh = ExtrudedMesh(m, layers=1)
  coords = mesh.coordinates
  coords.dat.data[:,0] = Lx*coords.dat.data[:,0]

The function space consists of degree 2 continuous Lagrange polynomials, and the functions :math:`\eta,\,\phi` are initialised to take the exact soliton solutions for :math:`t=0`, centered around the middle of the domain, i.e. with :math:`x_0=\frac{1}{2}L_x`::

  V = FunctionSpace(mesh,"CG",2)

  eta0 = Function(V)
  phi0 = Function(V)
  eta1 = Function(V)
  phi1 = Function(V)
  q1 = Function(V)
  ex_eta = Function(V)
  ex_phi = Function(V)

  q = TrialFunction(V)
  v = TestFunction(V)

  eta0.interpolate(Expression("1/3.0*c*pow(cosh(0.5*sqrt(c*epsilon/mu)*(x[0]-x0)),-2)",
                              c=1.5, epsilon=epsilon, mu=mu, x0=0.5*Lx))
  phi0.interpolate(Expression("2/3.0*sqrt(c*mu/epsilon)*(tanh(0.5*sqrt(c*epsilon/mu)*(x[0]-x0))+1)",
                              c=1.5, epsilon=epsilon, mu=mu, x0=0.5*Lx))

The equation for :math:`\phi` is first solved using a nonlinear variational solver::

  Fphi = ( v*(phi1-phi0)/dt + 0.5*mu*inner(grad(v),grad((phi1-phi0)/dt)) + v*eta0
           + 0.5*epsilon*inner(grad(phi1),grad(phi1))*v )*dx

  phi_problem = NonlinearVariationalProblem(Fphi,phi1)
  phi_solver = NonlinearVariationalSolver(phi_problem)

followed by the Laplace equation, which is solved using a linear solver::

  aq = v*q*dx
  Lq = 2/3.0*inner(grad(v),grad(phi1))*dx

  q_problem = LinearVariationalProblem(aq,Lq,q1)
  q_solver = LinearVariationalSolver(q_problem)

and finally the nonlinear equation for :math:`\eta`::

  Feta = ( v*(eta1-eta0)/dt + 0.5*mu*inner(grad(v),grad((eta1-eta0)/dt))
           - (1+epsilon*eta0)*inner(grad(v),grad(phi1)) - mu*inner(grad(v),grad(q1)) )*dx

  eta_problem = NonlinearVariationalProblem(Feta,eta1)
  eta_solver = NonlinearVariationalSolver(eta_problem)

For visualisation reasons, we print the results in files::

  phi_file = File('phi.pvd')
  eta_file = File('eta.pvd')
  phi_exact = File('phi_ex.pvd')
  eta_exact = File('eta_ex.pvd')

  phi_file << phi0
  eta_file << eta0
  phi_exact << phi0
  eta_exact << eta0

What is left before iterating over all time steps, is to find the initial energy :math:`E_0` use it to later evaluate the energy difference :math:`\left|E-E_0\right|/E_0`::

  t = 0
  E0 = assemble( (0.5*(1+epsilon*eta0)*abs(grad(phi0))**2 + 0.5*eta0**2
                  + mu*(inner(grad(q0),grad(phi0)) - 0.75*q1**2))*dx )
  E = E0

and define the exact solutions::

  expr_eta = Expression("1/3.0*c*pow(cosh(0.5*sqrt(c*epsilon/mu)*(x[0]-x0-t-epsilon*c*t/6.0)),-2)",
                        t=t, c=1.5, epsilon=epsilon, mu=mu, x0=0.5*Lx)
  expr_phi = Expression("2/3.0*sqrt(c*mu/epsilon)*(tanh(0.5*sqrt(c*epsilon/mu)*(x[0]-x0-t-epsilon*c*t/6.0))+1)",
                        t=t, c=1.5, epsilon=epsilon, mu=mu, x0=0.5*Lx)

We are now ready to enter the main time iteration loop::

  while(t < T):
        print t, abs((E-E0)/E0)
        t += dt

        expr_eta.t = t
        expr_phi.t = t

        ex_phi.interpolate(expr_phi)
        ex_eta.interpolate(expr_eta)

        phi_solver.solve()
        q_solver.solve()
        eta_solver.solve()

        eta0.assign(eta1)
        phi0.assign(phi1)

        phi_file << phi0
        eta_file << eta0
        phi_exact << ex_phi
        eta_exact << ex_eta

        E = assemble( (0.5*(1+epsilon*eta1)*abs(grad(phi1))**2 + 0.5*eta1**2
                     + mu*(inner(grad(q1),grad(phi1)) - 0.75*q1**2))*dx )


The output files can be visualised using `paraview <http://www.paraview.org/>`__.

A python script version of this demo can be found `here <benney_luke.py>`__.
