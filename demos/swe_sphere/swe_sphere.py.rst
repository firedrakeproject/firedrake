Shallow water solver on the sphere
==========================================

This demo solves the shallow water equations on the sphere of radius
:math:`R_0`, denoted :math:`\Omega`, in vector-invariant form

.. math::

   u_t + q F^{\perp} + \nabla \left(g(D+b) + \frac{1}{2}|u|^2 \right) = 0 \ \textrm{on}\ \Omega

   D_t + \nabla\cdot(uD) = 0 \ \textrm{on}\ \Omega

where :math:`u` is the velocity field, tangent to the sphere,
:math:`D` is the layer depth, :math:`b` is the bottom topography,
:math:`f=2|\Omega|z/R_0` is the Coriolis parameter, :math:`g` is the
acceleration due to gravity, :math:`F=uD` is the mass flux,
:math:`F^{\perp}=k\times F` where :math:`k` is the unit vector normal
to the sphere surface, and :math:`q` is the potential vorticity, given
by

.. math::
   q = \frac{\nabla^{\perp}u + f}{D}

where :math:`\nabla^{\perp}u = k\cdot \nabla\times u`.

We begin by initialising a sphere mesh. ::

   from firedrake import *
   op2.init(log_level = "WARNING")
   #Earth radius in metres
   mesh = IcosahedralSphereMesh(radius = 6371220, refinement_level = 5)

To use H(div) spaces such as :math:`BDM1`, it is necessary to determine
global cell orientations, which is set via an expression. ::

   #Define global normal
   global_normal = Expression(("x[0]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])" ,
      "x[1]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])" ,
      "x[2]/sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])"))
   mesh.init_cell_orientations(global_normal)


To build a finite element spatial discretisation, we choose compatible
finite element spaces :math:`V_0,\,V_1\,V_2` (in this demo we take
:math:`P2,\,BDM1,\,DG0`), and choose :math:`u\in V_1,\,D,b\in
V_2`. These fields are initialised from expressions. ::

   degree = 1
   V0 = FunctionSpace(mesh , "CG" , degree+1)
   V1 = FunctionSpace(mesh , "BDM" , degree)
   V2 = FunctionSpace(mesh , "DG" , degree-1)
   #Initial data
   uexpr = Expression(("-20.0*x[1]/6.37122e6" , "20.0*x[0]/6.37122e6" , "0.0"))
   Dexpr = Expression("5960 - ((6.37122e6 * 7.292e-5 * 20.0 + pow(20.0,2)/2.0)*(x[2]*x[2]/(6.37122e6*6.37122e6)))/9.80616 - (2000 * (1 - sqrt(fmin(pow(pi/9.0,2),pow(atan2(x[1]/6.37122e6,x[0]/6.37122e6)+1.0*pi/2.0,2)+pow(asin(x[2]/6.37122e6)-pi/6.0,2)))/(pi/9.0)))")
   bexpr = Expression("2000 * (1 - sqrt(fmin(pow(pi/9.0,2),pow(atan2(x[1]/6.37122e6,x[0]/6.37122e6)+1.0*pi/2.0,2)+pow(asin(x[2]/6.37122e6)-pi/6.0,2)))/(pi/9.0))")
   u0 = project(uexpr , V1)
   D0 = project(Dexpr , V2)
   b = project(bexpr , V2)

We also set a number of physical constants and timestepping parameters. ::

   #Physical constants
   f = Function(V0).interpolate(Expression("2*7.292e-5*x[2]/6.37122e6")) # Coriolis frequency (1/s)
   g = 9.80616  # gravitational constant (m/s^2)
   T =  1296000.0     # seconds
   dt = 180.0 # seconds
   theta = 0.5 # implicitness parameter
   maxiter = 3 # number of nonlinear iterations
   # get average height
   temp = Function(V2).assign(1.0)
   H = assemble(D0*dx)/assemble(temp*dx)
   t = 0.0

To compute the :math:`\perp` operator, it is necessary to have a
:math:`DG0^3` field that stores the outward element normal. This is
built using the following PyOP2 kernel. ::

   def _outward_normals(mesh):
      kernel = op2.Kernel("""void outward_normal (double** coords , double** normal)
      {
         double v0[3] ;
         double v1[3] ;
         double n[3] ;
         double x[3] ;
         double dot ;
         double norm ;
         norm = 0.0;
         dot = 0.0;
         int i ;
         int c ;
         int d ;
         for (i = 0; i < 3; ++i)
         {
            v0[i] = coords[1][i] - coords[0][i];
            v1[i] = coords[2][i] - coords[0][i];
            x[i] = 0.0;
         }
         /* Compute normal to triangle (v0 x v1)*/
         n[0] = v0[1] * v1[2] - v0[2] * v1[1];
         n[1] = v0[2] * v1[0] - v0[0] * v1[2];
         n[2] = v0[0] * v1[1] - v0[1] * v1[0];
         /* Compute a reference "outward" direction */
         for (i = 0; i < 3; ++i)
         {
            x[0] += coords[i][0];
            x[1] += coords[i][1];
            x[2] += coords[i][2];
         }
         /* Cell may be back to front, so figure out dot(n, outward) */
         dot += (x[0]) * n[0];
         dot += (x[1]) * n[1];
         dot += (x[2]) * n[2];
         norm += n[0] * n[0];
         norm += n[1] * n[1];
         norm += n[2] * n[2];
         norm = sqrt(norm);
         norm *= (dot < 0 ? -1 : 1);
         /* Write normal to output function */
         for (c = 0; c < 3; ++c)
         {
            normal[0][c] = n[c]/norm;
         }
      }""", "outward_normal")
      coords = mesh.coordinates
      fs = VectorFunctionSpace(mesh,"DG",0)
      normal = Function(fs)
      op2.par_loop(kernel, normal.cell_set,
      coords.dat(op2.READ, coords.cell_node_map()),
      normal.dat(op2.WRITE, normal.cell_node_map()))
      return normal

We then call this function to construct the outward normals field and
define expressions for the :math:`\perp` and :math:`\nabla^\perp`
operators. ::

   outward_normals = _outward_normals(mesh)
   perp = lambda u: cross(outward_normals, u)
   gradperp = lambda psi: cross(outward_normals, grad(psi))

Now we turn to the spatial discretisation. The finite element
discretisation of the velocity equation is

.. math::
   :label: ueqn
   
   \int_{\Omega} w \cdot \left( u_t + q F^{\perp}\right) - \nabla\cdot w \left(g(D+b) + \frac{1}{2}|u|^2\right)\ {\rm d} x = 0, \quad \forall w\in V_1,
   

whilst the depth continuity equation holds pointwise and satisfies

.. math::
   :label: deqn

   D_t + \nabla\cdot F = 0,

where :math:`F\in V_1` and :math:`q\in V_0` are functions of :math:`u`
and :math:`D` defined via the following weak approximations (note that
the :math:`q` equation has been multiplied by :math:`D`),

.. math::

   \int_{\Omega} w \cdot  F \ {\rm d} x &= \int_{\Omega} w \cdot  uD \ {\rm d} x, \quad \forall w\in V_1, \\
   \int_{\Omega} \gamma qD {\rm d} x &= -\int_{\Omega} \nabla^\perp \gamma \cdot u {\rm d} x + \int_{\Omega} \gamma f {\rm d} x, \quad \forall \gamma\in V_0.

We discretise Equations :eq:`ueqn` and :eq:`deqn` in time using the 
implicit midpoint method, 

.. math::
   
   \int_{\Omega} w \cdot \left(u^{n+1}-u^n + \Delta t \bar{q} \bar{F}^{\perp}\right) - \Delta t \nabla\cdot w \left(g(\bar{D}+b) + \frac{1}{2}|\bar{u}|^2\right)\ {\rm d} x &= 0, \quad \forall w\in V_1,\\
   D^{n+1}-D^n + \Delta t\nabla\cdot \bar{F} &= 0,

where 

.. math:: 

   \bar{u} = \theta u^{n+1} + (1-\theta)u^n, \quad \bar{D} = \theta D^{n+1} + (1-\theta)D^n,

and where :math:`\bar{F}` and :math:`\bar{q}` are obtained upon substituting
:math:`\bar{u}` and :math:`\bar{D}` into the above weak approximations.

teasasdasdad ::

   du = Function(V1)
   dD = Function(V2)
   utheta = u0 + (1-theta)*du      # implicit velocity
   Dtheta = D0 + (1-theta)*dD      # implicit fluid thickness
   Dbtheta = D0 + b + (1-theta)*dD # implicit surface height
   v = TrialFunction(V1)
   w = TestFunction(V1)
   #Mass flux equation
   aF = inner(v,w)*dx
   LF = inner(utheta*Dtheta,w)*dx
   F = Function(V1)
   #PV equation
   gamma = TestFunction(V0)
   qt = TrialFunction(V0)
   aQ = gamma*qt*Dtheta*dx
   LQ = (-inner(gradperp(gamma),utheta) + gamma*f)*dx
   q = Function(V0)
   qupwind = (q-(dt/2)*dot(utheta,grad(q)))
   phi = TestFunction(V2)
   h = TrialFunction(V2)
   #Mass residual as a pointwise expression
   MassRes = dD + dt*div(F)
   print "This needs upwinding for q for stability."
   URes = (inner(w,du + dt*qupwind*perp(F))
      - dt*div(w)*(g*Dbtheta + 0.5*inner(utheta,utheta)))*dx
   #Wave equation
   print "This could alternatively be implemented in mixed form using Schur complement."
   #D Equation for substitution into u equation
   Dwave = (1-theta)*dt*H*div(v) + MassRes
   #A Equation
   aWave = (inner(w,v) - (1-theta)*dt*div(w)*g*Dwave)*dx + URes
   aWaverhs = rhs(aWave)
   aWavelhs = lhs(aWave)
   #D Equation for updating D having solved for u.
   Dwave_update = phi*(h + (1-theta)*dt*H*div(du) + MassRes)*dx
   Dwave_update_lhs = lhs(Dwave_update)
   Dwave_update_rhs = rhs(Dwave_update)
   out = File("surface_height.pvd")
   #Time loop
   output = Function(V2)
   while(t<T-0.5*dt):
      #Newton loop
      du.assign(0.)
      dD.assign(0.)
      i = 0
      while(i<maxiter):
         i += 1
         #Get Mass Flux
         solve(aF==LF,F)
         #Get PV
         solve(aQ==LQ,q)
         #Get increment
         solve(aWavelhs==aWaverhs,du)
         solve(Dwave_update_lhs==Dwave_update_rhs,dD)
      D0 += dD
      u0 += du
      output.assign(D0 + b)
      out << output
      t += dt
