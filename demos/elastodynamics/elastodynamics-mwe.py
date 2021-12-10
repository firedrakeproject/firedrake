# Elastodynamics verification
# ===========================
# Here we solve an elastodynamics equation using an explicit
# timestepping scheme. This example demonstrates the use of pointwise 
# operations on Functions and a time-varying Neumann boundary condition. 
# The strong form of the equation we set out to solve is:

# .. math::

#    \rho\mathbf{a} - \nabla\cdot\mathbf{\sigma} = 0

#    \mathbf{\sigma} \cdot \mathbf{n} = \mathbf{t}^\sigma = -f(t) \ \textrm{on}\ \Gamma_N

#    \mathbf{u}\cdot\mathbf{n} = 0 \ \textrm{on}\ \Gamma_D

#    f(t) = \frac{f_0}{2}\left(1-\cos(\omega t)\right)

# where we have assumed linear elasticity such that:

# .. math::

#   \mathbf{\sigma} = \lambda\mathrm{tr}(\mathbf{u}) + 2\mu\epsilon(\mathbf{u})

#   \epsilon(\mathbf{u}) = \frac{1}{2}\left(\nabla\mathbf{u} + \left(\nabla\mathbf{u}\right)^T\right)

# We employ an explicit, central-differencing scheme. The acceleration update is written in terms of the displacements as:

# .. math::

#     \mathbf{a}_{n+1} = \frac{\mathbf{u}_{n+1} - 2\mathbf{u}_n + \mathbf{u}_{n-1}}{\Delta t^2}

# We then write the weak form of the equation for :math:`\mathbf{u}`. Find
# :math:`\mathbf{u} \in \mathbf{W}` such that:

# .. math::

#    \int_\Omega \frac{\rho}{\Delta t^2}\left(\mathbf{u}_{n+1} - 2\mathbf{u}_n + \mathbf{u}_{n-1}\right)\cdot\mathbf{w} d\Omega + \int_\Omega\mathbf{\sigma}_{n+1}:\epsilon(\mathbf{w}) d\Omega = \int_{\partial\Omega}\mathbf{t}\cdot\mathbf{w} dS
#    \quad \forall \mathbf{w} \in \mathbf{W}

# for a suitable function space :math:`\mathbf{W}`.

# This problem has an analytical solution that was developed in Eringen & Suhubi (1975):

# .. math::
#   u(z,t) = \frac{4}{\pi\rho c}\sum_{n=1}^\infty\frac{(-1)^n}{2n - 1}\left[\int_0^t f(\tau)\sin\left(\frac{(2n-1)\pi c(t-\tau)}{2H}\right)\dtau\right]\sin\left(\frac{(2n-1)\pi z}{2H}\right)

from firedrake import *
import math, sys
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1.0e-14

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lmbda*div(u)*Identity(3) + 2*mu*epsilon(u)

# Select whether or not the mass matrix will be lumped::
lump_mass = True

# Set time parameters::
T                = 0.4
dt               = 1e-3
Nsteps_numerical = int(T/dt)
t                = 0.
step             = 0

# We define Young's modulus and Poisson's ratio::
E  = 50.0e6
nu = 0.3

# Next we calculate Lame's constants::
lmbda   = E*nu/((1 + nu)*(1 - 2*nu))
mu      = E/(2*(1 + nu))
rho     = 1986.

# The loading parameters are::
amplitude = 40.0e3
omega     = 50.

# We create a rectangular column mesh where the number of elements in the
# :math:`z`-direction is allowed to vary to demonstrate convergence::
height = 20.
width  = 1.
nx     = 1
ny     = 1
nz     = 20
mesh   = BoxMesh(nx, ny, nz, width, width, height)

# Set up outputs::
u_numerical    = np.zeros((Nsteps_numerical+2))
time_numerical = np.linspace(0, T, Nsteps_numerical+2)

# We first compute all surface areas of the mesh and then assign
# the differential area element :math: `\Gamma_N`:: at the top surface
ds      = Measure('ds', mesh)
Gamma_N = ds(6)

# We choose a degree 1 continuous vector function space, and set up
# the function space and functions::
W = VectorFunctionSpace(mesh, 'Lagrange', 1)
u = TrialFunction(W)
w = TestFunction(W)

u_n   = Function(W, name='displacement')
u_np1 = Function(W)
u_nm1 = Function(W)

# We next establish the Dirichlet boundary conditions::
fixedSides  = Constant(0.)
fixedBottom = as_vector((0.,0.,0.))

fixedLeft_BC_x  = DirichletBC(W.sub(0), fixedSides, 1)
fixedLeft_BC_y  = DirichletBC(W.sub(1), fixedSides, 1)
fixedRight_BC_x = DirichletBC(W.sub(0), fixedSides, 2)
fixedRight_BC_y = DirichletBC(W.sub(1), fixedSides, 2)
fixedBack_BC_x  = DirichletBC(W.sub(0), fixedSides, 3)
fixedBack_BC_y  = DirichletBC(W.sub(1), fixedSides, 3)
fixedFront_BC_x = DirichletBC(W.sub(0), fixedSides, 4)
fixedFront_BC_y = DirichletBC(W.sub(1), fixedSides, 4)
fixedBottom_BC  = DirichletBC(W, fixedBottom, 5)

bcSet = [fixedLeft_BC_x, fixedRight_BC_x, fixedBack_BC_x, fixedFront_BC_x, \
         fixedLeft_BC_y, fixedRight_BC_y, fixedBack_BC_y, fixedFront_BC_y, \
         fixedBottom_BC]

# Next we define the traction vector which will be updated in the time loop::
t = Constant(t)
traction = as_vector([0, 0, -0.5*amplitude*(1 - cos(omega*t))])

# Next we define the bilinear forms::
a = dot(u, w)*dx
L = (dt**2/rho) * (dot(traction, w)*Gamma_N - inner(sigma(u_n), epsilon(w))*dx) - dot(u_nm1 - 2*u_n, w)*dx

print("Computing numerical solution...")

if lump_mass:
    
    alump = replace(a, {u: as_vector([1, 1, 1])})
    lumped_mass = assemble(alump)

    # Time-stepping loop::
    for step in range(1, Nsteps_numerical + 2):
        # Update time step::
        t.assign(t + dt)

        assemble(L, tensor=u_np1, bcs=bcSet)

        # Divide by lumped mass matrix::
        u_np1 /= lumped_mass

        # Update solutions for next time step::
        u_nm1.assign(u_n)
        u_n.assign(u_np1)

        # Save data::
        u_numerical[step] = u_n([0., 0., 20.])[2]

else:
    
    # Time-stepping loop::
    for step in range(1, Nsteps_numerical + 2):
        # Update time step::
        t.assign(t + dt)

        solve(a == L, u_np1, bcSet)

        # Update solutions for next time step::
        u_nm1.assign(u_n)
        u_n.assign(u_np1)
        
        # Save data::
        u_numerical[step] = u_n([0., 0., 20.])[2]

# Next we formulate the analytical solution developed by Eringen & Suhubi::
print("Computing analytical solution...")

def get_integralSine(a_t, a_tau, a_m):
  return np.sin((2*a_m - 1)*math.pi*c_p*(a_t - a_tau)/(2*height))

def get_fourierSine(a_m, a_z):
  return math.sin((2*a_m - 1)*math.pi*a_z/(2*height))

def get_Load(a_tau):
  return 0.5*amplitude*(1 - np.cos(omega*a_tau))

# Define number of series terms::
mseries = 100

# Initialize storage arrays::
dt_analytical     = 1e-3
Nsteps_analytical = int(T/dt_analytical)
time_analytical   = np.linspace(0, T, Nsteps_analytical+1)
u_analytical      = np.zeros((Nsteps_analytical+1))
integral          = np.zeros((Nsteps_analytical+1))
u_m               = np.zeros((mseries+1, Nsteps_analytical+1))

# Calculate p-wave modulus and resultant p-wave speed::
M   = E*((1 - nu)/((1 + nu)*(1 - 2*nu)))
c_p = np.sqrt(M/rho)

# Build up sine terms::
for m in range(1, mseries+1):
    term1 = (-1)**m/(2*m - 1)
    term2 = get_fourierSine(m, height)

    # Perform integral with time :math:`\tau`::
    for k in range(1, Nsteps_analytical):
        tau  = np.linspace(0, time_analytical[k+1], Nsteps_analytical)
        dtau = tau[1] - tau[0]

        load          = get_Load(tau)
        sine_term     = get_integralSine(time_analytical[k+1], tau, m)
        integral[k+1] = np.sum(load*sine_term)*dtau

    # Write one sine term::
    u_m[m, :] = integral*term1*term2

# Sum sine terms::
u_analytical = (4/(math.pi*rho*c_p))*np.sum(u_m[:,:], axis=0)

# Next plot and save one solution::
plt.figure(1)
plt.plot(time_numerical[::10], u_numerical[::10]*1e3, 'ro', label='Numerical solution, lumping = %r' %lump_mass)
plt.plot(time_analytical, u_analytical*1e3, 'r--', label='Analytical solution')
plt.legend(loc='lower right')
plt.ylabel('Displacement (mm)')
plt.xlabel('Time (s)')
plt.title('Comparison of solutions for Eringen\'s problem')
plt.xlim([0, 0.4])
plt.show()
