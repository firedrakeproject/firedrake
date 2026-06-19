# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A time-dependent problem, Burgers' equation
#
# We will solve the viscous Burgers equation, a nonlinear equation for the advection and diffusion on momentum in one dimension:
#
# $$
# \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0.
# $$
#
# We will solve on a periodic interval mesh, and therefore do not impose any boundary conditions.  As usual, we need to derive a variational form.
#
# ## Spatial discretisation
#
# We first discretise in space, mulitplying by a test function $v \in V$ and integrating the viscosity term by parts to obtain the semi-discrete problem. Find $u(x, t) \in V$ such that
#
# $$
# \int_\Omega \frac{\partial u}{\partial t} v + u \frac{\partial u}{\partial x} v + \nu \frac{\partial u}{\partial x}\frac{\partial v}{\partial x} \, \mathrm{d}x = 0 \quad \forall v \in V.
# $$
#
# ## Time discretisation
# We now need to discretise in time.  For simplicity, and stability we'll use backward Euler, replacing all instances of $u$ with $u^{n+1}$ and the time derivative by $\frac{u^{n+1} - u^n}{\Delta t}$.  We end up with the discrete problem, find $u^{n+1} \in V$ such that
#
# $$
# \int_\Omega \frac{u^{n+1} - u^n}{\Delta t} v + u^{n+1} \frac{\partial u^{n+1}}{\partial x} v + \nu \frac{\partial u^{n+1}}{\partial x}\frac{\partial v}{\partial x} \, \mathrm{d}x = 0 \quad \forall v \in V.
# $$
#
#

# %% [markdown]
# ## Implementation
#
# To solve the problem in a concrete setting, we need two things: a domain, and an initial condition for $u$.  For the former, we'll choose a periodic interval of length 2, for the latter, we'll start with $u = \sin(2 \pi x)$.
#
# In addition we need to choose the viscosity, which we will set to a small constant value $\nu = 10^{-2}$.
#
# As ever, we begin by importing Firedrake:

# %%
# Code in this cell makes plots appear an appropriate size and resolution in the browser window
# %matplotlib widget
# %config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (11, 6)

# %%
from firedrake import *
from numpy import linspace

n = 100
mesh = PeriodicIntervalMesh(n, length=2)

x = SpatialCoordinate(mesh)[0]

u_init = sin(2*pi*x)

# %%
nu = Constant(1e-2)

# %% [markdown]
# We choose degree 2 piecewise continuous Lagrange polynomials for our solution and test space:

# %%
V = FunctionSpace(mesh, "Lagrange", 2)

# %% [markdown]
# We also need solution functions for $u^{n+1}$ and $u^n$, along with a test function $v$.

# %%
u_n1 = Function(V, name="u^{n+1}")
u_n = Function(V, name="u^{n}")
v = TestFunction(V)

# %% [markdown]
# We provide the initial condition for $u_n$, and choose a $\Delta t$ such that the advective Courant number is around 1.  This is more restrictive than required for stability of the time integration, but gives us enough accuracy to see the temporal evolution of the system.

# %%
u_n.interpolate(u_init)
dt = 1.0 / n

# %% [markdown]
# Now we're ready to define the variational form.  Since this problem is nonlinear, note that we do not have a trial function anywhere.  We just write down the residual, Firedrake will automatically compute the Jacobian by differentiating the residual inside the nonlinear solver.

# %%
F = (((u_n1 - u_n)/dt) * v +
     u_n1 * u_n1.dx(0) * v + 
     nu*u_n1.dx(0)*v.dx(0))*dx

# %% [markdown]
# For visualisation purposes, we will save a copy of the state $u_n$ at each timestep, we can plot and animate these in the notebook if the `ipywidgets` package is installed.

# %%
# If passed an existing Function object, the Function 
# constructor makes a copy.
results = [Function(u_n)]

# %% [markdown]
# Finally, we loop over the timesteps, solving the equation and advancing in time. We use `firedrake.ProgressBar` to provide a visual indication of the progress of the simulation.
#

# %%
t_end=0.5
for t in ProgressBar("Time step").iter(linspace(0.0, t_end, int(t_end/dt))):
    solve(F == 0, u_n1)
    u_n.assign(u_n1)
    results.append(Function(u_n))

# %% [markdown]
# To visualize the results, we'll create a movie using matplotlib's animation tools.
# First, we'll create a figure and axes to draw on and plot the initial values.

# %%
help(plot)

# %% tags=["nbval-ignore-output"]
from firedrake.pyplot import plot

fig, axes = plt.subplots()
axes.set_ylim((-1., 1.))
plot(results[0], axes=axes)

# %% [markdown]
# Next, we'll create a function that tells matplotlib how to draw each frame of the animation, which in our case will just be plotting the value at that timestep.
# The `FuncAnimation` function will call this on the list of results that we pass in, together with a given interval in milliseconds between each frame.
# Finally, we'll use the IPython API to render the animation in the notebook.

# %% tags=["nbval-ignore-output"]
from matplotlib.animation import FuncAnimation

def animate(u):
    axes.clear()
    plot(u, axes=axes)
    axes.set_ylim((-1., 1.))

interval = 4e3 * float(dt)
animation = FuncAnimation(fig, animate, frames=results, interval=interval)

from IPython.display import HTML
HTML(animation.to_jshtml())

# %% [markdown]
# ## A faster implementation
#
# Although the code we wrote above works fine, it can be quite slow.  In particular, each call to `solve` necessitates rederiving the symbolic Jacobian, building new matrices and vectors and solver objects, using them once, and then destroying them.  To avoid this, we can create a solver object and reuse it.
#
# This is what the `solve` call does internally, only it then immediately discards all of this work.
#
# We start by creating a `NonlinearVariationalProblem` which gathers the information about the problem.  The residual, the solution variable, any boundary conditions, and so forth.

# %%
problem = NonlinearVariationalProblem(F, u_n1)

# %% [markdown]
# Now we create a `NonlinearVariationalSolver`.  Here we provide the problem to be solved, and any options to the solver.
#
# Note that the default solver options simply apply a full LU factorisation as a preconditioner. In one dimension, this produces no fill and is, obviously, an exact solve.

# %%
solver = NonlinearVariationalSolver(problem)

# %% [markdown]
# Now we just write the time loop as before, but instead of writing `solve(F == 0, u_n1)`, we just call the `solve` method on our `solver` object.

# %%
t = 0
t_end = 0.5
while t <= t_end:
    solver.solve()
    u_n.assign(u_n1)
    t += dt

# %% [markdown]
# ## Exercise 1
#
# Compare the speed of the two implementation choices on a mesh with 1000 elements.
#
# - Hint: You can use the "notebook magic" `%%timeit` to time the execution of a notebook cell.

# %%

# %% [markdown]
# ## Exercise 2
#
# Implement Crank-Nicolson timestepping instead of backward Euler.
#
# - Hint 1: The Crank-Nicolson scheme writes:
#
#    $$\frac{\partial u}{\partial t} + G(u) = 0$$
#
#   as
#
#   $$ \frac{u^{n+1} - u^n}{\Delta t} + \frac{1}{2}\left[G(u^{n+1}) + G(u^n)\right] = 0$$
#
#
# - Hint 2: It might be convenient to write a python function that returns $G(u)$ given a $u$.

# %%
