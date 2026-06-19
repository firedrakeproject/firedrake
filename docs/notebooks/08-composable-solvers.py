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
# # Programming your solver
#
# In this notebook, we will look at some of the more advanced capabilities Firedrake has for configuring and developing preconditioners. In particular, we will show support for geometric multigrid, as well as user-defined preconditioners.
#
# As our prototypical example, we will consider the Stokes equations. Find $(u, p) \in V \times Q \subset (H^1)^d \times L^2$ such that
#
# $$
# \begin{align}
#   \nu\int_\Omega \nabla u : \nabla v\,\mathrm{d}x - \int_\Omega p
#   \nabla \cdot v\,\mathrm{d}x
#   &= \int_\Omega f \cdot v\,\mathrm{d}x, \\
#   -\int_\Omega \nabla \cdot u q \,\mathrm{d}x&= 0.
# \end{align}
# $$
# for all $(v, q) \in V \times Q$. Where $\nu$ is the viscosity.
#
# We will use the inf-sup stable Taylor-Hood element pair of piecewise quadratic velocities and piecewise linear pressures.

# %%
# Code in this cell makes plots appear an appropriate size and resolution in the browser window
# %matplotlib widget
# %config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (11, 6)

# %%
from firedrake import *
mesh = UnitSquareMesh(8, 8)

# %% [markdown]
# We now build a hierarchy of regularly refined meshes with this as the coarsest mesh, and grab the finest one to define the problem.

# %%
meshes = MeshHierarchy(mesh, refinement_levels=3)
# Grab the finest mesh
mesh = meshes[-1]

V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
W = V*Q

# %% [markdown]
# We set up the problem in residual form (using `TestFunction`s but no `TrialFunction`s).

# %%
v, q = TestFunctions(W)
w = Function(W)
u, p = split(w)

nu = Constant(0.0001)
F = nu*inner(grad(u), grad(v))*dx - p*div(v)*dx - div(u)*q*dx

# %% [markdown]
# We now need to augment the problem with a forcing term and boundary conditions.  We will solve a regularised lid-driven cavity problem, and thus choose $f = 0$ and the boundary conditions:
# $$
# \begin{align}
# u &= \begin{pmatrix}\frac{x^2 (2 - x)^2 y^2}{4} \\ 0 \end{pmatrix} & \text{ on $\Gamma_1 = \{y = 1\}$},\\
# u &= 0 & \text{ otherwise.}\\
# \end{align}
# $$

# %%
x, y = SpatialCoordinate(mesh)
bc_value = as_vector([0.25 * x**2 * (2-x)**2 *y**2, 0])

bcs = [DirichletBC(W.sub(0), bc_value, 4),
       DirichletBC(W.sub(0), 0, (1, 2, 3))]

# %% [markdown]
# This problem has a null space of constant pressures, so we'll need to inform the solver about that too.

# %%
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)])


# %% [markdown]
# Since we're going to look at a bunch of different solver options, let's have a function that builds a solver with the provided options.

# %%
def create_solver(solver_parameters, *, pmat=None, appctx=None):
    p = {}
    if solver_parameters is not None:
        p.update(solver_parameters)
    # Default to linear SNES
    p.setdefault("snes_type", "ksponly")
    p.setdefault("ksp_rtol", 1e-7)
    problem = NonlinearVariationalProblem(F, w, bcs=bcs, Jp=pmat)
    solver = NonlinearVariationalSolver(problem, nullspace=nullspace, options_prefix="", 
                                        solver_parameters=p, appctx=appctx)
    return solver


# %% [markdown]
# First, let's go ahead and solve the problem using a direct solver. The solver is configured with a dictionary of PETSc options. Here we select MUMPS to perform the sparse LU factorisation. (Note that these are actually the default solver parameters that Firedrake assumes.)

# %%
solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    # Use MUMPS since it handles the null space
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}


# %%
# Programmatically inspect convergence of solver
def convergence(solver):
    from firedrake.solving_utils import KSPReasons, SNESReasons
    snes = solver.snes
    print("""
SNES iterations: {snes}; SNES converged reason: {snesreason}
   KSP iterations: {ksp}; KSP converged reason: {kspreason}""".format(snes=snes.getIterationNumber(),
                                                                      snesreason=SNESReasons[snes.getConvergedReason()],
                                                                      ksp=snes.ksp.getIterationNumber(),
                                                                      kspreason=KSPReasons[snes.ksp.getConvergedReason()]))


# %% [markdown]
# We're ready to solve.

# %%
w.zero()
solver = create_solver(solver_parameters)
solver.solve()
convergence(solver)

# %% [markdown]
# We can now have a look at the solution, using some simple builtin plotting that utilises matplotlib.

# %%
from firedrake.pyplot import streamplot

u_h, p_h = w.subfunctions
fig, axes = plt.subplots()
streamlines = streamplot(u_h, resolution=1/30, seed=0, axes=axes)
fig.colorbar(streamlines);

# %% [markdown]
# ## Configuring a better preconditioner
#
# For this small problem, we can (and probably should) use a direct factorisation method. But what if the problem is too big? Then we need an iterative method, and an appropriate preconditioner.
#
# Let's try everyone's favourite, ILU(0).

# %%
solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "gmres",
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_max_it": 2000,
    "ksp_converged_reason": None,
    "pc_type": "ilu"
}

# %%
w.zero()
solver = create_solver(solver_parameters)
solver.solve()
convergence(solver)

# %% [markdown]
# This is, unsurprisingly, bad. Fortunately, better options are available.

# %% [markdown]
# ### Block preconditioning
#
# Firedrake hooks up all the necessary machinery to access PETSc's [`PCFIELDSPLIT`](https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/) preconditioner. This provides mechanisms for building preconditioners based on block factorisations. The Stokes problem 
# $$
# \begin{align}
#   \nu\int_\Omega \color{#800020}{\nabla u : \nabla v}\,\mathrm{d}x - \int_\Omega
#   \color{#2A52BE}{p \nabla \cdot v}\,\mathrm{d}x
#   &= \int_\Omega f \cdot v\,\mathrm{d}x, \\
#   -\int_\Omega \color{#2A52BE}{\nabla \cdot u q} \,\mathrm{d}x&= 0
# \end{align}
# $$
# is a block system with matrix
# $$
# \mathcal{A} = \begin{bmatrix} \color{#800020}{A} & \color{#2A52BE}{B^T} \\ \color{#2A52BE}{B} & 0 \end{bmatrix},
# $$
#
# admitting a factorisation
#
# $$
# \begin{bmatrix} I & 0 \\ \color{#2A52BE}{B} \color{#800020}{A}^{-1} & I\end{bmatrix}
# \begin{bmatrix}\color{#800020}{A} & 0 \\ 0 & S\end{bmatrix}
# \begin{bmatrix} I & \color{#800020}{A}^{-1} \color{#2A52BE}{B^T} \\ 0 & I\end{bmatrix},
# $$
#
# with $S = -\color{#2A52BE}{B} \color{#800020}{A}^{-1} \color{#2A52BE}{B^T}$ the *Schur complement*.  This has an inverse
#
# $$
# \begin{bmatrix} I & -\color{#800020}{A}^{-1}\color{#2A52BE}{B^T} \\ 0 & I \end{bmatrix}
# \begin{bmatrix} \color{#800020}{A}^{-1} & 0 \\ 0 & S^{-1}\end{bmatrix}
# \begin{bmatrix} I & 0 \\ -\color{#2A52BE}{B}\color{#800020}{A}^{-1} & I\end{bmatrix}.
# $$
#
# $S$ is never formed, so it's inverse is approximated using an iterative method.

# %%
exact_inverse_parameters = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
    "fieldsplit_1": {
        "ksp_type": "cg",
        "ksp_rtol": 1e-8,
        "pc_type": "none",
    }
}

# %%
w.zero()
solver = create_solver(exact_inverse_parameters)
solver.solve()
convergence(solver)

# %% [markdown]
# This looks good, but we had to use an unpreconditioned Krylov method to invert $S$. To do better we need to provide either an approximation to $S$ or $S^{-1}$.
#
# For the Stokes equations, [Silvester and Wathen (1993)](https://epubs.siam.org/doi/10.1137/0730031) show that $S \approx -\nu^{-1} Q$ is a good approximation, where $Q$ is the pressure mass matrix.
#
# Problem: $Q$ is not available as one of the blocks of $\mathcal{A}$.

# %% [markdown]
# PETSc's approach is to allow us to supply a _separate_ matrix to the solver which will be used to construct the preconditioner. So, we just need to additionally supply
#
# $$
# \mathcal{P} = \mathcal{A} + \begin{bmatrix} 0 & 0 \\ 0 & -\nu^{-1}Q\end{bmatrix} = \begin{bmatrix} \color{#800020}{A} & \color{#2A52BE}{B^T} \\ \color{#2A52BE}{B} & -\nu^{-1} Q \end{bmatrix},
# $$
# where $Q = \int_\Omega p q \,\mathrm{d}x$.
#
# We will construct P by symbolically computing the derivative of the residual to get $\mathcal{A}$ and then subtracting $\nu^{-1} Q$.

# %%
trial = TrialFunction(W)
_, p_t = split(trial)

amat = lhs(derivative(F, w, trial))
pmat = amat - 1/nu * p_t * q*dx

# %% [markdown]
# We can now pass this pmat form to `create_solver` and can configure an appropriate preconditioner.

# %%
pmat_parameters = {
    "mat_type": "nest", # We only need the blocks
    "snes_type": "ksponly",
    "ksp_view": None,
    "ksp_monitor_true_residual": None,
    "ksp_max_it": 100,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
}

# %%
w.zero()
solver = create_solver(pmat_parameters, pmat=pmat)
solver.solve()
convergence(solver)


# %% [markdown]
# ### Providing auxiliary operators
#
# An inconvenience here is that we must build $\mathcal{P}$, even though we only need $-\nu^{-1} Q$ in additional to $\mathcal{A}$ in the preconditioner.
#
# Firedrake offers a facilities to build Python preconditioning objects, utilising petsc4py.
#
# In this case, we can subclass the 
# [`AuxiliaryOperatorPC`](https://www.firedrakeproject.org/firedrake.preconditioners.html#firedrake.preconditioners.assembled.AuxiliaryOperatorPC) to provide the mass matrix.

# %%
class MassMatrix(AuxiliaryOperatorPC):
    _prefix = "mass_"
    def form(self, pc, test, trial):
        # Extract the original form and bcs
        a, bcs = super().form(pc, test, trial)
        # Grab the definition of nu from the user application context (a dict)
        nu = self.get_appctx(pc)["nu"]
        return (-1/nu * test*trial*dx, bcs)


# %% [markdown]
# Now we just need to select parameters such that this Python preconditioner is used.

# %%
mass_parameters = {
    "mat_type": "nest", # We only need the blocks
    "ksp_view": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "__main__.MassMatrix",
        "mass_pc_type": "lu",
    }
}

# %%
appctx = {"nu": nu} # arbitrary user data that is available inside the user PC object
w.zero()
solver = create_solver(mass_parameters, appctx=appctx)
solver.solve()
convergence(solver)

# %% [markdown]
# This performs identically to the previous approach, except that the preconditioning matrix is only built for the pressure space, and constructed "on demand".

# %% [markdown]
# ## Multigrid preconditioners and smoothers
#
# So far, we've only used direct solvers for the blocks. We can also use iterative methods. Here we'll use geometric multigrid to solve
#
# In the same way that Firedrake hooks up solvers such that [`PCFIELDSPLIT`](https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/) is enabled, if a problem was defined on a mesh from a `MeshHierarchy`, [`PCMG`](https://petsc.org/release/manualpages/PC/PCMG/) and [`SNESFAS`](https://petsc.org/release/manualpages/SNESFAS/SNESFAS/) are also available.

# %%
fieldsplit_mg_parameters = {
    "mat_type": "nest",
    "ksp_view": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "mg",
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_max_it": 2,
        }
    },
    "fieldsplit_1": {
        "ksp_type": "chebyshev",
        "ksp_max_it": 2,
        "pc_type": "python",
        "pc_python_type": f"{__name__}.MassMatrix",
        "mass_pc_type": "sor",
    }
}

# %% [markdown]
# Now, when the solver runs, PETSc will call back in to Firedrake for restriction and prolongation, as well as rediscretising $A$ on the coarser levels.

# %%
appctx = {"nu": nu} # arbitrary user data that is available inside the user PC object
w.zero()
solver = create_solver(fieldsplit_mg_parameters, appctx=appctx)
solver.solve()
convergence(solver)

# %% [markdown]
# We can also do monolithic, or "all at once" multigrid. Here we're using Vanka smoothing. This is supported by a new preconditioner in PETSc `PCPATCH`.

# %%
vanka_parameters = {
    "mat_type": "matfree", # We only need the action
    "ksp_type": "fgmres",
    "ksp_max_it": 25,
    "pc_type": "mg",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_convergence_test": "skip",
        "ksp_max_it": 2,
        "pc_type": "python",
        "pc_python_type": "firedrake.PatchPC",
        "patch": {
            "pc_patch_save_operators": 1,
            "pc_patch_partition_of_unity": False,
            "pc_patch_construct_dim": 0,
            # Topological decomposition
            "pc_patch_construct_type": "vanka",
            # Pressure space is constraint space
            "pc_patch_exclude_subspaces": 1,
            # Configure the solver on each patch
            "pc_patch_sub": {
                "mat_type": "dense",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_shift_type": "nonzero",
            }
        }
    },
    "mg_coarse": {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
}

# %% [markdown]
# The solver can be invoked as below, but frequently crashes Jupyter notebooks:

# %%
#w.zero()
#solver = create_solver(vanka_parameters)
#solver.solve()
#convergence(solver)

# %%
