:orphan: true

Introductory Jupyter notebooks
==============================

These notebooks provide an introduction to usage of Firedrake, and are
designed to familiarise you with manipulating Firedrake objects to
solve finite element problems.

Running the notebooks locally
-----------------------------

To run the notebooks, you will need to `install jupyter
<https://jupyter.org/install.html>`__ *inside* your activated
Firedrake virtualenv.

These notebooks are maintained in the Firedrake repository, so all the
material is available in your Firedrake installation source
directory.  If you installed in ``Documents/firedrake``, then the
notebooks are in the directory
``Documents/firedrake/src/firedrake/docs/notebooks``.

Running the notebooks on Google Colab
-------------------------------------

Thanks to the excellent `FEM on
Colab <https://fem-on-colab.github.io/index.html>`__ by `Francesco
Ballarin <https://www.francescoballarin.it>`__, you can run the notebooks on
Google Colab through your web browser, without installing Firedrake.

We also provide links to non-interactive renderings of the notebooks using
`Jupyter nbviewer <https://nbviewer.jupyter.org>`__.


A first example
===============
In this notebook, we solve the symmetric positive definite "Helmholtz"
equation, and learn about meshes and function spaces.  A rendered
version of this notebook is available `here
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/01-spd-helmholtz.ipynb>`__
and there is a version `on Colab <https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/01-spd-helmholtz.ipynb>`__


Incorporating strong boundary conditions
========================================

Next, we modify the problem slightly and solve the Poisson equation.
We introduce strong (Dirichlet) boundary conditions and `how to use
them
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/02-poisson.ipynb>`__.
You can run this notebook yourself `on Colab <https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/02-poisson.ipynb>`__


A vector-valued problem
=======================

Moving on from scalar problems, we look at our first vector-valued problem,
namely the equations of linear elasticity.  In this notebook, we learn about
some of UFL's support for tensor algebra, and start looking at `configuring
linear solvers
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/03-elasticity.ipynb>`__.
You can run this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/03-elasticity.ipynb>`__


A time-dependent, nonlinear, problem
====================================

This notebook looks at a simple nonlinear problem, the viscous
Burgers' equation, and also treats simple timestepping schemes.  We
learn about formulating nonlinear, as opposed to linear problems, and
also a little bit about how to write `efficient Firedrake code
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/04-burgers.ipynb>`__.
You can run this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/04-burgers.ipynb>`__


A mixed formulation of the Poisson equation
===========================================

`In this notebook
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/05-mixed-poisson.ipynb>`__.,
we look at our first mixed finite element problem.  A dual formulation of the
Poisson equation.  This equation also appears in the context of flow in porous
media, as Darcy flow.  We introduce mixed function spaces and how to work with
them.  Equations with multiple variables are typically more challenging to
precondition, and so we discuss some of the preconditioning strategies for such
block systems, and how to control them using PETSc solver options. You can run
this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/05-mixed-poisson.ipynb>`__


PDE-constrained optimisation with `dolfin-adjoint <http://www.dolfin-adjoint.org/>`__
=====================================================================================

Now that we've learnt how to solve some PDEs, we might want to consider
optimisation subject to PDE constraints.  `This notebook
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/06-pde-constrained-optimisation.ipynb>`__
introduces the use of `dolfin-adjoint <http://www.dolfin-adjoint.org/>`__ to
solve PDE constrained optimisation problems.  We solve the Stokes equations and
minimise energy loss due to heat, controlling inflow/outflow in a pipe. You can
run this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/06-pde-constrained-optimisation.ipynb>`__


Geometric multigrid
===================

The next notebook looks a little bit at the support Firedrake has for
geometric multigrid, and how you can configure complex multilevel
solvers purely using `PETSc options
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/07-geometric-multigrid.ipynb>`__.
You can run this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/07-geometric-multigrid.ipynb>`__


Solver Composition
==================

We next dive a little deeper into the advanced ways in which Firedrake
and PETSc enable solvers and preconditioners to be composed in
arbitrarily complex ways to `create an optimal solution strategy for a
particular problem
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/08-composable-solvers.ipynb>`__.
You can run this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/08-composable-solvers.ipynb>`__


Hybridisation
=============

Building on the theme of composable solvers, we now explore
`Firedrake's capabilities in the area of static condensation and
hybridisation
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/09-hybridisation.ipynb>`__.
You can run this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/09-hybridisation.ipynb>`__


Sum Factorisation
=================

In this notebook, we take a look under the hood at the sorts of performance
optimisation that Firedrake's compilers can generate. In this case, we focus on
`sum factorisation for tensor product elements
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/10-sum-factorisation.ipynb>`__.
You can run this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/10-sum-factorisation.ipynb>`__


Solving adjoint problems
========================

In some cases, it can be useful to extract adjoint solution data which is
written to tape by `dolfin-adjoint`. This notebook shows how to do that:
`solving adjoint equations using firedrake-adjoint
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/11-extract-adjoint-solutions.ipynb>`__.
You can run this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/11-extract-adjoint-solutions.ipynb>`__


Running on HPC
==============

When it comes to running Firedrake on a high performance computer
there are a range of different techniques to get the best performance
from your code. The `HPC demonstration
<https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/12-HPC_demo.ipynb>`__
notebook builds up a multigrid solver for an elliptic problem
specifically designed for solving very large problems using Firedrake on
HPC. You can run this notebook yourself `on Colab
<https://colab.research.google.com/github/firedrakeproject/notebooks/blob/main/12-HPC_demo.ipynb>`__

