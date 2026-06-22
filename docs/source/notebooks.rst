:orphan: true

Introductory Jupyter notebooks
==============================

These notebooks provide an introduction to usage of Firedrake, and are
designed to familiarise you with manipulating Firedrake objects to
solve finite element problems.  The rendered notebooks below are executed
as part of the documentation build; each page also links to a version you
can run yourself on Google Colab.

Running the notebooks locally
-----------------------------

To run the notebooks, you will need to `install jupyter
<https://jupyter.org/install.html>`__ *inside* your activated
Firedrake virtualenv.

These notebooks are maintained in the Firedrake repository as
`jupytext <https://jupytext.readthedocs.io>`__ ``py:percent`` scripts, so all
the material is available in your Firedrake installation source directory.  If
you cloned Firedrake in ``Documents/firedrake``, then the notebooks are in the
directory ``Documents/firedrake/docs/notebooks``.  Jupyter (with the jupytext
extension installed) can open the ``.py`` files directly as notebooks, or you
can convert one to ``.ipynb`` with, for example::

    jupytext --to ipynb 01-spd-helmholtz.py

Running the notebooks on Google Colab
-------------------------------------

Thanks to the excellent `FEM on
Colab <https://fem-on-colab.github.io/index.html>`__ by `Francesco
Ballarin <https://www.francescoballarin.it>`__, you can run the notebooks on
Google Colab through your web browser, without installing Firedrake.  Each
rendered notebook below carries a link to its Colab version.

The notebooks
-------------

.. toctree::
   :maxdepth: 1

   notebooks/01-spd-helmholtz
   notebooks/02-poisson
   notebooks/03-elasticity
   notebooks/04-burgers
   notebooks/05-mixed-poisson
   notebooks/06-pde-constrained-optimisation
   notebooks/07-geometric-multigrid
   notebooks/08-composable-solvers
   notebooks/09-hybridisation
   notebooks/10-sum-factorisation
   notebooks/11-extract-adjoint-solutions
   notebooks/12-HPC_demo
