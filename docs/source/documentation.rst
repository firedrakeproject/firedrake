.. only:: html

  .. sidebar:: Current development information.
  
     Firedrake and PyOP2 are continually tested using `Jenkins <https://jenkins.io/>`__  and `Travis <https://travis-ci.org>`__ respectively.

     Latest Firedrake status: |firedrakebuild|

     .. |firedrakebuild| image:: https://jenkins.ese.ic.ac.uk:1080/job/firedrakeproject/job/firedrake/job/master/badge/icon
                                 :target: https://jenkins.ese.ic.ac.uk:1080/blue/organizations/jenkins/firedrakeproject%2Ffiredrake/branches/

     Latest PyOP2 status: |pyop2build|

     .. |pyop2build| image:: https://travis-ci.org/OP2/PyOP2.png?branch=master
                             :target: https://travis-ci.org/OP2/PyOP2

     Firedrake and PyOP2 are developed on `GitHub
     <http://github.com>`__ where we also maintain Firedrake-ready
     versions of the `FEniCS <http://fenicsproject.org>`__ components
     UFL and FIAT.

     * `Firedrake on GitHub <https://github.com/firedrakeproject/firedrake/>`__
     * `TSFC on GitHub <https://github.com/firedrakeproject/tsfc/>`__
     * `PyOP2 on GitHub <https://github.com/OP2/PyOP2>`__
     * `Firedrake version of UFL on GitHub <https://github.com/firedrakeproject/ufl>`__
     * `Firedrake version of FIAT on GitHub <https://github.com/firedrakeproject/fiat>`__

  Getting started
  ===============

  The first step is to download and install Firedrake and its
  dependencies. For full instructions, see :doc:`obtaining Firedrake
  <download>`.

  .. _firedrake_tutorials:

Introductory Tutorials
======================

Once you've built Firedrake, you'll want to actually solve some
PDEs. Below are a few tutorial examples to get you started.

.. toctree::
   :maxdepth: 1

   A basic Helmholtz equation.<demos/helmholtz.py>
   The Burgers equation, a non-linear, unsteady example.<demos/burgers.py>
   A mixed formulation of the Poisson equation.<demos/poisson_mixed.py>
   A time-dependent DG advection equation using upwinding.<demos/DG_advection.py>
   An extruded mesh example, using a steady-state continuity equation.<demos/extruded_continuity.py>
   A linear wave equation with optional mass lumping.<demos/linear_wave_equation.py>

Jupyter notebooks
-----------------

In addition to the documented tutorials, we also have some `Jupyter
notebooks <https://jupyter.org/>`__ that are a more interactive way of
getting to know Firedrake.  They are described in more detail :doc:`on
their own page <notebooks>`.

.. only:: html

  API documentation
  =================

  The complete list of all the classes and methods in Firedrake is
  available at the :doc:`firedrake` page. The same information is
  :ref:`indexed <genindex>` in alphabetical order. Another very
  effective mechanism is the site :ref:`search engine <search>`.

Manual
======

Once you have worked through the tutorials, the user manual is the
next step.  It goes in to more detail on how to set up and solve
finite element problems in Firedrake.

.. toctree::
   :maxdepth: 2

   variational-problems
   solving-interface
   boundary_conditions
   r-space
   extruded-meshes
   mesh-coordinates
   interpolation
   point-evaluation
   visualisation
   checkpointing
   matrix-free
   petsc-interface
   parallelism

Advanced tutorials
==================

These tutorials demonstrate some more advanced features of Firedrake's
PDE solving capabilities, such as block-preconditioning mixed finite
element systems.

.. toctree::
   :maxdepth: 1

   Printing in parallel.<demos/parprint.py>
   Benney-Luke nonlinear wave equation.<demos/benney_luke.py>
   Solving the one-layer Quasi-Geostrophic equations.<demos/qg_1layer_wave.py>
   Computing eigenmodes of the Quasi-Geostrophic equations using SLEPc.<demos/qgbasinmodes.py>
   A Quasi-Geostrophic wind driven gyre.<demos/qg_winddrivengyre.py>
   Preconditioning saddle-point systems, using the mixed Poisson problem as an example.<demos/saddle_point_systems.py>
   The Camassa-Holm equation, a nonlinear integrable PDE.<demos/camassaholm.py>
   The Monge-Ampère equation, a nonlinear PDE, demonstrating fieldsplit preconditioning.<demos/ma-demo.py>
   Preconditioning using geometric multigrid.<demos/geometric_multigrid.py>
   Linear mixed fluid-structure interaction system.<demos/linear_fluid_structure_interaction.py>
