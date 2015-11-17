.. only:: html

  .. sidebar:: Current development information.
  
     Firedrake and PyOP2 are continually tested using the `Travis
     continuous integration system <https://travis-ci.org>`__.

     Latest Firedrake status: |firedrakebuild|

     .. |firedrakebuild| image:: https://travis-ci.org/firedrakeproject/firedrake.png?branch=master
                                 :target: https://travis-ci.org/firedrakeproject/firedrake

     Latest PyOP2 status: |pyop2build|

     .. |pyop2build| image:: https://travis-ci.org/OP2/PyOP2.png?branch=master
                             :target: https://travis-ci.org/OP2/PyOP2

     Firedrake and PyOP2 are developed on `GitHub <http://github.com>`__ while we also maintain Firedrake-ready versions of the `FEniCS <http://fenicsproject.org>`__ tools on `Bitbucket <http://bitbucket.org>`__. 

     * `Firedrake on GitHub <https://github.com/firedrakeproject/firedrake/>`__
     * `PyOP2 on GitHub <https://github.com/OP2/PyOP2>`__
     * `Firedrake version of FFC on Bitbucket <https://bitbucket.org/mapdes/ffc>`__
     * `Firedrake version of UFL on Bitbucket <https://bitbucket.org/mapdes/ufl>`__
     * `Firedrake version of FIAT on Bitbucket <https://bitbucket.org/mapdes/fiat>`__

  Getting started
  ===============

  The first step is to download and install Firedrake and its
  dependencies. For full instructions, see :doc:`obtaining Firedrake
  <download>`.

  .. _firedrake_tutorials:

Tutorials
=========

Once you've built Firedrake, you'll want to actually solve some
PDEs. Below are a few tutorial examples to get you started.

.. toctree::

   A basic Helmholtz equation.<demos/helmholtz.py>
   The Burgers equation, a non-linear, unsteady example.<demos/burgers.py>
   A mixed formulation of the Poisson equation.<demos/poisson_mixed.py>
   A steady-state advection equation using upwinding, on an extruded mesh.<demos/upwind_advection.py>
   Benney-Luke nonlinear wave equation.<demos/benney_luke.py>
   A linear wave equation using explicit timestepping.<demos/linear_wave_equation.py>

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
   extruded-meshes
   mesh-coordinates
   point-evaluation
   checkpointing

.. only:: html

  API documentation
  =================

  The complete list of all the classes and methods in Firedrake is
  available at the :doc:`firedrake` page. The same information is
  :ref:`indexed <genindex>` in alphabetical order. Another very
  effective mechanism is the site :ref:`search engine <search>`.
