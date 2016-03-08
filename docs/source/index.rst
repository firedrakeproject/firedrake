.. title:: The Firedrake project

.. only:: html

   .. sidebar:: Latest commits to the Firedrake master branch on Github

      .. raw:: html

         <div class="latest-commit" data-github="firedrakeproject/firedrake" data-commits="5"></div>
         <script type="text/javascript" src="_static/jquery.latest-commit.js"></script>

.. only:: latex

   Introduction
   ------------

Firedrake is an automated system for the portable solution of partial
differential equations using the finite element method (FEM). Firedrake
enables users to employ a wide range of discretisations to an infinite
variety of PDEs and employ either conventional CPUs or GPUs to obtain
the solution.

Firedrake employs the Unifed Form Language (UFL) from `the FEniCS
Project <http://fenicsproject.org>`_ while the parallel execution of
FEM assembly is accomplished by the `PyOP2
<http://op2.github.io/PyOP2/>`_ system. The global mesh data
structures, as well as linear and non-linear solvers, are provided by
`PETSc <https://www.mcs.anl.gov/petsc/>`_.


.. only:: html

  .. container:: youtube

    .. youtube:: xhxvM1N8mDQ?modestbranding=1;controls=0;rel=0
       :width: 400px

.. only:: latex

  .. toctree::
 
     documentation
     firedrake
     funding
     team
