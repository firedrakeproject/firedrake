.. title:: The Firedrake project

.. only:: html

   .. sidebar:: Latest commits to the Firedrake master branch on Github

      .. raw:: html

         <div class="latest-commit" data-github="firedrakeproject/firedrake" data-commits="5"></div>
         <script type="text/javascript" src="_static/jquery.latest-commit.js"></script>

.. only:: latex

   Introduction
   ------------
 
Firedrake is an automated system for the solution of partial
differential equations using the finite element method
(FEM). Firedrake uses sophisticated code generation to provide
mathematicians, scientists, and engineers with a very high productivity
way to create sophisticated high performance simulations.

Features:
---------

* Expressive specification of any PDE using the Unified Form Language
  from `the FEniCS Project <http://fenicsproject.org>`_.
* Sophisticated, programmable solvers through seamless coupling with `PETSc
  <http://www.mcs.anl.gov/petsc/>`_.
* Triangular, quadrilateral, and tetrahedral unstructured meshes.
* Layered meshes of triangular wedges or hexahedra.
* Vast range of finite element spaces.
* Sophisticated automatic optimisation, including sum factorisation
  for high order elements, and vectorisation.
* Geometric multigrid.
* Customisable operator preconditioners.
* Support for static condensation, hybridisation, and HDG methods.


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
