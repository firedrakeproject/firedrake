
Firedrake and PyOP2 publications
================================

Citing Firedrake
----------------

If you publish results using Firedrake, we would be grateful if you
would cite the relevant papers.  The simplest way to determine what
these are is by asking Firedrake itself.  You can ask that a list of
citations relevant to your computation be printed when exiting by calling
:meth:`.Citations.print_at_exit` after importing Firedrake::

  from firedrake import *

  Citations.print_at_exit()

Alternatively, you can select that this should occur by passing the
command-line option ``-citations``.  In both cases, you will also
obtain the correct `citations for PETSc
<http://www.mcs.anl.gov/petsc/documentation/referencing.html>`_.

If you cannot use this approach, there are a number of papers.  Those
which are relevant depend a little on which functionality you used.

For Firedrake itself, please cite :cite:`Rathgeber2016`.  If you use
the :doc:`extruded mesh </extruded-meshes>` functionality please cite
:cite:`McRae2016` and :cite:`Bercea2016`. When using quadrilateral meshes,
please cite :cite:`Homolya2016` and :cite:`McRae2016`.

If your work relies on the kernel-level performance optimisations that
Firedrake performs using `COFFEE
<http://github.com/coneoproject/COFFEE>`_, please cite the
COFFEE papers :cite:`Luporini2015` and :cite:`Luporini2016`.

.. bibliography:: _static/bibliography.bib

Citing other packages
~~~~~~~~~~~~~~~~~~~~~

Firedrake relies heavily on PETSc, which you should cite
`appropriately
<http://www.mcs.anl.gov/petsc/documentation/referencing.html>`_.
Additionally, if you talk about UFL in your work, please cite the `UFL
paper <http://fenicsproject.org/citing/#ufl>`_.

Making your simulations reproducible with Zenodo integration
------------------------------------------------------------

In addition to citing the work you use, you will want to provide
references to the exact versions of Firedrake and its dependencies
which you used. Firedrake supports this through :doc:`Zenodo integration <zenodo>`.


Journal papers and conference proceedings about Firedrake
---------------------------------------------------------

.. raw:: html

   <script src="http://www.bibbase.org/show?bib=http%3A%2F%2Fwww.firedrakeproject.org%2F_static%2Fbibliography.bib&jsonp=1"></script>

Journal papers and conference proceedings using Firedrake
---------------------------------------------------------

.. rst-class:: emphasis

   If you have published work using Firedrake that does not appear
   below, we would :doc:`love to hear about it </contact>`.

.. raw:: html

   <script src="http://www.bibbase.org/show?bib=http%3A%2F%2Fwww.firedrakeproject.org%2F_static%2Ffiredrake-apps.bib&jsonp=1"></script>
