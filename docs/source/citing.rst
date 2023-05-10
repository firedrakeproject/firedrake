.. raw:: latex

   \clearpage

==================
 Citing Firedrake
==================

If you publish results using Firedrake, we would be grateful if you
would cite the relevant papers.

The simplest way to determine what these are is by asking Firedrake
itself.  You can ask that a list of citations relevant to your
computation be printed when exiting by calling
:meth:`.Citations.print_at_exit` after importing Firedrake::

  from firedrake import *

  Citations.print_at_exit()

Alternatively, you can select that this should occur by passing the
command-line option ``-citations``.  In both cases, you will also
obtain the correct `citations for PETSc
<https://petsc.org/release/#citing-petsc>`_.

If you cannot use this approach, there are a number of papers.  Those
which are relevant depend a little on which functionality you used.

For Firedrake itself, please cite :cite:`Rathgeber2016`.  If you use
the :doc:`extruded mesh </extruded-meshes>` functionality please cite
:cite:`McRae2016` and :cite:`Bercea2016`. When using quadrilateral meshes,
please cite :cite:`Homolya2016` and :cite:`McRae2016`. If you use
:py:func:`~.VertexOnlyMesh`, please cite :cite:`nixonhill2023consistent`.

The form compiler, TSFC, is documented in :cite:`Homolya2018` and
:cite:`Homolya2017a`.

If you make use of matrix-free functionality and custom block
preconditioning, please cite :cite:`Kirby2017`.

If you would like to help us to keep track of research directly
benefitting from Firedrake, please feel free to add your paper in
bibtex format in the `bibliography for firedrake applications
<https://github.com/firedrakeproject/firedrake/blob/master/docs/source/_static/firedrake-apps.bib>`_.

Citing other packages
~~~~~~~~~~~~~~~~~~~~~

Firedrake relies heavily on PETSc, which you should cite
`appropriately
<https://petsc.org/release/#citing-petsc>`_.
Additionally, if you talk about UFL in your work, please cite the `UFL
paper <http://fenicsproject.org/citing/>`_.

Making your simulations reproducible with Zenodo integration
------------------------------------------------------------

In addition to citing the work you use, you will want to provide
references to the exact versions of Firedrake and its dependencies
which you used. Firedrake supports this through :doc:`Zenodo
integration <zenodo>`.

.. bibliography:: _static/bibliography.bib
