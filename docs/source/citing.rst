.. raw:: latex

   \clearpage

==================
 Citing Firedrake
==================

If you publish results using Firedrake, we would be grateful if you
would cite the `Firedrake user manual
<https://doi.org/10.25561/104839>`_::

  @manual{FiredrakeUserManual,
    title        = {Firedrake User Manual},
    author       = {David A. Ham and Paul H. J. Kelly and Lawrence Mitchell and Colin J. Cotter and Robert C. Kirby and Koki Sagiyama and Nacime Bouziani and Sophia Vorderwuelbecke and Thomas J. Gregory and Jack Betteridge and Daniel R. Shapero and Reuben W. Nixon-Hill and Connor J. Ward and Patrick E. Farrell and Pablo D. Brubeck and India Marsden and Thomas H. Gibson and Miklós Homolya and Tianjiao Sun and Andrew T. T. McRae and Fabio Luporini and Alastair Gregory and Michael Lange and Simon W. Funke and Florian Rathgeber and Gheorghe-Teodor Bercea and Graham R. Markall},
    organization = {Imperial College London and University of Oxford and Baylor University and University of Washington},
    edition      = {First edition},
    year         = {2023},
    month        = {5},
    doi          = {10.25561/104839},
  }

The simplest way to determine any additional relevant papers to cite is
by asking Firedrake itself.  You can ask that a list of citations
relevant to your computation be printed when exiting by calling
:meth:`.Citations.print_at_exit` after importing Firedrake::

  from firedrake import *

  Citations.print_at_exit()

Alternatively, you can select that this should occur by passing the
command-line option ``-citations``.  In both cases, you will also
obtain the correct `citations for PETSc
<https://petsc.org/release/#citing-petsc>`_.

If you cannot use this approach, there are a number of papers.  Those
which are relevant depend a little on which functionality you used.

For Firedrake itself, please cite :cite:`FiredrakeUserManual`.  If you use
the :doc:`extruded mesh </extruded-meshes>` functionality please cite
:cite:`McRae2016` and :cite:`Bercea2016`. When using quadrilateral meshes,
please cite :cite:`Homolya2016` and :cite:`McRae2016`. If you use
:py:func:`~.VertexOnlyMesh`, please cite :cite:`nixonhill2023consistent`.

If you use the interfaces to couple Firedrake and machine learning frameworks such as PyTorch or JAX,
please cite :cite:`Bouziani2024`. If you use the :py:class:`~.AbstractExternalOperator`
interface, please cite :cite:`Bouziani2024` and :cite:`Bouziani2021`.

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
