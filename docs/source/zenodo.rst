Firedrake Zenodo integration: tools for reproducible science
============================================================

.. image:: _static/zenodo-gradient-1000.png
   :width: 25%
   :alt: Zenodo
   :align: right
   :target: https://zenodo.org/
   :class: round-corners

.. image:: _static/GitHub_Logo.png
   :width: 25%
   :alt: Zenodo
   :align: right
   :target: https://github.com/
   :class: round-corners


`Zenodo <https://zenodo.org/>`_ provides a facility for archiving
scientific data, such as software. Zenodo provides secure archiving
and referability, including digital object identifiers
(DOIs). Firedrake integrates with Zenodo and GitHub to provide Firedrake users
with the ability to generate a set of DOIs corresponding to the exact
set of Firedrake components which were used to conduct a particular
simulation.

These DOIs can be used in citations in publications to provide a
reference to the exact version of the software used, and thereby to
improve the reproducibility of your computational science.

How to register DOIs for a version of Firedrake
-----------------------------------------------

This section assumes that you have a Firedrake installation which you
have used to conduct some numerical experiment and which you wish to
publish or otherwise record for posterity. It is assumed that your
virtualenv is activated or that you otherwise have the firedrake
scripts in your path.

1. Use ``firedrake-zenodo`` to generate a JSON file containing the
   versions of Firedrake components you are using, as well as a title
   describing what this version was used for (this will appear online
   on Zenodo). For example::

     firedrake-zenodo -t "My paper title"

   You can additionally provide a single file that contains any extra
   (free-form) information that you want to appear in the uploaded
   Zenodo record::

     firedrake-zenodo -t "My paper title" --info-file README.txt

   This file could, for example, contain DOIs of any archived
   simulation code that you used over and above the core Firedrake
   components.  It can also be a single python script or a tarball (or
   other archive) of your code and any data required to reproduce your
   results.

   This will create a file ``firedrake.json`` containing the required
   information.

2. Create an issue on the Firedrake GitHub page asking that a Zenodo
   release be created. Attach the ``firedrake.json`` file to the
   issue. You can create the issue using the correct template `here
   <https://github.com/firedrakeproject/firedrake/issues/new?template=zenodo_release.md>`__.

3. The Firedrake developers will generate a bespoke Firedrake release
   containing exactly the set of versions your JSON file specifies, as
   well as creating a Zenodo record collating these. You will be
   provided with a firedrake release tag of the form
   ``Firedrake_YYYYMMDD.N``.

   You can see an example such a collated record `here
   <https://zenodo.org/record/1402622>`__.

4. You can use this release tag to generate a BibTeX entry (including
   the DOI) for the collated "meta"-record, which in turn links to all
   the individual components::

     firedrake-zenodo --bibtex Firedrake_YYYYMMDD.N

   Obviously, you substitute in your Firedrake release tag.

You can explore the full set of options for ``firedrake-zenodo``
with::

  firedrake-zenodo -h

What else do you need to do?
----------------------------

Archive your code
~~~~~~~~~~~~~~~~~

``firedrake-zenodo`` produces citable DOIs which point to the versions
of Firedrake components you used. This covers your bases as far as
concerns Firedrake, but doesn't cover your code which uses
Firedrake. Best practice in computational science also demands that
you provide the code which you used to conduct your experiments. You
could attach a tarball as a supplement to your paper, embed a tarball
(or single script) in the Zenodo release generated to record your
Firedrake components, or you could also use Zenodo to generate a DOI
directly from your GitHub source repository. Using Zenodo in
combination with GitHub for this purpose is documented `by github here
<https://guides.github.com/activities/citable-code/>`_.

.. note::

   If you archive your code before running ``firedrake-zenodo``, you
   can ensure that the eventual release also references these DOIs by
   providing them in a text file via the ``--info-file`` argument.
   You can also directly attach your code (either a single script or a
   single archive containing it) to the Firedrake Zenodo release using
   the same argument.

Cite your sources
~~~~~~~~~~~~~~~~~

Citing custom DOIs for particular versions of Firedrake and its
dependencies aids readers of your papers in reproducing your
science. However it's a supplement to, and not a replacement for,
citing the published resources for the computational methods you are
employing. Firedrake also offers support for citing the papers on
which your computations depend. This is documented on the
:doc:`citing` page.
