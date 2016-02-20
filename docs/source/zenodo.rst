
Firedrake Zenodo integration: tools for reproducible science
============================================================

`Zenodo <http://zenodo.org>`_ provides a facility for archiving
scientific data, such as software. Zenodo provides secure archiving
and referability, including digital object identifiers
(DOIs). Firedrake integrates with Zenodo to provide Firedrake users
with the ability to generate a set of DOIs corresponding to the exact
set of Firedrake components which were used to conduct a particular
simulation.

These DOIs can be used in citations in publications to provide a
reference to the exact version of the software used, and thereby to
improve the reproducibility of your computational science.

How to register DOIs for a version of Firedrake
-----------------------------------------------

This section assumes that you have a Firedrake installation which you
have used to conduct some numerical experiment which you wish to
publish or otherwise record for posterity. It is assumed that your
virtualenv is activated or that you otherwise have the firedrake
scripts in your path.

1. Use ``firedrake-zenodo`` to generate a JSON file containing the
   versions of Firedrake components you are using, as well as a
   documentation message describing what this version was used for
   (this will appear online on Zenodo). For example::

     firedrake-zenodo -m "Version of Firedrake used in 'My paper title'."

   This will create a file ``firedrake.json`` containing the required
   information.

2. Create an issue on the Firedrake GitHub page asking that a Zenodo
   release be created. Attach the ``firedrake.json`` file to the
   issue.