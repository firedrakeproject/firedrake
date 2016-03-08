from __future__ import absolute_import
from firedrake.petsc import PETSc


__all__ = ["Citations"]


class Citations(dict):

    """Entry point to citations management.

    This singleton object may be used to record Bibtex citation
    information and then register that a particular citation is
    relevant for a particular computation.  It hooks up with PETSc's
    citation registration mechanism, so that running with
    ``-citations`` does the right thing.

    Example usage::

        Citations().add("key", "bibtex-entry-for-my-funky-method")

        ...

        if using_funky_method:
            Citations().register("key")
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Citations, cls).__new__(cls)

        return cls._instance

    def add(self, key, entry):
        """Add a paper to the database of possible citations.

        :arg key: The key to use.
        :arg entry: The bibtex entry.
        """
        self[key] = entry

    def register(self, key):
        """Register a paper to be cited so that PETSc knows about it.

        :arg key: The key of the relevant citation.

        :raises: :exc:`~.exceptions.KeyError` if no such citation is
            found in the database.

        Papers to be cited can be added using :meth:`add`.

        .. note::

           The intended use is that :meth:`register` should be called
           only when the referenced functionality is actually being
           used.
        """
        cite = self.get(key, None)
        if cite is None:
            raise KeyError("Did not find a citation for '%s', did you forget to add it?" % key)
        PETSc.Sys.registerCitation(cite)

    @classmethod
    def print_at_exit(cls):
        """Print citations when exiting."""
        # We devolve to PETSc for actually printing citations.
        PETSc.Options()["citations"] = None


Citations().add("Rathgeber2015", """
@Article{Rathgeber2015,
  author =       {Florian Rathgeber and David A. Ham and Lawrence
                  Mitchell and Michael Lange and Fabio Luporini and
                  Andrew T. T. McRae and Gheorghe-Teodor Bercea and
                  Graham R. Markall and Paul H. J. Kelly},
  title =        {Firedrake: automating the finite element method by
                  composing abstractions},
  journal =      {Submitted to ACM TOMS},
  year =         2015,
  archiveprefix ={arXiv},
  eprint =       {1501.01809},
  url =          {http://arxiv.org/abs/1501.01809}
}
""")

# Register the firedrake paper for citations
Citations().register("Rathgeber2015")

# The rest are all registered only when using appropriate functionality.
Citations().add("McRae2014", """
@Article{McRae2014,
  author =       {Andrew T. T. McRae and Gheorghe-Teodor Bercea and
                  Lawrence Mitchell and David A. Ham and Colin
                  J. Cotter},
  title =        {Automated generation and symbolic manipulation of
                  tensor product finite elements},
  journal =      {Submitted to SIAM Journal on Scientific Computing},
  year =         2014,
  archiveprefix ={arXiv},
  eprint =       {1411.2940},
  url =          {http://arxiv.org/abs/1411.2940}
}
""")

Citations().add("Homolya2016", """
@Article{Homolya2016,
  author =       {Mikl\'os Homolya and David A. Ham},
  title =        {A parallel edge orientation algorithm for
                  quadrilateral meshes},
  journal =      {To appear in SIAM Journal on Scientific Computing},
  year =         2016,
  archiveprefix ={arXiv},
  eprint =       {1505.03357},
  url =          {http://arxiv.org/abs/1505.03357}
}
""")

Citations().add("Luporini2015", """
@Article{Luporini2015,
  author =       {Fabio Luporini and Ana Lucia Varbanescu and Florian
                  Rathgeber and Gheorghe-Teodor Bercea and
                  J. Ramanujam and David A. Ham and Paul H. J. Kelly},
  title =        {Cross-Loop Optimization of Arithmetic Intensity for
                  Finite Element Local Assembly},
  journal =      {ACM Transactions on Architecture and Code
                  Optimization},
  year =         2015,
  volume =       11,
  number =       4,
  pages =        {57:1--57:25},
  url =          {http://doi.acm.org/10.1145/2687415},
  doi =          {10.1145/2687415},
}
""")
