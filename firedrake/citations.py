from __future__ import absolute_import, print_function, division
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


Citations().add("Rathgeber2016", """
@Article{Rathgeber2016,
  author =       {Florian Rathgeber and David A. Ham and Lawrence
                  Mitchell and Michael Lange and Fabio Luporini and
                  Andrew T. T. McRae and Gheorghe-Teodor Bercea and
                  Graham R. Markall and Paul H. J. Kelly},
  title =        {Firedrake: automating the finite element method by
                  composing abstractions},
  journal =      {ACM Trans. Math. Softw.},
  year =         2016,
  volume =       {43},
  number =       {3},
  year =         {2016},
  issn =         {0098-3500},
  pages =        {24:1--24:27},
  doi =          {10.1145/2998441},
  archiveprefix ={arXiv},
  eprint =       {1501.01809},
  url =          {http://arxiv.org/abs/1501.01809}
}
""")

# Register the firedrake paper for citations
Citations().register("Rathgeber2016")

# The rest are all registered only when using appropriate functionality.
Citations().add("McRae2016", """
@Article{McRae2016,
  author =       {Andrew T. T. McRae and Gheorghe-Teodor Bercea and
                  Lawrence Mitchell and David A. Ham and Colin
                  J. Cotter},
  title =        {Automated generation and symbolic manipulation of
                  tensor product finite elements},
  journal =      {SIAM Journal on Scientific Computing},
  year =         2016,
  volume =       38,
  number =       5,
  pages =        {S25--S47},
  doi =          {10.1137/15M1021167},
  archiveprefix ={arXiv},
  eprint =       {1411.2940},
  primaryclass = {math.NA},
  url =          {http://arxiv.org/abs/1411.2940}
}
""")

Citations().add("Homolya2016", """
@Article{Homolya2016,
  author =       {Mikl\'os Homolya and David A. Ham},
  title =        {A parallel edge orientation algorithm for
                  quadrilateral meshes},
  journal =      {SIAM Journal on Scientific Computing},
  year =         2016,
  volume =       38,
  number =       5,
  pages =        {S48--S61},
  doi =          {10.1137/15M1021325},
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

Citations().add("Luporini2016", """
@article{Luporini2016,
  author =       {Fabio Luporini and David A. Ham and Paul H. J. Kelly},
  title =        {An algorithm for the optimization of finite element
                  integration loops},
  journal =      {ACM Transactions on Mathematical Software},
  year =         2017,
  volume =       44,
  issue =        1,
  pages =        {3:1--3:26},
  archiveprefix ={arXiv},
  eprint =       {1604.05872},
  url =          {http://arxiv.org/abs/1604.05872},
  doi =          {10.1145/3054944},
}
""")

Citations().add("Bercea2016", """
@Article{Bercea2016,
  author =       {Gheorghe{-}Teodor Bercea and Andrew T. T. McRae and
                  David A. Ham and Lawrence Mitchell and Florian
                  Rathgeber and Luigi Nardi and Fabio Luporini and
                  Paul H. J. Kelly},
  title =        {A structure-exploiting numbering algorithm for
                  finite elements on extruded meshes, and its
                  performance evaluation in Firedrake},
  journal =      {Geoscientific Model Development},
  year =         2016,
  volume =       9,
  number =       10,
  pages =        {3803--3815},
  doi =          {10.5194/gmd-9-3803-2016},
  archiveprefix ={arXiv},
  eprint =       {1604.05937},
  primaryclass = {cs.MS},
  url =          {http://arxiv.org/abs/1604.05937}
}
""")

Citations().add("Mitchell2016", """
@Article{Mitchell2016,
  author =       {Lawrence Mitchell and Eike Hermann M\"uller},
  title =        {High level implementation of geometric multigrid
                  solvers for finite element problems: applications in
                  atmospheric modelling},
  journal =      {Journal of Computational Physics},
  year =         2016,
  volume =       327,
  pages =        {1--18},
  doi =          {10.1016/j.jcp.2016.09.037},
  archiveprefix ={arXiv},
  eprint =       {1605.00492},
  primaryclass = {cs.MS},
  url =          {http://arxiv.org/abs/1605.00492}
}
""")

Citations().add("Homolya2017", """
@Misc{Homolya2017,
  author =        {Mikl\'os Homolya and Lawrence Mitchell and Fabio Luporini and
                   David A. Ham},
  title =         {{TSFC: a structure-preserving form compiler}},
  year =          2017,
  archiveprefix = {arXiv},
  eprint =        {1705.03667},
  primaryclass =  {cs.MS},
  url =           {http://arxiv.org/abs/1705.003667}
}
""")

Citations().add("Mitchell2017", """
@Misc{Mitchell2017,
  author =       {Lawrence Mitchell and Robert C. Kirby},
  title =        {{Solver composition across the PDE/linear algebra
                  barrier}},
  year =         2017,
  archiveprefix ={arXiv},
  eprint =       {1706.01346},
  primaryclass = {cs.MS},
  url =          {http://arxiv.org/abs/1706.01346}
}
""")
