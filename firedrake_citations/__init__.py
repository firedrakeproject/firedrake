import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc


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

        :raises: :exc:`KeyError` if no such citation is
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
  primaryclass = {cs.MS},
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
  primaryclass = {cs.MS},
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

Citations().add("Kirby2017", """
@Article{Kirby2017,
  author =       {Robert C. Kirby and Lawrence Mitchell},
  title =        {{Solver composition across the PDE/linear algebra
                  barrier}},
  journal =      {SIAM Journal on Scientific Computing},
  year =         2018,
  volume =       40,
  number =       1,
  pages =        {C76-C98},
  doi =          {10.1137/17M1133208},
  archiveprefix ={arXiv},
  eprint =       {1706.01346},
  primaryclass = {cs.MS},
  url =          {http://arxiv.org/abs/1706.01346}
}
""")

Citations().add("Homolya2017a", """
@Misc{Homolya2017a,
  author =       {Mikl\'os Homolya and Robert C. Kirby and David
                  A. Ham},
  title =        {{Exposing and exploiting structure: optimal code
                  generation for high-order finite element methods}},
  year =         2017,
  archiveprefix ={arXiv},
  eprint =       {1711.02473},
  primaryclass = {cs.MS},
  url =          {http://arxiv.org/abs/1711.02473}
}
""")

Citations().add("Gibson2018", """
@Misc{Gibson2018,
  author =       {Thomas H. Gibson and Lawrence Mitchell and David
                  A. Ham and Colin J. Cotter},
  title =        {{A domain-specific language for the hybridization
                  and static condensation of finite element methods}},
  year =         2018,
  archiveprefix ={arXiv},
  eprint =       {1802.00303},
  primaryclass = {cs.MS},
  url =          {https://arxiv.org/abs/1802.00303}
}
""")

Citations().add("Kolev2009", """
@Misc{Kolev2009,
  author =       {Kolev, Tzanio V and Vassilevski, Panayot S},
  title =        {{Parallel auxiliary space AMG for H (curl) problems}},
  journal =      {Journal of Computational Mathematics},
  year =         2009,
  volume =       27,
  number =       5,
  pages =        {604--623},
  url =          {https://www.jstor.org/stable/43693530}
}
""")

Citations().add("Hiptmair1998", """
@Misc{Hiptmair1998,
  author =       {Hiptmair, Ralf},
  title =        {{Multigrid Method for Maxwell's Equations}},
  journal =      {SIAM Journal on Numerical Analysis},
  volume =       {36},
  number =       {1},
  pages =        {204-225},
  year =         {1998},
  doi =          {10.1137/S0036142997326203},
  url =          {https://doi.org/10.1137/S0036142997326203},
}
""")

Citations().add("nixonhill2023consistent", """
@article{nixonhill2023consistent,
  author        = {Nixon-Hill, R. W. and Shapero, D. and Cotter, C. J. and Ham, D. A.},
  doi           = {10.5194/gmd-17-5369-2024},
  journal       = {Geoscientific Model Development},
  number        = {13},
  pages         = {5369--5386},
  title         = {Consistent point data assimilation in Firedrake and Icepack},
  url           = {https://gmd.copernicus.org/articles/17/5369/2024/},
  volume        = {17},
  year          = {2024}
}
""")

Citations().add("FiredrakeUserManual", """
@manual{FiredrakeUserManual,
  author        = {David A. Ham and Paul H. J. Kelly and Lawrence
Mitchell and Colin J. Cotter and Robert C. Kirby and Koki Sagiyama and
Nacime Bouziani and Sophia Vorderwuelbecke and Thomas J. Gregory and
Jack Betteridge and Daniel R. Shapero and Reuben W. Nixon-Hill and
Connor J. Ward and Patrick E. Farrell and Pablo D. Brubeck and India
Marsden and Thomas H. Gibson and Miklós Homolya and Tianjiao Sun and
Andrew T. T. McRae and Fabio Luporini and Alastair Gregory and
Michael Lange and Simon W. Funke and Florian Rathgeber and
Gheorghe-Teodor Bercea and Graham R. Markall},
  doi           = {10.25561/104839},
  edition       = {First edition},
  month         = {5},
  organization  = {Imperial College London and University of Oxford and
Baylor University and University of Washington},
  title         = {Firedrake User Manual},
  year          = {2023}
}
""")

Citations().add("Bouziani2021", """
@article{Bouziani2021,
  title={Escaping the abstraction: a foreign function interface for the {Unified} {Form} {Language} {[UFL]}},
  author={Bouziani, Nacime and Ham, David A},
  journal = {{Differentiable} {Programming} {Workshop} at {NeurIPS} 2021},
  url = {http://arxiv.org/abs/2111.00945},
  note = {arXiv: 2111.00945},
  year={2021}
}
""")

Citations().add("Bouziani2023", """
@inproceedings{Bouziani2023,
 title = {Physics-driven machine learning models coupling {PyTorch} and {Firedrake}},
 author = {Bouziani, Nacime and Ham, David A.},
 booktitle = {{ICLR} 2023 {Workshop} on {Physics} for {Machine} {Learning}},
 year = {2023},
 doi = {10.48550/arXiv.2303.06871}
}
""")
