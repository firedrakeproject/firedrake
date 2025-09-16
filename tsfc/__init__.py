from tsfc.driver import compile_form, compile_expression_dual_evaluation  # noqa: F401
from tsfc.parameters import default_parameters  # noqa: F401


def register_citations():
    import petsctools

    petsctools.add_citation("Kirby2006", """
    @Article{Kirby2006,
      author =       {Kirby, Robert C. and Logg, Anders},
      title =        {A Compiler for Variational Forms},
      journal =      {ACM Trans. Math. Softw.},
      year =         2006,
      volume =       32,
      number =       3,
      pages =        {417--444},
      month =        sep,
      numpages =     28,
      doi =          {10.1145/1163641.1163644},
      acmid =        1163644,
    }""")

    petsctools.add_citation("Luporini2016", """
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

    petsctools.add_citation("Homolya2017", """
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

    petsctools.add_citation("Homolya2017a", """
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


register_citations()
del register_citations
