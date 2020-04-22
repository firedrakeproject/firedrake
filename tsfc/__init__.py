from tsfc.driver import compile_form, compile_expression_dual_evaluation  # noqa: F401
from tsfc.parameters import default_parameters  # noqa: F401

try:
    from firedrake_citations import Citations
    Citations().add("Kirby2006", """
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
    del Citations
except ImportError:
    pass
