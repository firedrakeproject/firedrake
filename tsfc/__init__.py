from tsfc.driver import compile_form, compile_expression_at_points  # noqa: F401
from tsfc.parameters import default_parameters  # noqa: F401

try:
    from firedrake_citations import Citations
    Citations().register("Homolya2017")
    del Citations
except ImportError:
    pass
