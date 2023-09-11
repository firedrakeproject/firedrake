import firedrake.utils as utils
from .compat import compat


class Backend:
    @utils.cached_property
    def backend(self):
        import firedrake
        return firedrake

    @utils.cached_property
    def compat(self):
        import firedrake
        return compat(firedrake)
