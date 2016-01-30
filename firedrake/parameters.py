"""The parameters dictionary contains global parameter settings."""
from __future__ import absolute_import

from firedrake.fc.constants import default_parameters
from pyop2.configuration import configuration
from firedrake.citations import Citations


__all__ = ['Parameters', 'parameters', 'disable_performance_optimisations']


class Parameters(dict):
    def __init__(self, name=None, **kwargs):
        self._name = name
        self._update_function = None

        for key, value in kwargs.iteritems():
            self.add(key, value)

    def add(self, key, value=None):
        if isinstance(key, Parameters):
            self[key.name()] = key
        else:
            self[key] = value

    def __setitem__(self, key, value):
        super(Parameters, self).__setitem__(key, value)
        if self._update_function:
            self._update_function(key, value)

    def name(self):
        return self._name

    def rename(self, name):
        self._name = name

    def __getstate__(self):
        # Remove non-picklable update function slot
        d = self.__dict__.copy()
        del d["_update_function"]
        return d

    def set_update_function(self, callable):
        """Set a function to be called whenever a dictionary entry is changed.

        :arg callable: the function.

        The function receives two arguments, the key-value pair of
        updated entries."""
        self._update_function = callable


parameters = Parameters()
"""A nested dictionary of parameters used by Firedrake"""

parameters.add(Parameters("assembly_cache",
                          enabled=True,
                          eviction=True,
                          max_bytes=float("Inf"),
                          max_factor=0.6,
                          max_misses=3))

parameters.add(Parameters("coffee",
                          O2=False))

# Spew citation for coffee paper if user modifies coffee options.
parameters["coffee"].set_update_function(lambda k, v: Citations().register("Luporini2015"))

# Default to the values of PyOP2 configuration dictionary
pyop2_opts = Parameters("pyop2_options",
                        **configuration)

pyop2_opts.set_update_function(lambda k, v: configuration.unsafe_reconfigure(**{k: v}))

# Override values
pyop2_opts["type_check"] = True
pyop2_opts["log_level"] = "INFO"

parameters.add(pyop2_opts)

ffc_parameters = default_parameters()
parameters.add(Parameters("form_compiler", **ffc_parameters))

parameters["reorder_meshes"] = True

parameters["matnest"] = True

parameters["type_check_safe_par_loops"] = False


def disable_performance_optimisations():
    """Switches off performance optimisations in Firedrake.

    This is mostly useful for debugging purposes.

    This switches off all of COFFEE's kernel compilation optimisations
    and enables PyOP2's runtime checking of par_loop arguments in all
    cases (even those where they are claimed safe).  Additionally, it
    switches to compiling generated code in debug mode.

    Returns a function that can be called with no arguments, to
    restore the state of the parameters dict."""

    check = parameters["pyop2_options"]["type_check"]
    debug = parameters["pyop2_options"]["debug"]
    lazy = parameters["pyop2_options"]["lazy_evaluation"]
    safe_check = parameters["type_check_safe_par_loops"]
    coffee = parameters["coffee"]
    cache = parameters["assembly_cache"]["enabled"]

    def restore():
        parameters["pyop2_options"]["type_check"] = check
        parameters["pyop2_options"]["debug"] = debug
        parameters["pyop2_options"]["lazy_evaluation"] = lazy
        parameters["type_check_safe_par_loops"] = safe_check
        parameters["coffee"] = coffee
        parameters["assembly_cache"]["enabled"] = cache

    parameters["pyop2_options"]["type_check"] = True
    parameters["pyop2_options"]["debug"] = True
    parameters["pyop2_options"]["lazy_evaluation"] = False
    parameters["type_check_safe_par_loops"] = True
    parameters["coffee"] = {}
    parameters["assembly_cache"]["enabled"] = False

    return restore
