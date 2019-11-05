"""The parameters dictionary contains global parameter settings."""
from coffee import coffee_reconfigure
from pyop2.configuration import configuration
from tsfc import default_parameters
import sys

max_float = sys.float_info[0]

__all__ = ['Parameters', 'parameters', 'disable_performance_optimisations']


class Parameters(dict):
    def __init__(self, name=None, **kwargs):
        self._name = name
        self._update_function = None

        for key, value in kwargs.items():
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

# The COFFEE default optimization level is O2
coffee_default_optlevel = "Ov"
coffee_opts = Parameters("coffee", optlevel=coffee_default_optlevel)
coffee_opts.set_update_function(lambda k, v: coffee_reconfigure(**{k: v}))
parameters.add(coffee_opts)

# Default to the values of PyOP2 configuration dictionary
pyop2_opts = Parameters("pyop2_options",
                        **configuration)

pyop2_opts.set_update_function(lambda k, v: configuration.unsafe_reconfigure(**{k: v}))

# Override values
pyop2_opts["type_check"] = True

# PyOP2 must know about the COFFEE optimization level chosen by Firedrake
pyop2_opts["opt_level"] = coffee_default_optlevel

parameters.add(pyop2_opts)

parameters.add(Parameters("form_compiler", **default_parameters()))

parameters["reorder_meshes"] = True

# One of nest, aij, baij or matfree
parameters["default_matrix_type"] = "nest"
# One of aij or baij
parameters["default_sub_matrix_type"] = "baij"

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
    safe_check = parameters["type_check_safe_par_loops"]
    coffee = parameters["coffee"]

    def restore():
        parameters["pyop2_options"]["type_check"] = check
        parameters["pyop2_options"]["debug"] = debug
        parameters["type_check_safe_par_loops"] = safe_check
        parameters["coffee"] = coffee

    parameters["pyop2_options"]["type_check"] = True
    parameters["pyop2_options"]["debug"] = True
    parameters["type_check_safe_par_loops"] = True
    parameters["coffee"] = {}

    return restore
