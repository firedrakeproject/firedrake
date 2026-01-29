"""The parameters dictionary contains global parameter settings."""
import dataclasses

from pyop3.config import config as PYOP3_CONFIG
from pyop3.ir.lower import LOOPY_TARGET
from tsfc import default_parameters
import sys
from firedrake.utils import ScalarType, ScalarType_c

max_float = sys.float_info[0]

__all__ = ['Parameters', 'parameters']


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
        if hasattr(self, "_update_function") and self._update_function:
            self._update_function(key, value)

    def name(self):
        return self._name

    def rename(self, name):
        self._name = name

    def __getstate__(self):
        # Remove non-picklable update function slot
        d = self.__dict__.copy()
        try:
            del d["_update_function"]
        except KeyError:
            pass
        return d

    def set_update_function(self, callable):
        """Set a function to be called whenever a dictionary entry is changed.

        :arg callable: the function.

        The function receives two arguments, the key-value pair of
        updated entries."""
        self._update_function = callable


parameters = Parameters()
"""A nested dictionary of parameters used by Firedrake"""

# Default to the values of PyOP2 configuration dictionary
pyop2_opts = Parameters("pyop2_options",
                        **dataclasses.asdict(PYOP3_CONFIG))

target = LOOPY_TARGET

parameters.add(pyop2_opts)

parameters.add(Parameters("form_compiler", **default_parameters()))
parameters["form_compiler"]['scalar_type'] = ScalarType
parameters["form_compiler"]['scalar_type_c'] = ScalarType_c

parameters["reorder_meshes"] = True

# One of nest, aij, baij or matfree
parameters["default_matrix_type"] = "aij"
# One of aij or baij
parameters["default_sub_matrix_type"] = "baij"

parameters["type_check_safe_par_loops"] = False

parameters.add(Parameters("slate_compiler"))
parameters["slate_compiler"]["optimise"] = True
# Should a Slate multiplication be replaced by an action?
parameters["slate_compiler"]["replace_mul"] = False
