"""The parameters dictionary contains global parameter settings."""
from __future__ import absolute_import

from ffc import default_parameters
from pyop2.configuration import configuration

__all__ = ['Parameters', 'parameters']


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

    def set_update_function(self, callable):
        """Set a function to be called whenever a dictionary entry is changed.

        :arg callable: the function.

        The function receives two arguments, the key-value pair of
        updated entries."""
        self._update_function = callable


parameters = Parameters()

parameters.add(Parameters("assembly_cache",
                          enabled=False,
                          eviction=True,
                          max_bytes=float("Inf"),
                          max_factor=0.6,
                          max_misses=3))

parameters.add(Parameters("coffee",
                          O2=True))

# Default to the values of PyOP2 configuration dictionary
pyop2_opts = Parameters("pyop2_options",
                        **configuration)

pyop2_opts.set_update_function(lambda k, v: configuration.reconfigure(**{k: v}))

# Override values
pyop2_opts["type_check"] = False
pyop2_opts["log_level"] = "INFO"

parameters.add(pyop2_opts)

ffc_parameters = default_parameters()
ffc_parameters['write_file'] = False
ffc_parameters['format'] = 'pyop2'
ffc_parameters['representation'] = 'quadrature'
ffc_parameters['pyop2-ir'] = True
parameters.add(Parameters("form_compiler", **ffc_parameters))

parameters["reorder_meshes"] = True

parameters["matnest"] = True
