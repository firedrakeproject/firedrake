"""The parameters dictionary contains global parameter settings."""

import os
from ffc import default_parameters

__all__ = ['Parameters', 'parameters']


class Parameters(dict):
    def __init__(self, name=None, **kwargs):
        self._name = name

        for key, value in kwargs.iteritems():
            self.add(key, value)

    def add(self, key, value=None):
        if isinstance(key, Parameters):
            self[key.name()] = key
        else:
            self[key] = value

    def name(self):
        return self._name

    def rename(self, name):
        self._name = name

parameters = Parameters()

parameters.add(Parameters("assembly_cache",
                          enabled=True,
                          eviction=True,
                          max_bytes=float("Inf"),
                          max_factor=0.6,
                          max_misses=3))

parameters.add(Parameters("coffee",
                          compiler=os.environ.get('PYOP2_BACKEND_COMPILER', 'gnu'),
                          simd_isa=os.environ.get('PYOP2_SIMD_ISA', 'sse'),
                          O2=True))

ffc_parameters = default_parameters()
ffc_parameters['write_file'] = False
ffc_parameters['format'] = 'pyop2'
ffc_parameters['representation'] = 'quadrature'
ffc_parameters['pyop2-ir'] = True
parameters.add(Parameters("form_compiler", **ffc_parameters))

parameters["reorder_meshes"] = True
