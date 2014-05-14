__all__ = ['Parameters', 'parameters']

"""The parameters dictionary contains global parameter settings."""


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
                          compiler='gnu',
                          simd_isa='avx',
                          licm=False,
                          slice=None,
                          vect=None,
                          ap=False,
                          split=None))

parameters["reorder_meshes"] = True
