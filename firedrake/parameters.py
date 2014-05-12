__all__ = ['Parameters', 'parameters']

"""The parameters dictionary contains global parameter settings."""


class Parameters(dict):
    def __init__(self, name=None, **kwargs):
        self._name = name

        for key, value in kwargs.iteritems():
            self.add(key, value)

    def add(self, key, value=None):
        if value is not None:
            self[key] = value
        else:
            self[key.name()] = key

    def name(self):
        return self._name

    def rename(self, name):
        self._name = name

parameters = Parameters()

parameters["assembly_cache"] = Parameters(enabled=True,
                                          eviction=True,
                                          max_bytes=float("Inf"),
                                          max_factor=0.6,
                                          max_misses=3)

parameters["reorder_meshes"] = True
