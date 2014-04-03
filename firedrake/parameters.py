__all__ = ['Parameters', 'parameters']

"""The parameters dictionary contains global parameter settings."""


class Parameters(dict):
    def __init__(self, name=None):
        self._name = name

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

parameters["assembly_cache"] = {"enabled": True}
