from __future__ import absolute_import

import firedrake

from firedrake.logging import warning, RED

__all__ = ["FunctionHierarchy"]


class FunctionHierarchy(object):
    """ outdated and returns warning & list of functions corresponding to each level
    of a functionspace hierarchy """
    def __init__(self, fs_hierarchy, functions=None):
        """
        :arg fs_hierarchy: the :class:`~.FunctionSpaceHierarchy` to build on.
        :arg functions: optional :class:`~.Function` for each level.
        """

        if functions is not None:
            assert all(isinstance(f, firedrake.Function) for f in functions)
            assert len(functions) == len(fs_hierarchy)
            self._list = tuple([f for f in functions])
        else:
            self._list = tuple([firedrake.Function(f) for f in fs_hierarchy])

        warning(RED % "FunctionHierarchy is obsolete. Falls back by returning list of functions")

    def __iter__(self):
        """Iterate over the :class:`Function`\s in this list (from
        coarse to fine)."""
        for f in self._list:
            yield f

    def __len__(self):
        """Return the size of this list"""
        return len(self._list)

    def __getitem__(self, idx):
        """Return a function in the list

        :arg idx: the :class:`~.Function` to return"""
        return self._list[idx]
