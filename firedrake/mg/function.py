from __future__ import absolute_import

import firedrake

from firedrake.logging import warning, RED

__all__ = ["FunctionHierarchy"]


def FunctionHierarchy(fs_hierarchy, functions=None):
    """ outdated and returns warning & list of functions corresponding to each level
    of a functionspace hierarchy

        :arg fs_hierarchy: the :class:`~.FunctionSpaceHierarchy` to build on.
        :arg functions: optional :class:`~.Function` for each level.

    """

    warning(RED % "FunctionHierarchy is obsolete. Falls back by returning list of functions")

    if functions is not None:
        assert len(functions) == len(fs_hierarchy)
        return tuple(functions)
    else:
        return tuple([firedrake.Function(f) for f in fs_hierarchy])
