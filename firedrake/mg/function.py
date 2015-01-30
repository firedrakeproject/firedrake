from __future__ import absolute_import

from firedrake import function
from . import functionspace
from .utils import set_level

__all__ = ["FunctionHierarchy"]


class FunctionHierarchy(object):
    """Build a hierarchy of :class:`~.Function`\s"""
    def __init__(self, fs_hierarchy, functions=None):
        """
        :arg fs_hierarchy: the :class:`~.FunctionSpaceHierarchy` to build on.
        :arg functions: optional :class:`~.Function` for each level in
             the hierarchy.
        """
        self._function_space = fs_hierarchy

        if functions is not None:
            assert all(isinstance(f, function.Function) for f in functions)
            assert len(functions) == len(self._function_space)
            self._hierarchy = tuple([set_level(f, self, lvl)
                                     for lvl, f in enumerate(functions)])
        else:
            self._hierarchy = tuple([set_level(function.Function(f),
                                               self, lvl)
                                     for lvl, f in enumerate(fs_hierarchy)])

        if isinstance(self._function_space, functionspace.MixedFunctionSpaceHierarchy):
            split = []
            for i, fs in enumerate(self.function_space().split()):
                split.append(FunctionHierarchy(fs, [f.split()[i] for f in self]))
            self._split = tuple(split)
        else:
            self._split = (self, )

    def __iter__(self):
        """Iterate over the :class:`Function`\s in this hierarchy (from
        coarse to fine)."""
        for f in self._hierarchy:
            yield f

    def __len__(self):
        """Return the size of this function hierarchy"""
        return len(self._hierarchy)

    def __getitem__(self, idx):
        """Return a function in the hierarchy

        :arg idx: the :class:`~.Function` to return"""
        return self._hierarchy[idx]

    def split(self):
        """Return a tuple of the constituent
        :class:`FunctionHierarchy`\s in this
        :class:`FunctionHierarchy`.  This is just a single tuple
        unless the space was a
        :class:`~.MixedFunctionSpaceHierarchy`.
        """
        return self._split

    def function_space(self):
        """Return the :class:`~.FunctionSpaceHierarchy` this
        :class:`FunctionHierarchy` is built on."""
        return self._function_space

    def cell_node_map(self, level):
        """A :class:`pyop2.Map` from cells on a coarse mesh to the
        corresponding degrees of freedom on a the fine mesh below it.

        :arg level: the coarse level the map should be from.
        """
        return self._function_space.cell_node_map(level)

    def prolong(self, level):
        """Prolong from a coarse to the next finest hierarchy level.

        :arg level: The coarse level to prolong from"""
        self.function_space().prolong(self, level)

    def restrict(self, level):
        """Restrict from a fine to the next coarsest hierarchy level.

        :arg level: The fine level to restrict from
        """
        self.function_space().restrict(self, level)

    def inject(self, level):
        """Inject from a fine to the next coarsest hierarchy level.

        :arg level: the fine level to inject from"""
        self.function_space().inject(self, level)

    def assign(self, other):
        """Assign into this :class:`FunctionHierarchy`.

        :arg other: another :class:`FunctionHierarchy` built on the
            same :class:`~.FunctionSpaceHierarchy` or a scalar value
            (e.g. a :class:`~.Constant` or a literal float)."""
        try:
            for (self_, other_) in zip(self, other):
                self_.assign(other_)
        except (TypeError, NotImplementedError):
            for self_ in self:
                self_.assign(other)
