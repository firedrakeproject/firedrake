from firedrake import function
from firedrake.mg import functionspace


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
            self._hierarchy = tuple(functions)
        else:
            self._hierarchy = tuple(function.Function(f) for f in fs_hierarchy)

        if isinstance(self._function_space, functionspace.MixedFunctionSpaceHierarchy):
            split = []
            for i, fs in enumerate(self.function_space().split()):
                split.append(FunctionHierarchy(fs, [f.split()[i] for f in self]))
            self._split = tuple(split)
        else:
            self._split = (self, )

    def __iter__(self):
        for f in self._hierarchy:
            yield f

    def __len__(self):
        return len(self._hierarchy)

    def __getitem__(self, idx):
        return self._hierarchy[idx]

    def split(self):
        return self._split

    def function_space(self):
        return self._function_space

    def cell_node_map(self, level):
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
