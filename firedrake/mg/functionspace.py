from pyop2 import op2
from pyop2.utils import flatten

from firedrake import functionspace
from firedrake.mg import impl


__all__ = ["FunctionSpaceHierarchy", "VectorFunctionSpaceHierarchy",
           "MixedFunctionSpaceHierarchy"]


class BaseHierarchy(object):

    def __init__(self, mesh_hierarchy, fses):
        self._mesh_hierarchy = mesh_hierarchy
        self._hierarchy = tuple(fses)
        self._map_cache = {}
        self._cell_sets = tuple(op2.LocalSet(m.cell_set) for m in self._mesh_hierarchy)
        self._ufl_element = self[0].ufl_element()

    def __len__(self):
        return len(self._hierarchy)

    def __iter__(self):
        for fs in self._hierarchy:
            yield fs

    def __getitem__(self, idx):
        return self._hierarchy[idx]

    def ufl_element(self):
        return self._ufl_element

    def cell_node_map(self, level):
        """A :class:`pyop2.Map` from cells on a coarse mesh to the
        corresponding degrees of freedom on a the fine mesh below it.

        :arg level: the coarse level the map should be from.
        """
        if not 0 <= level < len(self) - 1:
            raise RuntimeError("Requested coarse level %d outside permissible range [0, %d)" %
                               (level, len(self) - 1))
        try:
            return self._map_cache[level]
        except KeyError:
            pass
        Vc = self._hierarchy[level]
        Vf = self._hierarchy[level + 1]

        c2f, vperm = self._mesh_hierarchy._cells_vperm[level]

        map_vals = impl.create_cell_node_map(Vc, Vf, c2f, vperm)
        map = op2.Map(self._cell_sets[level],
                      Vf.node_set,
                      map_vals.shape[1],
                      map_vals)
        self._map_cache[level] = map
        return map

    def split(self):
        return (self, )

    def __mul__(self, other):
        return MixedFunctionSpaceHierarchy([self, other])


class FunctionSpaceHierarchy(BaseHierarchy):
    """Build a hierarchy of function spaces.

    Given a hierarchy of meshes, this constructs a hierarchy of
    function spaces, with the property that every coarse space is a
    subspace of the fine spaces that are a refinement of it.
    """
    def __init__(self, mesh_hierarchy, family, degree=None,
                 name=None, vfamily=None, vdegree=None):
        """
        :arg mesh_hierarchy: a :class:`.MeshHierarchy` to build the
             function spaces on.
        :arg family: the function space family
        :arg degree: the degree of the function space
        """
        fses = [functionspace.FunctionSpace(m, family, degree=degree,
                                            name=name, vfamily=vfamily,
                                            vdegree=vdegree)
                for m in mesh_hierarchy]
        self.dim = 1
        super(FunctionSpaceHierarchy, self).__init__(mesh_hierarchy, fses)


class VectorFunctionSpaceHierarchy(BaseHierarchy):

    def __init__(self, mesh_hierarchy, family, degree, dim=None, name=None, vfamily=None, vdegree=None):
        fses = [functionspace.VectorFunctionSpace(m, family, degree,
                                                  dim=dim, name=name, vfamily=vfamily,
                                                  vdegree=vdegree)
                for m in mesh_hierarchy]
        self.dim = fses[0].dim
        super(VectorFunctionSpaceHierarchy, self).__init__(mesh_hierarchy, fses)


class MixedFunctionSpaceHierarchy(object):

    def __init__(self, spaces, name=None):
        spaces = [x for x in flatten([s.split() for s in spaces])]
        assert all(isinstance(s, BaseHierarchy) for s in spaces)
        self._hierarchy = tuple(functionspace.MixedFunctionSpace(s) for s in zip(*spaces))
        self._spaces = tuple(spaces)
        self._ufl_element = self._hierarchy[0].ufl_element()

    def __mul__(self, other):
        """Create a :class:`MixedFunctionSpaceHierarchy`.

        :arg other: The other :class:`FunctionSpaceHierarchy` in the
             mixed space."""
        return MixedFunctionSpaceHierarchy([self, other])

    def __len__(self):
        """Return the size of this function space hierarchy"""
        return len(self._hierarchy)

    def __iter__(self):
        """Iterate over the mixed function spaces in this hierarchy (from
        coarse to fine)."""
        for fs in self._hierarchy:
            yield fs

    def __getitem__(self, idx):
        """Return a function space in the hierarchy

        :arg idx: The :class:`~.MixedFunctionSpace` to return"""
        return self._hierarchy[idx]

    def split(self):
        """Return a tuple of the constituent
        :class:`FunctionSpaceHierarchy`\s in this mixed hierarchy."""
        return self._spaces

    def cell_node_map(self, level):
        """A :class:`pyop2.MixedMap` from cells on a coarse mesh to the
        corresponding degrees of freedom on a the fine mesh below it.

        :arg level: the coarse level the map should be from.
        """
        return op2.MixedMap(s.cell_node_map(level) for s in self.split())
