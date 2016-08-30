from __future__ import absolute_import

from pyop2 import op2
from pyop2.utils import flatten

from firedrake import functionspace
from . import impl
from . import utils
from .utils import set_level, get_level


__all__ = ["FunctionSpaceHierarchy", "VectorFunctionSpaceHierarchy",
           "TensorFunctionSpaceHierarchy", "MixedFunctionSpaceHierarchy"]


def coarsen(dm, comm):
    m = dm.getAttr("__fs__")()
    if m is None:
        raise RuntimeError("No functionspace found on DM")
    hierarchy, level = get_level(m)
    if level < 1:
        raise RuntimeError("Cannot coarsen coarsest DM")
    return hierarchy[level-1]._dm


def refine(dm, comm):
    m = dm.getAttr("__fs__")()
    if m is None:
        raise RuntimeError("No functionspace found on DM")
    hierarchy, level = get_level(m)
    if level >= len(hierarchy) - 1:
        raise RuntimeError("Cannot refine finest DM")
    return hierarchy[level+1]._dm


class BaseHierarchy(object):

    def __init__(self, spaces):
        """
        Build a hierarchy of function spaces

        :arg spaces: The spaces to use.
        """
        self._mesh_hierarchy, _ = get_level(spaces[0].ufl_domain())
        if self._mesh_hierarchy is None:
            raise ValueError("Provided spaces are not from a hierarchy")

        self._hierarchy = tuple([set_level(V, self, lvl)
                                 for lvl, V in enumerate(spaces)])
        self._map_cache = {}
        self._cell_sets = tuple(op2.LocalSet(m.cell_set) for m in self._mesh_hierarchy)
        self._ufl_element = self[0].ufl_element()
        self._restriction_weights = None
        fiat_element = spaces[0].fiat_element
        ncelldof = len(fiat_element.entity_dofs()[spaces[0].mesh().cell_dimension()][0])
        self._discontinuous = ncelldof == spaces[0].cell_node_map().arity
        element = self[0].fiat_element
        omap = self[1].cell_node_map().values
        c2f, vperm = self._mesh_hierarchy._cells_vperm[0]

        self.dim = spaces[0].dim
        self.shape = spaces[0].shape

        indices, _ = utils.get_unique_indices(element,
                                              omap[c2f[0, :], ...].reshape(-1),
                                              vperm[0, :],
                                              offset=None)
        self._prolong_kernel = utils.get_prolongation_kernel(element, indices, self.dim)
        self._restrict_kernel = utils.get_restriction_kernel(element, indices, self.dim,
                                                             no_weights=self._discontinuous)
        self._inject_kernel = utils.get_injection_kernel(element, indices, self.dim)

        for V in self:
            dm = V._dm
            dm.setCoarsen(coarsen)
            dm.setRefine(refine)

        # Refine the DM so that we can find where it lives in the
        # hierarchy.
        for V in self[:-1]:
            V._dm.refine()

    def __len__(self):
        """Return the size of this function space hierarchy"""
        return len(self._hierarchy)

    def __iter__(self):
        """Iterate over the function spaces in this hierarchy (from
        coarse to fine)."""
        for fs in self._hierarchy:
            yield fs

    def __getitem__(self, idx):
        """Return a function space in the hierarchy

        :arg idx: The :class:`~.FunctionSpace` to return"""
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

        map_vals, offset = impl.create_cell_node_map(Vc, Vf, c2f, vperm)
        map = op2.Map(self._cell_sets[level],
                      Vf.node_set,
                      map_vals.shape[1],
                      map_vals, offset=offset)
        self._map_cache[level] = map
        return map

    def split(self):
        """Return a tuple of this :class:`FunctionSpaceHierarchy`."""
        return (self, )

    def __mul__(self, other):
        """Create a :class:`MixedFunctionSpaceHierarchy`.

        :arg other: The other :class:`FunctionSpaceHierarchy` in the
             mixed space."""
        return MixedFunctionSpaceHierarchy([self, other])


def FunctionSpaceHierarchy(mesh_hierarchy, family, degree=None,
                           name=None, vfamily=None, vdegree=None):
    """Build a hierarchy of function spaces.

    :arg mesh_hierarchy: A :class:`~.MeshHierarchy` on which to build
        the spaces.

    See :func:`~.FunctionSpace` for a description of the
    remaining arguments.
    """
    spaces = [functionspace.FunctionSpace(mesh, family, degree=degree,
                                          name=name, vfamily=vfamily,
                                          vdegree=vdegree)
              for mesh in mesh_hierarchy]
    return BaseHierarchy(spaces)


def VectorFunctionSpaceHierarchy(mesh_hierarchy, family, degree=None,
                                 dim=None, name=None, vfamily=None,
                                 vdegree=None):
    """Build a hierarchy of vector function spaces.

    :arg mesh_hierarchy: A :class:`~.MeshHierarchy` on which to build
        the spaces.

    See :func:`~.VectorFunctionSpace` for a description of the
    remaining arguments.
    """
    spaces = [functionspace.VectorFunctionSpace(mesh, family, degree,
                                                dim=dim, name=name,
                                                vfamily=vfamily,
                                                vdegree=vdegree)
              for mesh in mesh_hierarchy]
    return BaseHierarchy(spaces)


def TensorFunctionSpaceHierarchy(mesh_hierarchy, family, degree=None,
                                 shape=None, symmetry=None, name=None,
                                 vfamily=None, vdegree=None):
    """Build a hierarchy of tensor function spaces.

    :arg mesh_hierarchy: A :class:`~.MeshHierarchy` on which to build
        the spaces.

    See :func:`~.TensorFunctionSpace` for a description of the
    remaining arguments.
    """
    spaces = [functionspace.TensorFunctionSpace(mesh, family, degree,
                                                shape=shape, symmetry=symmetry,
                                                name=name,
                                                vfamily=vfamily,
                                                vdegree=vdegree)
              for mesh in mesh_hierarchy]
    return BaseHierarchy(spaces)


class MixedFunctionSpaceHierarchy(object):

    """Build a hierarchy of mixed function spaces.

    This is effectively a bag for a number of
    :class:`FunctionSpaceHierarchy`\s.  At each level in the hierarchy
    is a :class:`~.MixedFunctionSpace`.
    """
    def __init__(self, spaces, name=None):
        """
        :arg spaces: A list of :class:`FunctionSpaceHierarchy`\s
        """
        spaces = [x for x in flatten([s.split() for s in spaces])]
        assert all(isinstance(s, BaseHierarchy) for s in spaces)
        self._hierarchy = tuple([set_level(functionspace.MixedFunctionSpace(s), self, lvl)
                                for lvl, s in enumerate(zip(*spaces))])
        # Attach level info to the new ProxyFunctionSpaces inside the mixed spaces.
        for lvl, mixed_space in enumerate(self._hierarchy):
            for i, space in enumerate(mixed_space):
                set_level(space, spaces[i], lvl)
        self._spaces = tuple(spaces)
        self._ufl_element = self._hierarchy[0].ufl_element()
        for V in self:
            dm = V._dm
            dm.setCoarsen(coarsen)
            dm.setRefine(refine)
        # Refine the DM so that we can find where it lives in the
        # hierarchy.
        for V in self[:-1]:
            V._dm.refine()

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
