from __future__ import absolute_import

from pyop2 import op2
from pyop2.utils import flatten

from firedrake import functionspace
from . import impl
from . import utils
from .utils import set_level, get_level
from . import interface


__all__ = ["FunctionSpaceHierarchy", "VectorFunctionSpaceHierarchy",
           "MixedFunctionSpaceHierarchy"]


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

    def __init__(self, mesh_hierarchy, fses):
        """
        Build a hierarchy of function spaces

        :arg mesh_hierarchy: a :class:`~.MeshHierarchy` on which to
             build the function spaces.
        :arg fses: an iterable of :class:`~.FunctionSpace`\s.
        """
        self.refinements_per_level = mesh_hierarchy.refinements_per_level
        self._mesh_hierarchy = mesh_hierarchy
        hierarchy = tuple([set_level(fs, self, lvl)
                          for lvl, fs in enumerate(fses)])
        self._map_cache = {}
        self._cell_sets = tuple(op2.LocalSet(m.cell_set) for m in self._mesh_hierarchy._full_hierarchy)
        self._ufl_element = hierarchy[0].ufl_element()
        self._restriction_weights = None
        fiat_element = fses[0].fiat_element
        ncelldof = len(fiat_element.entity_dofs()[fses[0].mesh().cell_dimension()][0])
        self._discontinuous = ncelldof == fses[0].cell_node_map().arity
        try:
            element = hierarchy[0].fiat_element
            omap = hierarchy[1].cell_node_map().values
            c2f, vperm = self._mesh_hierarchy._cells_vperm[0]
            indices, _ = utils.get_unique_indices(element,
                                                  omap[c2f[0, :], ...].reshape(-1),
                                                  vperm[0, :],
                                                  offset=None)
            self._prolong_kernel = utils.get_prolongation_kernel(element, indices, self.dim)
            self._restrict_kernel = utils.get_restriction_kernel(element, indices, self.dim,
                                                                 no_weights=self._discontinuous)
            self._inject_kernel = utils.get_injection_kernel(element, indices, self.dim)
        except:
            pass

        for V in hierarchy:
            dm = V._dm
            dm.setCoarsen(coarsen)
            dm.setRefine(refine)

        # reset the hierarchy with skipped refinements and set full hierarchy
        self._full_hierarchy = hierarchy
        self._hierarchy = hierarchy[::self.refinements_per_level]

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
        if not 0 <= level < len(self._full_hierarchy) - 1:
            raise RuntimeError("Requested coarse level %d outside permissible range [0, %d)" %
                               (level, len(self._full_hierarchy) - 1))
        try:
            return self._map_cache[level]
        except KeyError:
            pass
        Vc = self._full_hierarchy[level]
        Vf = self._full_hierarchy[level + 1]

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

    def restrict(self, residual, level):
        """
        Restrict a residual (a dual quantity) from level to level-1

        :arg residual: the residual to restrict
        :arg level: the fine level to restrict from
        """
        coarse = residual[level-1]
        fine = residual[level]
        interface.restrict(fine, coarse)

    def inject(self, state, level):
        """
        Inject state (a primal quantity) from level to level - 1

        :arg state: the state to inject
        :arg level: the fine level to inject from
        """
        coarse = state[level-1]
        fine = state[level]
        interface.inject(fine, coarse)

    def prolong(self, solution, level):
        """
        Prolong a solution (a primal quantity) from level - 1 to level

        :arg solution: the solution to prolong
        :arg level: the coarse level to prolong from
        """
        coarse = solution[level]
        fine = solution[level+1]
        interface.prolong(coarse, fine)


class FunctionSpaceHierarchy(BaseHierarchy):
    """Build a hierarchy of function spaces.

    Given a hierarchy of meshes, this constructs a hierarchy of
    function spaces, with the property that every coarse space is a
    subspace of the fine spaces that are a refinement of it.
    """
    def __init__(self, mesh_hierarchy, family, degree=None,
                 name=None, vfamily=None, vdegree=None):
        """
        :arg mesh_hierarchy: a :class:`~.MeshHierarchy` to build the
             function spaces on.
        :arg family: the function space family
        :arg degree: the degree of the function space

        See :class:`~.FunctionSpace` for more details on the form of
        the remaining parameters.
        """
        fses = [functionspace.FunctionSpace(m, family, degree=degree,
                                            name=name, vfamily=vfamily,
                                            vdegree=vdegree)
                for m in mesh_hierarchy._full_hierarchy]
        self.dim = 1
        super(FunctionSpaceHierarchy, self).__init__(mesh_hierarchy, fses)


class VectorFunctionSpaceHierarchy(BaseHierarchy):

    """Build a hierarchy of vector function spaces.

    Given a hierarchy of meshes, this constructs a hierarchy of vector
    function spaces, with the property that every coarse space is a
    subspace of the fine spaces that are a refinement of it.

    """
    def __init__(self, mesh_hierarchy, family, degree, dim=None, name=None, vfamily=None, vdegree=None):
        """
        :arg mesh_hierarchy: a :class:`~.MeshHierarchy` to build the
             function spaces on.
        :arg family: the function space family
        :arg degree: the degree of the function space

        See :class:`~.VectorFunctionSpace` for more details on the form of
        the remaining parameters.
        """
        fses = [functionspace.VectorFunctionSpace(m, family, degree,
                                                  dim=dim, name=name, vfamily=vfamily,
                                                  vdegree=vdegree)
                for m in mesh_hierarchy._full_hierarchy]
        self.dim = fses[0].dim
        super(VectorFunctionSpaceHierarchy, self).__init__(mesh_hierarchy, fses)


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
        # swap full hierarchies will refined hierarchies
        for s in range(len(spaces)):
            spaces[s]._hierarchy = spaces[s]._full_hierarchy

        spaces = [x for x in flatten([s.split() for s in spaces])]
        self.refinements_per_level = spaces[0].refinements_per_level
        assert all(isinstance(s, BaseHierarchy) for s in spaces)
        hierarchy = tuple([set_level(functionspace.MixedFunctionSpace(s), self, lvl)
                          for lvl, s in enumerate(zip(*spaces))])
        # Attach level info to the new ProxyFunctionSpaces inside the mixed spaces.
        for lvl, mixed_space in enumerate(hierarchy):
            for i, space in enumerate(mixed_space):
                set_level(space, spaces[i], lvl)
        self._spaces = tuple(spaces)
        self._ufl_element = hierarchy[0].ufl_element()
        for V in hierarchy:
            dm = V._dm
            dm.setCoarsen(coarsen)
            dm.setRefine(refine)

        # reset the hierarchy with skipped refinements and set full hierarchy
        self._full_hierarchy = hierarchy
        self._hierarchy = hierarchy[::self.refinements_per_level]

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

    def restrict(self, residual, level):
        """
        Restrict a residual (a dual quantity) from level to level-1

        :arg residual: the residual to restrict
        :arg level: the fine level to restrict from
        """
        interface.restrict(residual[level], residual[level-1])

    def prolong(self, solution, level):
        """
        Prolong a solution (a primal quantity) from level - 1 to level

        :arg solution: the solution to prolong
        :arg level: the coarse level to prolong from
        """
        interface.prolong(solution[level], solution[level+1])

    def inject(self, state, level):
        """
        Inject state (a primal quantity) from level to level - 1

        :arg state: the state to inject
        :arg level: the fine level to inject from
        """
        interface.inject(state[level], state[level-1])
