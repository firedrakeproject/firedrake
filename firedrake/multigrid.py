import numpy as np

from pyop2 import op2

import function
import functionspace
import mesh


__all__ = ['MeshHierarchy', 'FunctionSpaceHierarchy', 'FunctionHierarchy']


class MeshHierarchy(mesh.Mesh):
    """Build a hierarchy of meshes by uniformly refining a coarse mesh"""
    def __init__(self, m, refinement_levels):
        """
        :arg m: the coarse :class:`~.Mesh` to refine
        :arg refinement_levels: the number of levels of refinement
        """
        m._plex.setRefinementUniform(True)
        dm_hierarchy = m._plex.refineHierarchy(refinement_levels)
        for dm in dm_hierarchy:
            dm.removeLabel("boundary_faces")
            dm.markBoundaryFaces("boundary_faces")
            dm.removeLabel("exterior_facets")
            dm.removeLabel("interior_facets")
            dm.removeLabel("op2_core")
            dm.removeLabel("op2_non_core")
            dm.removeLabel("op2_exec_halo")
            if isinstance(m, mesh.IcosahedralSphereMesh):
                coords = dm.getCoordinatesLocal().array.reshape(-1, 3)
                scale = (m._R / np.linalg.norm(coords, axis=1)).reshape(-1, 1)
                coords *= scale

        self._hierarchy = [m] + [mesh.Mesh(None, name="%s_refined_%d" % (m.name, i + 1),
                                           plex=dm, distribute=False)
                                 for i, dm in enumerate(dm_hierarchy)]

        self._ufl_cell = m.ufl_cell()
        # Simplex only
        factor = 2 ** self.ufl_cell().topological_dimension()
        self._c2f_cells = []
        for mc, mf in zip(self._hierarchy[:-1], self._hierarchy[1:]):
            cback = mc._inv_cells
            fforward = mf._cells
            ofcells = np.dstack([(cback * factor) + i for i in range(factor)]).flatten()
            fcells = fforward[ofcells]
            self._c2f_cells.append(fcells.reshape(-1, factor))

    def __iter__(self):
        for m in self._hierarchy:
            yield m

    def __len__(self):
        return len(self._hierarchy)

    def __getitem__(self, idx):
        return self._hierarchy[idx]


class FunctionSpaceHierarchy(object):
    """Build a hierarchy of function spaces.

    Given a hierarchy of meshes, this constructs a hierarchy of
    function spaces, with the property that every coarse space is a
    subspace of the fine spaces that are a refinement of it.
    """
    def __init__(self, mesh_hierarchy, family, degree):
        """
        :arg mesh_hierarchy: a :class:`.MeshHierarchy` to build the
             function spaces on.
        :arg family: the function space family
        :arg degree: the degree of the function space
        """
        self._mesh_hierarchy = mesh_hierarchy
        self._hierarchy = [functionspace.FunctionSpace(m, family, degree)
                           for m in self._mesh_hierarchy]

        self._map_cache = {}
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

    def cell_node_map(self, level, bcs=None):
        """A :class:`pyop2.Map` from cells on a coarse mesh to the
        corresponding degrees of freedom on a the fine mesh below it.

        :arg level: the coarse level the map should be from.
        :arg bcs: optional iterable of :class:`.DirichletBC`\s
             (currently ignored).
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

        c2f = self._mesh_hierarchy._c2f_cells[level]
        arity = Vf.cell_node_map().arity * c2f.shape[1]
        map_vals = Vf.cell_node_map().values_with_halo[c2f].flatten()

        map = op2.Map(Vc.mesh().cell_set,
                      Vf.node_set,
                      arity,
                      map_vals)

        self._map_cache[level] = map
        return map


class FunctionHierarchy(object):
    """Build a hierarchy of :class:`~.Function`\s"""
    def __init__(self, fs_hierarchy):
        """
        :arg fs_hierarchy: the :class:`~.FunctionSpaceHierarchy` to build on.

        `fs_hierarchy` may also be an existing
        :class:`FunctionHierarchy`, in which case a copy of the
        hierarchy is returned.
        """
        if isinstance(fs_hierarchy, FunctionHierarchy):
            self._function_space = fs_hierarchy.function_space()
        else:
            self._function_space = fs_hierarchy

        self._hierarchy = [function.Function(f) for f in fs_hierarchy]

    def __iter__(self):
        for f in self._hierarchy:
            yield f

    def __len__(self):
        return len(self._hierarchy)

    def __getitem__(self, idx):
        return self._hierarchy[idx]

    def function_space(self):
        return self._function_space

    def cell_node_map(self, i):
        return self._function_space.cell_node_map(i)
