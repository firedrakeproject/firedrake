import numpy as np

from pyop2 import op2
import pyop2.coffee.ast_base as ast

import dmplex
import function
import functionspace
import mesh


__all__ = ['MeshHierarchy', 'FunctionSpaceHierarchy', 'FunctionHierarchy']


class MeshHierarchy(mesh.Mesh):
    """Build a hierarchy of meshes by uniformly refining a coarse mesh"""
    def __init__(self, m, refinement_levels, reorder=None):
        """
        :arg m: the coarse :class:`~.Mesh` to refine
        :arg refinement_levels: the number of levels of refinement
        :arg reorder: optional flag indicating whether to reorder the
             refined meshes.
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

        self._hierarchy = [m] + [mesh.Mesh(None, dim=m.ufl_cell().geometric_dimension(),
                                           name="%s_refined_%d" % (m.name, i + 1),
                                           plex=dm, distribute=False, reorder=reorder)
                                 for i, dm in enumerate(dm_hierarchy)]

        self._ufl_cell = m.ufl_cell()
        # Simplex only
        factor = 2 ** self.ufl_cell().topological_dimension()
        self._c2f_cells = []

        for mc, mf in zip(self._hierarchy[:-1], self._hierarchy[1:]):
            if not hasattr(mc, '_cell_new_to_old'):
                o, n = dmplex.get_cell_renumbering(mc._plex, mc._cell_numbering)
                mc._cell_old_to_new = o
                mc._cell_new_to_old = n
            if not hasattr(mf, '_cell_old_to_new'):
                o, n = dmplex.get_cell_renumbering(mf._plex, mf._cell_numbering)
                mf._cell_old_to_new = o
                mf._cell_new_to_old = n

            cback = mc._cell_new_to_old
            fforward = mf._cell_old_to_new
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

        family = self.ufl_element().family()
        degree = self.ufl_element().degree()

        if family == "Discontinuous Lagrange":
            if degree != 0:
                raise RuntimeError
            c2f = self._mesh_hierarchy._c2f_cells[level]
            arity = Vf.cell_node_map().arity * c2f.shape[1]
            map_vals = Vf.cell_node_map().values_with_halo[c2f].flatten()

            map = op2.Map(Vc.mesh().cell_set,
                          Vf.node_set,
                          arity,
                          map_vals)

            self._map_cache[level] = map
            return map

        if family == "Lagrange":
            if degree != 1:
                raise RuntimeError
            cm = Vc.mesh()
            fm = Vf.mesh()
            c2f_is = cm._plex.createCoarsePointIS().indices
            c2f = self._mesh_hierarchy._c2f_cells[level]
            if cm.ufl_cell().cellname() == 'interval':
                ndof = 3
            elif cm.ufl_cell().cellname() == 'triangle':
                ndof = 6
            elif cm.ufl_cell().cellname() == 'tetrahedron':
                ndof = 10
            else:
                raise RuntimeError("Don't know how to make map")
            map_vals = np.empty((c2f.shape[0], ndof), dtype=np.int32)

            vStart, vEnd = cm._plex.getDepthStratum(0)

            inv = np.empty(cm.num_vertices(), dtype=np.int32)
            for i in range(*cm._plex.getChart()):
                if cm._vertex_numbering.getDof(i) > 0:
                    inv[cm._vertex_numbering.getOffset(i)] = i - vStart
            for c in range(cm.num_cells()):
                got = []
                for i, cv in enumerate(Vc.cell_node_map().values_with_halo[c]):
                    orig_v = inv[cv]
                    fv = fm._vertex_numbering.getOffset(c2f_is[orig_v + vStart])
                    map_vals[c, i] = fv
                    got.append(fv)
                non_unique = Vf.cell_node_map().values_with_halo[c2f[c, :]]
                unique = set(non_unique.flatten())
                want = unique.difference(set(got))
                i = 3
                for g in got:
                    idx = np.where(g == non_unique)
                    map_vals[c, i] = want.difference(non_unique[idx[0][0], :]).pop()
                    i += 1
            map = op2.Map(Vc.mesh().cell_set,
                          Vf.node_set,
                          ndof,
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

    def prolong(self, level):
        """Prolong from a coarse to the next finest hierarchy level.

        :arg level: The coarse level to prolong from"""

        if not 0 <= level < len(self) - 1:
            raise RuntimeError("Requested coarse level %d outside permissible range [0, %d)" %
                               (level, len(self) - 1))
        fs = self[level].function_space()
        family = fs.ufl_element().family()
        degree = fs.ufl_element().degree()

        if family == "Discontinuous Lagrange":
            if degree != 0:
                raise RuntimeError("Can only prolong P0 fields, not P%dDG" % degree)
            self._prolong_dg0(level)

    def restrict(self, level):
        """Restrict from a fine to the next coarsest hierarchy level.

        :arg level: The fine level to restrict from"""

        if not 0 < level < len(self):
            raise RuntimeError("Requested fine level %d outside permissible range [1, %d)" %
                               (level, len(self)))

        fs = self[level].function_space()
        family = fs.ufl_element().family()
        degree = fs.ufl_element().degree()
        if family == "Discontinuous Lagrange":
            if degree == 0:
                self._restrict_dg0(level)
            else:
                raise RuntimeError("Can only restrict P0 fields, not P%dDG" % degree)

    def _prolong_dg0(self, level):
        c2f_map = self.cell_node_map(level)
        coarse = self[level]
        fine = self[level + 1]
        if not hasattr(self, '_prolong_kernel'):
            k = ast.FunDecl("void", "prolong_dg0",
                            [ast.Decl(coarse.dat.ctype, "**coarse"),
                             ast.Decl(fine.dat.ctype, "**fine")],
                            body=ast.c_for("fdof", c2f_map.arity,
                                           ast.Assign(ast.Symbol("fine", ("fdof", 0)),
                                                      ast.Symbol("coarse", (0, 0))),
                                           pragma=None),
                            pred=["static", "inline"])
            self._prolong_kernel = op2.Kernel(k, "prolong_dg0")
        op2.par_loop(self._prolong_kernel, coarse.cell_set,
                     coarse.dat(op2.READ, coarse.cell_node_map()),
                     fine.dat(op2.WRITE, c2f_map))

    def _restrict_dg0(self, level):
        c2f_map = self.cell_node_map(level - 1)
        coarse = self[level - 1]
        fine = self[level]
        if not hasattr(self, '_restrict_kernel'):
            k = ast.FunDecl("void", "restrict_dg0",
                            [ast.Decl(coarse.dat.ctype, "**coarse"),
                             ast.Decl(fine.dat.ctype, "**fine")],
                            body=ast.Block([ast.Decl(coarse.dat.ctype, "tmp", init=0.0),
                                            ast.c_for("fdof", c2f_map.arity,
                                                      ast.Incr(ast.Symbol("tmp"),
                                                               ast.Symbol("fine", ("fdof", 0))),
                                                      pragma=None),
                                            ast.Assign(ast.Symbol("coarse", (0, 0)),
                                                       ast.Div(ast.Symbol("tmp"),
                                                               c2f_map.arity))]),
                            pred=["static", "inline"])
            self._restrict_kernel = op2.Kernel(k, "restrict_dg0")

        op2.par_loop(self._restrict_kernel, coarse.cell_set,
                     coarse.dat(op2.WRITE, coarse.cell_node_map()),
                     fine.dat(op2.READ, c2f_map))
