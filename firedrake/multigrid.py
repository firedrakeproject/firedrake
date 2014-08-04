import numpy as np
import os

import FIAT

from pyop2 import op2
import pyop2.coffee.ast_base as ast

import dmplex
import function
import functionspace
import mesh
import solving
import ufl_expr


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
        dm_hierarchy = []
        parent_cells = []

        dm = m._plex
        for i in range(refinement_levels):
            rdm = dm.refine()
            # Remove interior facet label (re-construct from
            # complement of exterior facets).  Necessary because the
            # refinement just marks points "underneath" the refined
            # facet with the appropriate label.  This works for
            # exterior, but not marked interior facets
            rdm.removeLabel("interior_facets")
            # Remove vertex (and edge) points from labels on exterior
            # facets.  Interior facets will be relabeled in Mesh
            # construction below.
            dmplex.filter_exterior_facet_labels(rdm)
            rdm.removeLabel("op2_core")
            rdm.removeLabel("op2_non_core")
            rdm.removeLabel("op2_exec_halo")
            rdm.removeLabel("op2_non_exec_halo")

            parent_cells.append(dmplex.compute_parent_cells(rdm))
            if isinstance(m, mesh.IcosahedralSphereMesh):
                coords = rdm.getCoordinatesLocal().array.reshape(-1, 3)
                scale = (m._R / np.linalg.norm(coords, axis=1)).reshape(-1, 1)
                coords *= scale

            dm_hierarchy.append(rdm)
            dm = rdm

        self._hierarchy = [m] + [mesh.Mesh(None, dim=m.ufl_cell().geometric_dimension(),
                                           name="%s_refined_%d" % (m.name, i + 1),
                                           plex=dm, distribute=False, reorder=reorder)
                                 for i, dm in enumerate(dm_hierarchy)]

        self._ufl_cell = m.ufl_cell()
        self._c2f_cells = []

        for mc, mf, parents in zip(self._hierarchy[:-1],
                                   self._hierarchy[1:],
                                   parent_cells):
            c2f = dmplex.coarse_to_fine_cells(mc, mf, parents)
            self._c2f_cells.append(c2f)

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
        self._cell_sets = tuple(op2.LocalSet(m.cell_set) for m in self._mesh_hierarchy)
        self._ufl_element = self[0].ufl_element()
        self._lumped_mass = [None for _ in self]

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

        c2f = self._mesh_hierarchy._c2f_cells[level]
        if family == "Discontinuous Lagrange":
            if degree != 0:
                raise RuntimeError
            arity = Vf.cell_node_map().arity * c2f.shape[1]
            map_vals = Vf.cell_node_map().values[c2f].flatten()

            map = op2.Map(self._cell_sets[level],
                          Vf.node_set,
                          arity,
                          map_vals)

            self._map_cache[level] = map
            return map

        if family == "Lagrange":
            if degree != 1:
                raise RuntimeError
            map_vals = dmplex.p1_coarse_fine_map(Vc, Vf, c2f)

            arity = map_vals.shape[1]
            map = op2.Map(self._cell_sets[level], Vf.node_set, arity, map_vals)

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
        elif family == "Lagrange":
            if degree != 1:
                raise RuntimeError("Can only prolong P1 fields, not P%d" % degree)
            self._prolong_cg1(level)
        else:
            raise RuntimeError("Prolongation only implemented for P0DG and P1")

    def restrict(self, level, is_solution=False):
        """Restrict from a fine to the next coarsest hierarchy level.

        :arg level: The fine level to restrict from
        :kwarg is_solution: optional keyword argument indicating if
            the :class:`~.Function` being restricted is a *solution*,
            living in the primal space or a *residual* (cofunction)
            living in the dual space (the default).  Residual
            restriction is weighted by the size of the coarse cell
            relative to the fine cells (i.e. the mass matrix) whereas
            solution restriction need not be weighted."""

        if not 0 < level < len(self):
            raise RuntimeError("Requested fine level %d outside permissible range [1, %d)" %
                               (level, len(self)))

        fs = self[level].function_space()
        family = fs.ufl_element().family()
        degree = fs.ufl_element().degree()
        if family == "Discontinuous Lagrange":
            if degree == 0:
                self._restrict_dg0(level, is_solution=is_solution)
            else:
                raise RuntimeError("Can only restrict P0 fields, not P%dDG" % degree)
        elif family == "Lagrange":
            if degree != 1:
                raise RuntimeError("Can only restrict P1 fields, not P%d" % degree)
            self._restrict_cg1(level, is_solution=is_solution)
        else:
            raise RuntimeError("Restriction only implemented for P0DG and P1")

    def inject(self, level):
        """Inject from a fine to the next coarsest hierarchy level.

        :arg level: the fine level to inject from"""
        if not 0 < level < len(self):
            raise RuntimeError

        self._inject_cg1(level)

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
        op2.par_loop(self._prolong_kernel, self.function_space()._cell_sets[level],
                     coarse.dat(op2.READ, coarse.cell_node_map()),
                     fine.dat(op2.WRITE, c2f_map))

    def _restrict_dg0(self, level, is_solution=False):
        c2f_map = self.cell_node_map(level - 1)
        coarse = self[level - 1]
        fine = self[level]
        if not hasattr(self, '_restrict_kernel'):
            if is_solution:
                detJ = 1.0
            else:
                # we need to weight the restricted residual by detJ of the
                # big cell relative to the small cells.  Since we have
                # regular refinement, this is just 2^tdim
                detJ = 2.0 ** self.function_space().ufl_element().cell().topological_dimension()
            k = ast.FunDecl("void", "restrict_dg0",
                            [ast.Decl(coarse.dat.ctype, "**coarse"),
                             ast.Decl(fine.dat.ctype, "**fine")],
                            body=ast.Block([ast.Decl(coarse.dat.ctype, "tmp", init=0.0),
                                            ast.c_for("fdof", c2f_map.arity,
                                                      ast.Incr(ast.Symbol("tmp"),
                                                               ast.Symbol("fine", ("fdof", 0))),
                                                      pragma=None),
                                            # Need to multiply restricted function by a factor 4
                                            ast.Assign(ast.Symbol("coarse", (0, 0)),
                                                       ast.Prod(detJ/c2f_map.arity, ast.Symbol("tmp")))]),
                            pred=["static", "inline"])
            self._restrict_kernel = op2.Kernel(k, "restrict_dg0")

        op2.par_loop(self._restrict_kernel, self.function_space()._cell_sets[level-1],
                     coarse.dat(op2.WRITE, coarse.cell_node_map()),
                     fine.dat(op2.READ, c2f_map))

    def _prolong_cg1(self, level):
        c2f_map = self.cell_node_map(level)
        coarse = self[level]
        fine = self[level + 1]
        if not hasattr(self, '_prolong_kernel'):
            # Only 2D for now.
            # Due to smart map, fine field is:
            # u_f = A u_c
            # Where A is [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
            k = """void prolong_cg1(double **coarse, double **fine)
            {
                fine[0][0] = coarse[0][0];
                fine[1][0] = coarse[1][0];
                fine[2][0] = coarse[2][0];
                fine[3][0] = 0.5*(coarse[1][0] + coarse[2][0]);
                fine[4][0] = 0.5*(coarse[0][0] + coarse[2][0]);
                fine[5][0] = 0.5*(coarse[0][0] + coarse[1][0]);
            }"""
            self._prolong_kernel = op2.Kernel(k, "prolong_cg1")
        op2.par_loop(self._prolong_kernel, self.function_space()._cell_sets[level],
                     coarse.dat(op2.READ, coarse.cell_node_map()),
                     fine.dat(op2.WRITE, c2f_map))

    def _inject_cg1(self, level):
        c2f_map = self.cell_node_map(level - 1)
        coarse = self[level - 1]
        fine = self[level]
        if not hasattr(self, '_inject_kernel'):
            k = """void inject_cg1(double **coarse, double **fine)
            {
                for ( int i = 0; i < 3; i++ ) {
                    coarse[i][0] = fine[i][0];
                }
            }"""
            self._inject_kernel = op2.Kernel(k, "inject_cg1")
        op2.par_loop(self._inject_kernel, self.function_space()._cell_sets[level-1],
                     coarse.dat(op2.WRITE, coarse.cell_node_map()),
                     fine.dat(op2.READ, c2f_map))

    def _restrict_cg1(self, level, is_solution=False):
        c2f_map = self.cell_node_map(level - 1)
        coarse = self[level - 1]
        fine = self[level]
        if not hasattr(self, '_restrict_kernel'):
            element = coarse.function_space().fiat_element
            quadrature = FIAT.make_quadrature(element.ref_el, 2)
            weights = quadrature.get_weights()
            points = quadrature.get_points()

            fine_basis = element.tabulate(0, points).values()[0]

            # Fine cells numbered as P2 coarse cell:
            #
            # 2.
            # | \
            # |  \
            # | 2 \
            # 4----3.
            # | \ 3| \
            # |  \ |  \
            # | 0 \| 1 \
            # 0----5----1
            #
            # The transformations from coordinates in each fine
            # reference cell to coordinates in the coarse reference
            # cell are:
            #
            # T0: dofs 0 5 4
            #     X_c = 1/2 (1 0) X_f
            #               (0 1)
            #
            # T1: dofs 1 3 5
            #     X_c = 1/2 (-1 -1) X_f + (1)
            #               ( 1  0)       (0)
            #
            # T2: dofs 2 4 3
            #     X_c = 1/2 ( 0  1) X_f + (0)
            #               (-1 -1)       (1)
            #
            # T3: dofs 3 5 4
            #     X_c = 1/2 ( 0 -1) X_f + (1/2)
            #               (-1  0)       (1/2)
            #
            # Hence, to calculate the coarse basis functions, we take
            # the reference cell fine quadrature points and hit them
            # with the transformation, before tabulating.
            X_c0 = 0.5 * points
            X_c1 = np.asarray([0.5 * np.dot([[-1, -1], [1, 0]], pt) + np.array([1, 0], dtype=float) for pt in points])
            X_c2 = np.asarray([0.5 * np.dot([[0, 1], [-1, -1]], pt) + np.array([0, 1], dtype=float) for pt in points])
            X_c3 = np.asarray([0.5 * (np.dot([[0, -1], [-1, 0]], pt) + np.array([1, 1], dtype=float)) for pt in points])

            coarse_basis = element.tabulate(0, np.concatenate([X_c0, X_c1, X_c2, X_c3])).values()[0]

            k = """
            #include "firedrake_geometry.h"
            #include <stdio.h>
            static inline void restrict_cg1(double coarse[3], double **fine, double **coordinates)
            {
            const double fine_basis[4][3] = %(fine_basis)s;
            const double coarse_basis[16][3] = %(coarse_basis)s;
            const double weight[4] = %(weight)s;
            const int fine_cell_dofs[4][3] = {{0, 5, 4}, {1, 3, 5}, {2, 4, 3}, {3, 5, 4}};
            double J[4];
            compute_jacobian_triangle_2d(J, coordinates);
            double K[4];
            double detJ;
            compute_jacobian_inverse_triangle_2d(K, detJ, J);
            const double det = fabs(detJ);
            for ( int fcell = 0; fcell < 4; fcell++ ) {
                for ( int ip = 0; ip < 4; ip++ ) {
                    double fine_coeff = 0;
                    for ( int i = 0; i < 3; i++ ) {
                        fine_coeff += fine[fine_cell_dofs[fcell][i]][0] * fine_basis[ip][i];
                    }
                    for ( int i = 0; i < 3; i++ ) {
                        coarse[i] += (weight[ip] * fine_coeff * coarse_basis[fcell * 4 + ip][i]) * det/4.0;
                    }
                }
            }
            }
            """ % {"fine_basis": "{{" + "},\n{".join([", ".join(map(str, x)) for x in fine_basis.T])+"}}",
                   "coarse_basis": "{{" + "},\n{".join([", ".join(map(str, x)) for x in coarse_basis.T])+"}}",
                   "weight": "{" + ", ".join(["%s" % w for w in weights]) + "}"}

            k = op2.Kernel(k, 'restrict_cg1', include_dirs=[os.path.dirname(__file__)])

            self._restrict_kernel = k
        ccoords = coarse.function_space().mesh().coordinates

        coarse.dat.zero()
        op2.par_loop(self._restrict_kernel, self.function_space()._cell_sets[level-1],
                     coarse.dat(op2.INC, coarse.cell_node_map()[op2.i[0]], flatten=True),
                     fine.dat(op2.READ, c2f_map, flatten=True),
                     ccoords.dat(op2.READ, ccoords.cell_node_map(), flatten=True))

        fs = self.function_space()
        if is_solution:
            detJ = 1.0
        else:
            # we need to weight the restricted residual by detJ of the
            # big cell relative to the small cells.  Since we have
            # regular refinement, this is just 2^tdim
            detJ = 2.0 ** self.function_space().ufl_element().cell().topological_dimension()
        if fs._lumped_mass[level - 1] is None:
            v = ufl_expr.TestFunction(fs[level - 1])
            fs._lumped_mass[level - 1] = solving.assemble(v*v.function_space().mesh()._dx)
            fs._lumped_mass[level - 1].assign(detJ / fs._lumped_mass[level - 1])

        coarse *= fs._lumped_mass[level - 1]
