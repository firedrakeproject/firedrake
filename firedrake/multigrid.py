import numpy as np


from pyop2 import op2
import pyop2.coffee.ast_base as ast

import dmplex
import function
import functionspace
import mesh


__all__ = ['MeshHierarchy', 'FunctionSpaceHierarchy', 'FunctionHierarchy',
           'ExtrudedMeshHierarchy']


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

        dm = m._plex
        fpoint_ises = []
        for i in range(refinement_levels):
            rdm = dm.refine()
            fpoint_ises.append(dm.createCoarsePointIS())
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

            if isinstance(m, mesh.IcosahedralSphereMesh):
                coords = rdm.getCoordinatesLocal().array.reshape(-1, 3)
                scale = (m._R / np.linalg.norm(coords, axis=1)).reshape(-1, 1)
                coords *= scale

            dm_hierarchy.append(rdm)
            dm = rdm

        m._init()
        self._hierarchy = [m] + [mesh.Mesh(None, dim=m.ufl_cell().geometric_dimension(),
                                           name="%s_refined_%d" % (m.name, i + 1),
                                           plex=dm, distribute=False, reorder=reorder)
                                 for i, dm in enumerate(dm_hierarchy)]

        self._ufl_cell = m.ufl_cell()
        self._c2f_cells = []
        for m in self:
            m._init()

        for mc, mf, fpointis in zip(self._hierarchy[:-1],
                                    self._hierarchy[1:],
                                    fpoint_ises):
            mc._fpointIS = fpointis
            c2f = dmplex.coarse_to_fine_cells(mc, mf)
            self._c2f_cells.append(c2f)

    def __iter__(self):
        for m in self._hierarchy:
            yield m

    def __len__(self):
        return len(self._hierarchy)

    def __getitem__(self, idx):
        return self._hierarchy[idx]


class ExtrudedMeshHierarchy(MeshHierarchy):
    def __init__(self, mesh_hierarchy, layers, kernel=None, layer_height=None,
                 extrusion_type='uniform', gdim=None):
        self._base_hierarchy = mesh_hierarchy
        self._hierarchy = [mesh.ExtrudedMesh(m, layers, kernel=kernel,
                                             layer_height=layer_height,
                                             extrusion_type=extrusion_type,
                                             gdim=gdim)
                           for m in mesh_hierarchy]
        self._ufl_cell = self[0].ufl_cell()
        self._c2f_cells = mesh_hierarchy._c2f_cells


class FunctionSpaceHierarchy(object):
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
        self._mesh_hierarchy = mesh_hierarchy
        self._hierarchy = [functionspace.FunctionSpace(m, family, degree=degree,
                                                       name=name, vfamily=vfamily,
                                                       vdegree=vdegree)
                           for m in self._mesh_hierarchy]

        self._map_cache = {}
        self._cell_sets = tuple(op2.LocalSet(m.cell_set) for m in self._mesh_hierarchy)
        self._ufl_element = self[0].ufl_element()
        self._restriction_weights = None

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

        element = self.ufl_element()
        family = element.family()
        degree = element.degree()

        c2f = self._mesh_hierarchy._c2f_cells[level]

        if isinstance(self._mesh_hierarchy, ExtrudedMeshHierarchy):
            if not (element._A.family() == "Discontinuous Lagrange" and
                    element._B.family() == "Discontinuous Lagrange" and
                    degree == (0, 0)):
                raise NotImplementedError
            arity = Vf.cell_node_map().arity * c2f.shape[1]
            map_vals = Vf.cell_node_map().values[c2f].flatten()
            offset = np.repeat(Vf.cell_node_map().offset, c2f.shape[1])
            map = op2.Map(self._cell_sets[level],
                          Vf.node_set,
                          arity,
                          map_vals,
                          offset=offset)
            self._map_cache[level] = map
            return map

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
        element = fs.ufl_element()
        family = element.family()
        degree = element.degree()

        if family == "OuterProductElement":
            if not (element._A.family() == "Discontinuous Lagrange" and
                    element._B.family() == "Discontinuous Lagrange" and
                    degree == (0, 0)):
                raise NotImplementedError
            self._prolong_dg0(level)
        elif family == "Discontinuous Lagrange":
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
        element = fs.ufl_element()
        family = element.family()
        degree = element.degree()

        if family == "OuterProductElement":
            if not (element._A.family() == "Discontinuous Lagrange" and
                    element._B.family() == "Discontinuous Lagrange" and
                    degree == (0, 0)):
                raise NotImplementedError
            self._restrict_dg0(level, is_solution=is_solution)
        elif family == "Discontinuous Lagrange":
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
                detJ = 1.0/c2f_map.arity
            else:
                detJ = 1.0
            k = ast.FunDecl("void", "restrict_dg0",
                            [ast.Decl(coarse.dat.ctype, "**coarse"),
                             ast.Decl(fine.dat.ctype, "**fine")],
                            body=ast.Block([ast.Decl(coarse.dat.ctype, "tmp", init=0.0),
                                            ast.c_for("fdof", c2f_map.arity,
                                                      ast.Incr(ast.Symbol("tmp"),
                                                               ast.Symbol("fine", ("fdof", 0))),
                                                      pragma=None),
                                            ast.Assign(ast.Symbol("coarse", (0, 0)),
                                                       ast.Prod(detJ, ast.Symbol("tmp")))]),
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
            # Residual lives in the test space, so we restrict it by
            # writing the coarse basis functions as a linear combination
            # of fine space basis functions.  This turns out to be the
            # transpose of the prolongation.  We have to carry these
            # weights around since we walk over cells and touch some
            # coarse dofs a varying number of times.
            k = """
            static inline void restrict_cg1(double coarse[3], double **weight,
                                            double **fine)
            {
            coarse[0] += weight[0][0]*fine[0][0] + 0.5*(weight[4][0]*fine[4][0] + weight[5][0]*fine[5][0]);
            coarse[1] += weight[1][0]*fine[1][0] + 0.5*(weight[3][0]*fine[3][0] + weight[5][0]*fine[5][0]);
            coarse[2] += weight[2][0]*fine[2][0] + 0.5*(weight[3][0]*fine[3][0] + weight[4][0]*fine[4][0]);
            }
            """
            k = op2.Kernel(k, 'restrict_cg1')

            self._restrict_kernel = k
        fs = self.function_space()
        if fs._restriction_weights is None:
            fs._restriction_weights = FunctionHierarchy(fs)
            k = """
            static inline void weights(double weight[6])
            {
                for ( int i = 0; i < 6; i++ ) {
                    weight[i] += 1.0;
                }
            }"""
            fn = fs._restriction_weights
            k = op2.Kernel(k, 'weights')
            # Count number of times cell loop hits
            for lvl in range(1, len(fn)):
                op2.par_loop(k, self.function_space()._cell_sets[lvl-1],
                             fn[lvl].dat(op2.INC, fn.cell_node_map(lvl-1)[op2.i[0]]))
                # Inverse, since we're using these as weights, not
                # counts.
                fn[lvl].assign(1.0/fn[lvl])

        coarse.dat.zero()
        weights = fs._restriction_weights[level]
        op2.par_loop(self._restrict_kernel, self.function_space()._cell_sets[level-1],
                     coarse.dat(op2.INC, coarse.cell_node_map()[op2.i[0]], flatten=True),
                     weights.dat(op2.READ, c2f_map, flatten=True),
                     fine.dat(op2.READ, c2f_map, flatten=True))
