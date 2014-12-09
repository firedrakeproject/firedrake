from __future__ import absolute_import

import numpy as np
import ufl

from pyop2 import op2
from pyop2.utils import flatten

from firedrake import functionspace
from . import impl
from . import mesh
from . import utils
import firedrake.mg.function


__all__ = ["FunctionSpaceHierarchy", "VectorFunctionSpaceHierarchy",
           "MixedFunctionSpaceHierarchy"]


class BaseHierarchy(object):

    def __init__(self, mesh_hierarchy, fses):
        """
        Build a hierarchy of function spaces

        :arg mesh_hierarchy: a :class:`~.MeshHierarchy` on which to
             build the function spaces.
        :arg fses: an iterable of :class:`~.FunctionSpace`\s.
        """
        self._mesh_hierarchy = mesh_hierarchy
        self._hierarchy = tuple(fses)
        self._map_cache = {}
        self._cell_sets = tuple(op2.LocalSet(m.cell_set) for m in self._mesh_hierarchy)
        self._ufl_element = self[0].ufl_element()
        self._restriction_weights = None
        element = self.ufl_element()
        family = element.family()
        degree = element.degree()
        self._P0 = ((family == "OuterProductElement" and
                     (element._A.family() == "Discontinuous Lagrange" and
                      element._B.family() == "Discontinuous Lagrange" and
                      degree == (0, 0))) or
                    (family == "Discontinuous Lagrange" and degree == 0))
        if self._P0:
            self._prolong_kernel = op2.Kernel("""
                void prolongation(double fine[%d], double **coarse)
                {
                    for ( int k = 0; k < %d; k++ ) {
                        for ( int i = 0; i < %d; i++ ) {
                            fine[i*%d + k] = coarse[0][k];
                        }
                    }
                }""" % (self.cell_node_map(0).arity*self.dim,
                        self.dim, self.cell_node_map(0).arity,
                        self.dim), "prolongation")
            self._restrict_kernel = op2.Kernel("""
                void restriction(double coarse[%d], double **fine)
                {
                    for ( int k = 0; k < %d; k++ ) {
                        for ( int i = 0; i < %d; i++ ) {
                            coarse[k] += fine[i][k];
                        }
                    }
                }""" % (self.dim, self.dim, self.cell_node_map(0).arity), "restriction")
            self._inject_kernel = op2.Kernel("""
                void injection(double coarse[%d], double **fine)
                {
                    for ( int k = 0; k < %d; k++ ) {
                        for ( int i = 0; i < %d; i++ ) {
                            coarse[k] += fine[i][k];
                        }
                        coarse[k] *= %g;
                    }
                }""" % (self.dim, self.dim, self.cell_node_map(0).arity,
                        1.0/self.cell_node_map(0).arity), "injection")
        else:
            try:
                element = self[0].fiat_element
                omap = self[1].cell_node_map().values
                c2f, vperm = self._mesh_hierarchy._cells_vperm[0]
                indices, _ = utils.get_unique_indices(element,
                                                      omap[c2f[0, :], ...].reshape(-1),
                                                      vperm[0, :],
                                                      offset=None)
                self._prolong_kernel = utils.get_prolongation_kernel(element, indices, self.dim)
                self._restrict_kernel = utils.get_restriction_kernel(element, indices, self.dim)
                self._inject_kernel = utils.get_injection_kernel(element, indices, self.dim)
            except:
                pass

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

    def restrict(self, residual, level):
        """
        Restrict a residual (a dual quantity) from level to level-1

        :arg residual: the residual to restrict
        :arg level: the fine level to restrict from
        """
        if not 0 < level < len(self):
            raise RuntimeError("Requested fine level %d outside permissible range [1, %d)" %
                               (level, len(self)))

        # We hit each fine dof more than once since we loop
        # elementwise over the coarse cells.  So we need a count of
        # how many times we did this to weight the final contribution
        # appropriately.
        if not self._P0 and self._restriction_weights is None:
            if isinstance(self.ufl_element(), (ufl.VectorElement,
                                               ufl.OuterProductVectorElement)):
                element = self.ufl_element().sub_elements()[0]
                restriction_fs = FunctionSpaceHierarchy(self._mesh_hierarchy, element)
            else:
                restriction_fs = self
            self._restriction_weights = firedrake.mg.function.FunctionHierarchy(restriction_fs)

            k = """
            static inline void weights(double weight[%(d)d])
            {
                for ( int i = 0; i < %(d)d; i++ ) {
                    weight[i] += 1.0;
                }
            }""" % {'d': self.cell_node_map(0).arity}
            k = op2.Kernel(k, "weights")
            weights = self._restriction_weights
            # Count number of times each fine dof is hit
            for lvl in range(1, len(weights)):
                op2.par_loop(k, self._cell_sets[lvl-1],
                             weights[lvl].dat(op2.INC, weights.cell_node_map(lvl-1)[op2.i[0]]))
                # Inverse, since we're using as weights not counts
                weights[lvl].assign(1.0/weights[lvl])

        coarse = residual[level-1]
        fine = residual[level]

        args = [coarse.dat(op2.INC, coarse.cell_node_map()[op2.i[0]]),
                fine.dat(op2.READ, self.cell_node_map(level-1))]

        if not self._P0:
            weights = self._restriction_weights[level]
            args.append(weights.dat(op2.READ, self._restriction_weights.cell_node_map(level-1)))
        coarse.dat.zero()
        op2.par_loop(self._restrict_kernel, self._cell_sets[level-1],
                     *args)

    def inject(self, state, level):
        """
        Inject state (a primal quantity) from level to level - 1

        :arg state: the state to inject
        :arg level: the fine level to inject from
        """
        if not 0 < level < len(self):
            raise RuntimeError("Requested fine level %d outside permissible range [1, %d)" %
                               (level, len(self)))

        coarse = state[level-1]
        fine = state[level]
        op2.par_loop(self._inject_kernel, self._cell_sets[level-1],
                     coarse.dat(op2.WRITE, coarse.cell_node_map()[op2.i[0]]),
                     fine.dat(op2.READ, self.cell_node_map(level-1)))

    def prolong(self, solution, level):
        """
        Prolong a solution (a primal quantity) from level - 1 to level

        :arg solution: the solution to prolong
        :arg level: the coarse level to prolong from
        """
        if not 0 <= level < len(self) - 1:
            raise RuntimeError("Requested coarse level %d outside permissible range [0, %d)" %
                               (level, len(self) - 1))
        coarse = solution[level]
        fine = solution[level+1]
        op2.par_loop(self._prolong_kernel, self._cell_sets[level],
                     fine.dat(op2.WRITE, self.cell_node_map(level)[op2.i[0]]),
                     coarse.dat(op2.READ, coarse.cell_node_map()))


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
                for m in mesh_hierarchy]
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
                for m in mesh_hierarchy]
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

    def restrict(self, residual, level):
        """
        Restrict a residual (a dual quantity) from level to level-1

        :arg residual: the residual to restrict
        :arg level: the fine level to restrict from
        """
        for res, fs in zip(residual.split(), self.split()):
            fs.restrict(res, level)

    def prolong(self, solution, level):
        """
        Prolong a solution (a primal quantity) from level - 1 to level

        :arg solution: the solution to prolong
        :arg level: the coarse level to prolong from
        """
        for sol, fs in zip(solution.split(), self.split()):
            fs.prolong(sol, level)

    def inject(self, state, level):
        """
        Inject state (a primal quantity) from level to level - 1

        :arg state: the state to inject
        :arg level: the fine level to inject from
        """
        for st, fs in zip(state.split(), self.split()):
            fs.inject(st, level)
