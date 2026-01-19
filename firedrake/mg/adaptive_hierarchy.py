"""
This module contains the class for the AdaptiveMeshHierarchy and
related helper functions
"""

from fractions import Fraction
from collections import defaultdict
import numpy as np

from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import Mesh, Submesh, RelabeledMesh
from firedrake.mg import HierarchyBase
from firedrake.mg.utils import set_level, get_level
from firedrake.petsc import PETSc

__all__ = ["AdaptiveMeshHierarchy"]


class AdaptiveMeshHierarchy(HierarchyBase):
    """
    HierarchyBase for hierarchies of adaptively refined meshes
    """
    def __init__(self, base_mesh, refinements_per_level=1, nested=True):
        self.meshes = (base_mesh,)
        self._meshes = (base_mesh,)
        self.submesh_hierarchies = []
        self.coarse_to_fine_cells = {}
        self.fine_to_coarse_cells = {}
        self.fine_to_coarse_cells[Fraction(0, 1)] = None
        self.refinements_per_level = refinements_per_level
        self.nested = nested
        set_level(base_mesh, self, 0)
        self.split_cache = {}

    def add_mesh(self, mesh):
        """
        Adds newly refined mesh into hierarchy.
        Then computes the coarse_to_fine and fine_to_coarse mappings.
        Constructs intermediate submesh hierarchies with this.
        """
        max_children = 2 ** mesh.topological_dimension
        coarse_mesh = self.meshes[-1]
        self._meshes += (mesh,)
        self.meshes += (mesh,)
        level = len(self.meshes)
        set_level(self.meshes[-1], self, level - 1)
        self._shared_data_cache = defaultdict(dict)

        # extract parent child relationships from netgen meshes
        (c2f, f2c) = get_c2f_f2c_fd(mesh, coarse_mesh)
        c2f_global_key = Fraction(len(self.meshes) - 2, 1)
        f2c_global_key = Fraction(len(self.meshes) - 1, 1)
        self.coarse_to_fine_cells[c2f_global_key] = c2f
        self.fine_to_coarse_cells[f2c_global_key] = np.array(f2c)

        # split both the fine and coarse meshes into the submeshes
        (coarse_splits, fine_splits, num_children) = split_to_submesh(
            mesh, coarse_mesh, c2f, f2c
        )
        for i in range(1, max_children + 1):
            coarse_mesh.mark_entities(coarse_splits[i], i)
            mesh.mark_entities(fine_splits[i], int(f"10{i}"))

        coarse_labels = list(range(1, max_children + 1))
        coarse_indicators = [coarse_splits[i] for i in coarse_labels]

        has_children = {}
        for label in coarse_labels:
            with coarse_splits[label].dat.vec as v:
                has_children[label] = v.norm() > 1E-12

        coarse_mesh = RelabeledMesh(
            coarse_mesh,
            coarse_indicators,
            coarse_labels,
            name="Relabeled_coarse",
        )
        c_subm = {
            j: Submesh(coarse_mesh, coarse_mesh.topological_dimension, j)
            for j in range(1, max_children + 1)
            if has_children[j]
        }
        set_level(coarse_mesh, self, level - 2)

        fine_labels = list(range(1, max_children + 1))
        fine_indicators = [fine_splits[i] for i in fine_labels]
        mesh = RelabeledMesh(
            mesh,
            fine_indicators,
            fine_labels,
            name="Relabeled_fine",
        )
        f_subm = {
            int(str(j)[-2:]): Submesh(mesh, mesh.topological_dimension, j)
            for j in [int("10" + str(i)) for i in range(1, max_children + 1)]
            if has_children[int(str(j)[-2:])]
        }
        set_level(mesh, self, level - 1)

        # stores number of parents for each amount of children
        parents_per_child_count = [
            len([el for el in c2f if len(el) == j])
            for j in range(1, max_children + 1)
        ]

        # update c2f and f2c for submeshes by mapping numberings
        # on full mesh to numberings on coarse mesh
        c2f_adjusted = {
            j: np.zeros((num_parents, j))
            for j, num_parents in enumerate(parents_per_child_count, 1)
            if num_parents != 0
        }
        f2c_adjusted = {
            j: np.zeros((num_parents * j, 1))
            for j, num_parents in enumerate(parents_per_child_count, 1)
            if num_parents != 0
        }

        coarse_full_to_sub_map = {
            i: get_full_to_sub_numbering(coarse_mesh, c_subm[i])
            for i in c_subm
        }
        fine_full_to_sub_map = {
            j: get_full_to_sub_numbering(mesh, f_subm[j])
            for j in f_subm
        }

        for i, children in enumerate(c2f):
            n = len(children)
            if n in coarse_full_to_sub_map:
                coarse_id_sub = coarse_full_to_sub_map[n][i]
                fine_id_sub = fine_full_to_sub_map[n][np.array(children)]
                c2f_adjusted[n][coarse_id_sub] = fine_id_sub

        for j, parent in enumerate(f2c):
            n = num_children[parent].item()
            if n in coarse_full_to_sub_map:
                coarse_id_sub = coarse_full_to_sub_map[n][parent.item()]
                fine_id_sub = fine_full_to_sub_map[n][j]
                f2c_adjusted[n][fine_id_sub, 0] = coarse_id_sub

        c2f_subm = {
            i: {Fraction(0, 1): c2f_adjusted[i].astype(PETSc.IntType)}
            for i in c2f_adjusted
        }
        f2c_subm = {i: {Fraction(1, 1): f2c_adjusted[i]} for i in f2c_adjusted}

        hierarchy_dict = {
            i: HierarchyBase(
                [c_subm[i], f_subm[i]], c2f_subm[i], f2c_subm[i], nested=True
            )
            for i in c_subm
        }
        self.submesh_hierarchies.append(hierarchy_dict)

    def refine(self, refinements):
        """
        Refines and adds mesh if input a boolean vector corresponding to cells
        """
        ngmesh = self.meshes[-1].netgen_mesh
        for i, el in enumerate(ngmesh.Elements2D()):
            el.refine = refinements[i]

        ngmesh.Refine(adaptive=True)
        mesh = Mesh(ngmesh)
        self.add_mesh(mesh)

    def adapt(self, eta: Function | Cofunction, theta: float):
        """
        Add a refinement level to the hierarchy by local refinement
        with a simplified variant of Dorfler marking.

        Parameters
        ----------
        eta
            A DG0 :class:`~firedrake.function.Function` with the local error estimator.
        theta
            The threshold for marking as a fraction of the maximum error.

        Note
        ----
        Dorfler marking involves sorting all of the elements by decreasing
        error estimator and taking the minimal set that exceeds some fixed
        fraction of the total error. What this code implements is the simpler
        variant that doesn't have a proof of convergence (as far as I know)
        but works as well in practice.

        """
        if not isinstance(eta, (Function, Cofunction)):
            raise TypeError(f"eta must be a Function or Cofunction, not a {type(eta).__name__}")
        M = eta.function_space()
        if M.finat_element.space_dimension() != 1:
            raise ValueError("eta must be a Function or Cofunction in DG0")
        mesh = self.meshes[-1]
        if M.mesh() is not mesh:
            raise ValueError("eta must be defined on the finest mesh of the hierarchy")

        # Take the maximum over all processes
        with eta.dat.vec_ro as eta_:
            eta_max = eta_.max()[1]

        threshold = theta * eta_max
        should_refine = eta.dat.data_ro > threshold

        markers = Function(M)
        markers.dat.data_wo[should_refine] = 1

        refined_mesh = mesh.refine_marked_elements(markers)
        self.add_mesh(refined_mesh)
        return refined_mesh

    def split_function(self, u, child=True):
        """
        Split input function across submeshes
        """
        V = u.function_space()
        full_mesh = V.mesh()
        _, level = get_level(full_mesh)

        ind = 1 if child else 0
        hierarchy_dict = self.submesh_hierarchies[int(level) - ind]
        parent_mesh = tuple(hierarchy_dict.values())[0].meshes[ind].submesh_parent
        parent_space = V.reconstruct(parent_mesh)
        u_corr_space = Function(parent_space, val=u.dat)
        key = (u, child)
        try:
            split_functions = self.split_cache[key]
        except KeyError:
            split_functions = self.split_cache.setdefault(key, {})

        for i in hierarchy_dict:
            try:
                f = split_functions[i].zero()
            except KeyError:
                V_split = V.reconstruct(mesh=hierarchy_dict[i].meshes[ind])
                assert (
                    V_split.mesh().submesh_parent
                    == u_corr_space.function_space().mesh()
                )
                f = split_functions.setdefault(
                    i,
                    Function(V_split, name=str(i))
                )

            f.assign(u_corr_space)
        return split_functions

    def use_weight(self, V, child):
        """
        Counts DoFs across submeshes, computes partition of unity
        """
        w = Function(V).assign(1)
        splits = self.split_function(w, child)

        self.recombine(splits, w, child)
        with w.dat.vec as wvec:
            wvec.reciprocal()
        return w

    def recombine(self, split_funcs, f, child=True):
        """
        Recombines functions on submeshes back to the parent mesh
        """
        V = f.function_space()
        f.zero()
        parent_mesh = (
            split_funcs[[*split_funcs][0]].function_space().mesh().submesh_parent
        )
        V_label = V.reconstruct(mesh=parent_mesh)
        f_label = Function(V_label, val=f.dat)

        for split_label, val in split_funcs.items():
            assert val.function_space().mesh().submesh_parent == parent_mesh
            if child:
                split_label = int("10" + str(split_label))
            if isinstance(f_label, Function):
                f_label.assign(val, allow_missing_dofs=True)
            else:
                curr = Function(f_label.function_space()).assign(
                    val, allow_missing_dofs=True
                )
                f_label.assign(f_label + curr)  # partition of unity for restriction
        return f


def get_c2f_f2c_fd(fine_mesh, coarse_mesh):
    """
    Construct coarse->fine and fine->coarse relations by mapping netgen elements to firedrake ones
    """
    fine_dm = fine_mesh.topology_dm
    tdim = fine_dm.getDimension()
    ngmesh = fine_mesh.netgen_mesh
    if tdim == 2:
        parents = ngmesh.parentsurfaceelements.NumPy()
    elif tdim == 3:
        parents = ngmesh.parentelements.NumPy()
    else:
        raise RuntimeError("Adaptivity not implemented in dimension of mesh")

    fstart, fend = fine_mesh.topology_dm.getDepthStratum(tdim)
    cstart, cend = coarse_mesh.topology_dm.getDepthStratum(tdim)
    total_coarse_cells = FunctionSpace(coarse_mesh, "DG", 0).dim()

    c2f = [[] for _ in range(cstart, cend)]
    f2c = [[] for _ in range(fstart, fend)]

    if parents.shape[0] == 0:
        raise RuntimeError("Added mesh has not refined any cells from previous mesh")

    for l in range(fstart, fend):
        a = l
        while a >= total_coarse_cells:
            a = parents[a][0]

        fine_ind = fine_mesh._cell_numbering.getOffset(l)
        coarse_ind = coarse_mesh._cell_numbering.getOffset(a)

        fine_ind -= fstart
        coarse_ind -= cstart

        f2c[fine_ind].append(coarse_ind)
        c2f[coarse_ind].append(fine_ind)

    return c2f, np.array(f2c, dtype=PETSc.IntType)


def split_to_submesh(mesh, coarse_mesh, c2f, f2c):
    """
    Computes submesh split from full mesh.
    Returns splits which are Functions denoting whether elements
    belong to the corresponing submesh (bool)
    """
    max_children = 2 ** mesh.topological_dimension
    V = FunctionSpace(mesh, "DG", 0)
    V2 = FunctionSpace(coarse_mesh, "DG", 0)
    coarse_splits = {
        i: Function(V2, name=f"{i}_elements") for i in range(1, max_children + 1)
    }
    fine_splits = {
        i: Function(V, name=f"{i}_elements") for i in range(1, max_children + 1)
    }
    num_children = np.zeros((len(c2f),))
    for i, children in enumerate(c2f):
        n = len(children)
        if 1 <= n <= max_children:
            num_children[i] = n

    for i in range(1, max_children + 1):
        coarse_splits[i].dat.data_wo_with_halos[num_children == i] = 1
        fine_splits[i].dat.data_wo_with_halos[num_children[f2c.squeeze()] == i] = 1

    return coarse_splits, fine_splits, num_children


def get_full_to_sub_numbering(mesh, submesh):
    """
    Returns the submesh cell id associated with the full mesh cell id
    """
    V1 = FunctionSpace(mesh, "DG", 0)
    V2 = FunctionSpace(submesh, "DG", 0)
    u1 = Function(V1)
    u2 = Function(V2)
    u2.dat.data_wo[:] = np.arange(submesh.cell_set.size)
    u1.assign(u2, allow_missing_dofs=True)

    return u1.dat.data_ro_with_halos.astype(PETSc.IntType)
