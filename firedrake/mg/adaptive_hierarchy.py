from collections import defaultdict
from fractions import Fraction

import numpy as np

from firedrake.mesh import MeshGeometry
from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.mg import HierarchyBase
from firedrake.mg.utils import set_level
from firedrake.utils import IntType

__all__ = ["AdaptiveMeshHierarchy"]


def _netgen_cells(mesh):
    tdim = mesh.topological_dimension
    if tdim == 2:
        return mesh.netgen_mesh.Elements2D()
    elif tdim == 3:
        return mesh.netgen_mesh.Elements3D()
    raise NotImplementedError("Adaptive hierarchy maps are only implemented in dimension 2 and 3.")


def _distribute_cell_data(mesh, values):
    from firedrake.functionspace import FunctionSpace
    from firedrake.netgen import netgen_distribute

    DG0 = FunctionSpace(mesh, "DG", 0)
    data = np.asarray(netgen_distribute(DG0, values), dtype=IntType)
    cell_nodes = DG0.cell_node_map().values_with_halo[:, 0]
    return data[cell_nodes]


def _adaptive_cell_maps(coarse_mesh, fine_mesh, parent_cell_numbers):
    if not (hasattr(coarse_mesh, "netgen_mesh") and hasattr(fine_mesh, "netgen_mesh")):
        return None, None

    coarse_cell_numbers = _distribute_cell_data(
        coarse_mesh, np.arange(len(_netgen_cells(coarse_mesh)), dtype=IntType)
    )
    fine_parent_numbers = _distribute_cell_data(
        fine_mesh, np.asarray(parent_cell_numbers, dtype=IntType)
    )

    coarse_owned = coarse_mesh.cell_set.size
    coarse_total = min(coarse_mesh.cell_set.total_size, coarse_cell_numbers.size)
    fine_owned = fine_mesh.cell_set.size
    fine_total = min(fine_mesh.cell_set.total_size, fine_parent_numbers.size)

    coarse_lookup = {}
    for cell, cell_number in enumerate(coarse_cell_numbers[:coarse_total]):
        coarse_lookup.setdefault(int(cell_number), cell)
    coarse_owned_lookup = {
        int(cell_number): cell
        for cell, cell_number in enumerate(coarse_cell_numbers[:coarse_owned])
    }

    fine_to_coarse = np.full((fine_owned, 1), -1, dtype=IntType)
    for fine_cell, parent_number in enumerate(fine_parent_numbers[:fine_owned]):
        fine_to_coarse[fine_cell, 0] = coarse_lookup.get(int(parent_number), -1)

    children = [[] for _ in range(coarse_owned)]
    for fine_cell, parent_number in enumerate(fine_parent_numbers[:fine_total]):
        coarse_cell = coarse_owned_lookup.get(int(parent_number))
        if coarse_cell is not None:
            children[coarse_cell].append(fine_cell)

    max_children = max((len(local_children) for local_children in children), default=0)
    coarse_to_fine = np.full((coarse_owned, max(1, max_children)), -1, dtype=IntType)
    for coarse_cell, local_children in enumerate(children):
        coarse_to_fine[coarse_cell, :len(local_children)] = local_children

    return coarse_to_fine, fine_to_coarse


class AdaptiveMeshHierarchy(HierarchyBase):
    """
    HierarchyBase for hierarchies of adaptively refined meshes.

    Parameters
    ----------
    base_mesh
        The coarsest mesh in the hierarchy.
    nested: bool
        A flag to indicate whether the meshes are nested.

    """
    def __init__(self, base_mesh: MeshGeometry, nested: bool = True):
        self.meshes = []
        self._meshes = []
        self.coarse_to_fine_cells = {}
        self.fine_to_coarse_cells = {Fraction(0, 1): None}
        self.refinements_per_level = 1
        self.nested = nested
        self._shared_data_cache = defaultdict(dict)
        self.add_mesh(base_mesh)

    def add_mesh(self, mesh: MeshGeometry, coarse_to_fine_cells=None, fine_to_coarse_cells=None):
        """
        Adds a mesh into the hierarchy.

        Parameters
        ----------
        mesh
            The mesh to be added to the finest level.
        coarse_to_fine_cells
            Optional map from cells on the previous finest level to cells on
            ``mesh``.
        fine_to_coarse_cells
            Optional map from cells on ``mesh`` to cells on the previous
            finest level.
        """
        level = len(self.meshes)
        if level > 0 and (coarse_to_fine_cells is None or fine_to_coarse_cells is None):
            parent_cell_numbers = getattr(mesh, "_adaptive_parent_cell_numbers", None)
            if parent_cell_numbers is not None:
                coarse_to_fine_cells, fine_to_coarse_cells = _adaptive_cell_maps(
                    self.meshes[-1], mesh, parent_cell_numbers
                )

        self._meshes.append(mesh)
        self.meshes.append(mesh)
        set_level(mesh, self, level)

        if level > 0 and coarse_to_fine_cells is not None and fine_to_coarse_cells is not None:
            self.coarse_to_fine_cells[Fraction(level - 1, 1)] = coarse_to_fine_cells
            self.fine_to_coarse_cells[Fraction(level, 1)] = fine_to_coarse_cells

    def adapt(self, eta: Function | Cofunction, theta: float):
        """
        Adds a new mesh to the hierarchy by locally refining the finest mesh
        with a simplified variant of Dorfler marking. The finest mesh must
        come from a netgen mesh.

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
        with eta.dat.vec_ro as evec:
            _, eta_max = evec.max()

        threshold = theta * eta_max
        should_refine = eta.dat.data_ro > threshold

        markers = Function(M)
        markers.dat.data_wo[should_refine] = 1

        refined_mesh = mesh.refine_marked_elements(markers)
        self.add_mesh(refined_mesh)
        return refined_mesh
