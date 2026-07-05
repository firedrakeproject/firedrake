from collections import defaultdict
from fractions import Fraction

from firedrake.mesh import MeshGeometry
from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.mg.mesh import HierarchyBase
from firedrake.mg.utils import set_level, set_refine_level
from firedrake.netgen import (adaptive_cell_maps, adaptive_netgen_cells,
                              adaptive_parent_cell_numbers,
                              make_adaptive_refined_mesh)

__all__ = ["AdaptiveMeshHierarchy"]


class AdaptiveMeshHierarchy(HierarchyBase):
    """
    HierarchyBase for hierarchies of adaptively refined meshes.

    Parameters
    ----------
    base_mesh
        The coarsest mesh in the hierarchy.
    nested: bool
        A flag to indicate whether the meshes are nested.
    redistribute: bool
        If ``True``, keep adaptively refined meshes redistributed when
        this is needed to avoid empty ranks and use an internal
        parent-owned mesh for transfer operators.

    """
    def __init__(self, base_mesh: MeshGeometry, nested: bool = True,
                 redistribute: bool = True):
        self.meshes = []
        self._meshes = []
        self.coarse_to_fine_cells = {}
        self.fine_to_coarse_cells = {Fraction(0, 1): None}
        self.refinements_per_level = 1
        self.nested = nested
        self.redistribute = redistribute
        self._shared_data_cache = defaultdict(dict)
        self.add_mesh(base_mesh)

    def add_mesh(self, mesh: MeshGeometry,
                 coarse_to_fine_cells=None,
                 fine_to_coarse_cells=None):
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
            redist = getattr(mesh, "redist", None)
            transfer_mesh = redist.orig if redist is not None else mesh
            parent_cell_numbers = getattr(transfer_mesh, "_adaptive_parent_cell_numbers", None)
            if parent_cell_numbers is None:
                parent_cell_numbers = getattr(mesh, "_adaptive_parent_cell_numbers", None)
            if parent_cell_numbers is None:
                parent_cell_numbers = adaptive_parent_cell_numbers(self.meshes[-1], mesh)

            if parent_cell_numbers is not None:
                if not getattr(transfer_mesh, "_adaptive_parent_owned", False):
                    if mesh.comm.size == 1:
                        mesh._adaptive_parent_cell_numbers = parent_cell_numbers
                        mesh._adaptive_parent_owned = True
                        transfer_mesh = mesh
                    else:
                        mesh = make_adaptive_refined_mesh(
                            self.meshes[-1], mesh.netgen_mesh, mesh.netgen_flags,
                            parent_cell_numbers, self.redistribute,
                        )
                        redist = getattr(mesh, "redist", None)
                        transfer_mesh = redist.orig if redist is not None else mesh
                        parent_cell_numbers = transfer_mesh._adaptive_parent_cell_numbers

                coarse_to_fine_cells, fine_to_coarse_cells = adaptive_cell_maps(
                    self.meshes[-1], transfer_mesh, parent_cell_numbers
                )

        self._meshes.append(mesh)
        self.meshes.append(mesh)
        set_level(mesh, self, level)
        set_refine_level(mesh, level)
        redist = getattr(mesh, "redist", None)

        if hasattr(mesh, "netgen_mesh"):
            mesh._adaptive_netgen_num_cells = len(adaptive_netgen_cells(mesh))
            if redist is not None:
                redist.orig._adaptive_netgen_num_cells = mesh._adaptive_netgen_num_cells

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

        refined_mesh = mesh.refine_marked_elements(markers, redistribute=self.redistribute)
        self.add_mesh(refined_mesh)
        return self.meshes[-1]
