from collections import defaultdict
from fractions import Fraction

from firedrake.mesh import MeshGeometry
from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.mg.mesh import HierarchyBase
from firedrake.mg.utils import set_level, set_dm_refine_level

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
            ``mesh``. If not given, it is read from ``mesh._adaptive_cell_maps``
            (set automatically when ``mesh`` was produced by
            :meth:`~firedrake.mesh.MeshGeometry.refine_marked_elements`); if
            that attribute is absent too, no cell maps are recorded for this
            level.
        fine_to_coarse_cells
            Optional map from cells on ``mesh`` to cells on the previous
            finest level. Falls back the same way as ``coarse_to_fine_cells``.
        """
        level = len(self.meshes)
        if level > 0 and (coarse_to_fine_cells is None or fine_to_coarse_cells is None):
            # A mesh returned by MeshGeometry.refine_marked_elements carries
            # its own cell maps as a private `_adaptive_cell_maps` attribute
            # (relative to the mesh it was refined from), set at construction
            # time in firedrake.adapt.refine_marked_elements. This lets a mesh
            # be adaptively refined on its own, without being attached to a
            # hierarchy, and still have `add_mesh` pick up its cell maps here
            # if it is added to one later. Meshes built any other way do not
            # have this attribute, so getattr's default of (None, None) is
            # used: no cell maps are recorded for this level.
            coarse_to_fine_cells, fine_to_coarse_cells = getattr(
                mesh, "_adaptive_cell_maps", (None, None)
            )

        self._meshes.append(mesh)
        self.meshes.append(mesh)
        set_level(mesh, self, level)
        set_dm_refine_level(mesh, level)

        if level > 0 and coarse_to_fine_cells is not None and fine_to_coarse_cells is not None:
            self.coarse_to_fine_cells[Fraction(level - 1, 1)] = coarse_to_fine_cells
            self.fine_to_coarse_cells[Fraction(level, 1)] = fine_to_coarse_cells

    def adapt(self, eta: Function | Cofunction, theta: float):
        """
        Adds a new mesh to the hierarchy by locally refining the finest mesh
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
        with eta.dat.vec_ro as evec:
            _, eta_max = evec.max()

        threshold = theta * eta_max
        should_refine = eta.dat.data_ro > threshold

        markers = Function(M)
        markers.dat.data_wo[should_refine] = 1

        refined_mesh = mesh.refine_marked_elements(markers)
        self.add_mesh(refined_mesh)
        return self.meshes[-1]
