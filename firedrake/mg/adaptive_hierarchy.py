"""
This module contains the class for the AdaptiveMeshHierarchy and
related helper functions
"""

from fractions import Fraction
from collections import defaultdict

from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.mesh import Mesh
from firedrake.mg import HierarchyBase
from firedrake.mg.utils import set_level

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
        self._meshes += (mesh,)
        self.meshes += (mesh,)
        level = len(self.meshes)
        set_level(self.meshes[-1], self, level - 1)
        self._shared_data_cache = defaultdict(dict)

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
