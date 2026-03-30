from firedrake.mesh import MeshGeometry
from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.mg import HierarchyBase
from firedrake.mg.utils import set_level

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
        self.nested = nested
        self.add_mesh(base_mesh)

    def add_mesh(self, mesh: MeshGeometry):
        """
        Adds a mesh into the hierarchy.

        Parameters
        ----------
        mesh
            The mesh to be added to the finest level.
        """
        level = len(self.meshes)
        self._meshes.append(mesh)
        self.meshes.append(mesh)
        set_level(mesh, self, level)

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
