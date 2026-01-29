from tsfc.exceptions import MismatchingDomainError  # noqa: F401


class FiredrakeException(Exception):
    """Base class for all Firedrake exceptions."""


class ConvergenceError(FiredrakeException):
    """Error raised when a solver fails to converge."""


class DofNotDefinedError(FiredrakeException):
    r"""Raised when attempting to interpolate across function spaces where the
    target function space contains degrees of freedom (i.e. nodes) which cannot
    be defined in the source function space. This typically occurs when the
    target mesh covers a larger domain than the source mesh.
    """


class VertexOnlyMeshMissingPointsError(FiredrakeException):
    """Exception raised when 1 or more points are not found by a
    :func:`~.VertexOnlyMesh` in its parent mesh.

    Attributes
    ----------
    n_missing_points
        The number of points which were not found in the parent mesh.
    """

    def __init__(self, n_missing_points: int):
        self.n_missing_points = n_missing_points

    def __str__(self):
        return (
            f"{self.n_missing_points} vertices are outside the mesh and have "
            "been removed from the VertexOnlyMesh."
        )


class NonUniqueMeshSequenceError(FiredrakeException):
    """Raised when calling `.unique()` on a MeshSequence which contains
    non-unique meshes.
    """
