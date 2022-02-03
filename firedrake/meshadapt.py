from abc import ABCMeta, abstractmethod
import numpy as np

import firedrake
import firedrake.constant as const
import firedrake.cython.dmcommon as dmcommon
import firedrake.function as func
import firedrake.functionspace as fs
import firedrake.mesh as fmesh
from firedrake.petsc import PETSc
import firedrake.utils as utils
import ufl


__all__ = []


class Metric(object):
    """
    Abstract class that defines the API for all metrics.
    """

    __metaclass__ = ABCMeta

    def __init__(self, mesh, metric_parameters={}):
        """
        :arg mesh: mesh upon which to build the metric
        """
        mesh.init()
        self.dim = mesh.topological_dimension()
        if self.dim not in (2, 3):
            raise ValueError(f"Mesh must be 2D or 3D, not {self.dim}")
        self.mesh = mesh
        self.update_plex_coordinates()
        self.set_metric_parameters(metric_parameters)

    # TODO: This will be redundant at some point
    def update_plex_coordinates(self):
        """
        Ensure that the coordinates of the Firedrake mesh and
        the underlying DMPlex are consistent.
        """
        self.plex = self.mesh.topology_dm
        entity_dofs = np.zeros(self.dim + 1, dtype=np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        coord_section = self.mesh.create_section(entity_dofs)
        # FIXME: section doesn't have any fields, but PETSc assumes it to have one
        coord_dm = self.plex.getCoordinateDM()
        coord_dm.setDefaultSection(coord_section)
        coords_local = coord_dm.createLocalVec()
        coords_local.array[:] = np.reshape(
            self.mesh.coordinates.dat.data_ro_with_halos, coords_local.array.shape
        )
        self.plex.setCoordinatesLocal(coords_local)

    @abstractmethod
    def set_parameters(self, metric_parameters={}):
        """
        Set the :attr:`metric_parameters` so that they can be
        used to drive the mesh adaptation routine.
        """
        pass


class RiemannianMetric(Metric):
    r"""
    Class for defining a Riemannian metric over a
    given mesh.

    A metric is a symmetric positive-definite field,
    which conveys how the mesh is to be adapted. If
    the mesh is of dimension :math:`d` then the metric
    takes the value of a square :math:`d\times d`
    matrix at each point.

    The implementation of metric-based mesh adaptation
    used in PETSc assumes that the metric is piece-wise
    linear and continuous, with its degrees of freedom
    at the mesh vertices.

    For details, see the PETSc manual entry:
      https://petsc.org/release/docs/manual/dmplex.html#metric-based-mesh-adaptation
    """
    normalisation_order = 1.0
    target_complexity = None

    @PETSc.Log.EventDecorator("RiemannianMetric.__init__")
    def __init__(self, mesh, metric_parameters={}):
        """
        :arg mesh: mesh upon which to build the metric
        :kwarg metric_parameters: PETSc parameters for
            metric construction
        """
        super().__init__(mesh, metric_parameters=metric_parameters)
        self.V = fs.TensorFunctionSpace(mesh, "CG", 1)
        self.function = func.Function(self.V)

    @property
    def dat(self):
        return self.function.dat

    @property
    def vec(self):
        with self.dat.vec_ro as v:
            return v

    @property
    @PETSc.Log.EventDecorator("RiemannianMetric.reordered")
    def reordered(self):
        return dmcommon.to_petsc_local_numbering(self.vec, self.V)

    @staticmethod
    def _process_parameters(metric_parameters):
        mp = metric_parameters.copy()
        if "dm_plex_metric" in mp:
            for key, value in mp["dm_plex_metric"].items():
                mp["_".join(["dm_plex_metric", key])] = value
            mp.pop("dm_plex_metric")
        return mp

    def set_metric_parameters(self, metric_parameters={}):
        """
        Apply the :attr:`metric_parameters` to the DMPlex.
        """
        mp = self._process_parameters(metric_parameters)
        OptDB = PETSc.Options()
        if "dm_plex_metric_p" in mp:
            p = mp.pop("dm_plex_metric_p")
            self.normalisation_order = p
            if not np.isinf(p) and p < 1.0:
                raise ValueError(
                    "Metric normalisation order must be at least 1,"
                    f" not {p}"
                )
        if "dm_plex_metric_target_complexity" in mp:
            target = mp.pop("dm_plex_metric_target_complexity")
            self.target_complexity = target
            if target <= 0.0:
                raise ValueError(
                    "Target metric complexity must be positive,"
                    f" not {target}"
                )
        for key, value in mp.items():
            OptDB.setValue(key, value)
        self.plex.metricSetFromOptions()

    @PETSc.Log.EventDecorator("RiemannianMetric.enforce_spd")
    def enforce_spd(self, restrict_sizes=False, restrict_anisotropy=False):
        """
        Enforce that the metric is symmetric positive-definite.

        :kwarg restrict_sizes: should minimum and maximum metric magnitudes
            be enforced?
        :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
        """
        tmp = self.plex.metricEnforceSPD(
            self.vec,
            restrictSizes=restrict_sizes,
            restrictAnisotropy=restrict_anisotropy,
        )
        with self.dat.vec_wo as v:
            v.copy(tmp)
        return self

    @PETSc.Log.EventDecorator("RiemannianMetric.normalise")
    def normalise(self, boundary=False, global_factor=None, **kwargs):
        """
        Apply :math:`L^p` normalisation to the metric.

        :kwarg boundary: should the normalisation be performed over the
            domain boundary?
        :kwarg global_factor: pre-computed global normalisation factor
        :kwarg restrict_sizes: should minimum and maximum metric magnitudes
            be enforced?
        :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
        """
        kwargs.setdefault("restrict_sizes", True)
        kwargs.setdefault("restrict_anisotropy", True)
        d = self.V.mesh().topological_dimension()
        if boundary:
            d -= 1
        p = self.normalisation_order
        target = self.target_complexity
        if target is None:
            raise ValueError("dm_plex_metric_target_complexity must be set.")

        # Compute global normalisation factor
        detM = ufl.det(self.function)
        if global_factor is None:
            dX = (ufl.ds if boundary else ufl.dx)(domain=self.mesh)
            exponent = 0.5 if np.isinf(p) else p / (2 * p + d)
            integral = firedrake.assemble(pow(detM, exponent) * dX)
            global_factor = const.Constant(pow(target / integral, 2 / d))

        # Normalise the metric
        determinant = 1 if np.isinf(p) else pow(detM, -1 / (2 * p + d))
        self.interpolate(global_factor * determinant * self.function)

        # Enforce element constraints
        return self.enforce_spd(**kwargs)

    @PETSc.Log.EventDecorator("RiemannianMetric.intersect")
    def intersect(self, *metrics):
        """
        Intersect the metric with other metrics.

        Metric intersection means taking the minimal ellipsoid in the
        direction of each eigenvector at each point in the domain.

        :arg metrics: the metrics to be intersected with
        """
        for metric in metrics:
            assert isinstance(metric, RiemannianMetric)
        num_metrics = len(metrics)
        if num_metrics == 0:
            return self
        elif num_metrics == 1:
            tmp = self.plex.metricIntersection2(self.vec, metrics[0].vec)
        elif num_metrics == 2:
            tmp = self.plex.metricIntersection3(
                self.vec, metrics[0].vec, metrics[1].vec
            )
        else:
            raise NotImplementedError(
                f"Can only intersect 1, 2 or 3 metrics, not {num_metrics+1}"
            )
        with self.dat.vec_wo as v:
            v.copy(tmp)
        return self

    def rename(self, name):
        self.function.rename(name)
        with self.dat.vec_wo as v:
            v.setName(name)

    def assign(self, *args, **kwargs):
        self.function.assign(*args, **kwargs)
        return self

    def interpolate(self, *args, **kwargs):
        self.function.interpolate(*args, **kwargs)
        return self

    def project(self, *args, **kwargs):
        self.function.project(*args, **kwargs)
        return self
