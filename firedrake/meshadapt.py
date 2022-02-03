import abc
import numpy as np

import firedrake
import firedrake.cython.dmcommon as dmcommon
import firedrake.mesh as fmesh
from firedrake.petsc import PETSc, OptionsManager
import firedrake.utils as utils
import ufl


__all__ = []


class Metric(abc.ABC):
    """
    Abstract base class that defines the API for all metrics.
    """
    def __init__(self, mesh, metric_parameters={}):
        """
        :arg mesh: mesh upon which to build the metric
        :kwarg metric_parameters: a dictionary of parameters related
            to the metric
        """
        mesh.init()
        self.dim = mesh.topological_dimension()
        if self.dim not in (2, 3):
            raise ValueError(f"Mesh must be 2D or 3D, not {self.dim}")
        self.mesh = mesh
        self.plex = mesh.topology_dm
        self.update_plex_coordinates()
        self.set_parameters(metric_parameters)

    # NOTE: This will become redundant at some point
    def update_plex_coordinates(self):
        """
        Ensure that the coordinates of the Firedrake mesh and
        the underlying DMPlex are consistent.
        """
        entity_dofs = np.zeros(self.dim + 1, dtype=np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        coord_section = self.mesh.create_section(entity_dofs)
        # NOTE: section doesn't have any fields, but PETSc assumes it to have one
        coord_dm = self.plex.getCoordinateDM()
        coord_dm.setDefaultSection(coord_section)
        coords_local = coord_dm.createLocalVec()
        coords_local.array[:] = np.reshape(
            self.mesh.coordinates.dat.data_ro_with_halos, coords_local.array.shape
        )
        self.plex.setCoordinatesLocal(coords_local)

    @abc.abstractmethod
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
      https://petsc.org/release/docs/manual/dmplex/#metric-based-mesh-adaptation
    """

    @PETSc.Log.EventDecorator("RiemannianMetric.__init__")
    def __init__(self, mesh, metric_parameters={}):
        """
        :arg mesh: mesh upon which to build the metric
        :kwarg metric_parameters: PETSc parameters for
            metric construction
        """
        self.normalisation_order = 1.0
        self.target_complexity = None

        # Initialisation, including setting parameters
        super().__init__(mesh, metric_parameters=metric_parameters)
        self.function_space = firedrake.TensorFunctionSpace(mesh, "CG", 1)
        self.function = firedrake.Function(self.function_space)
        entity_dofs = np.zeros(self.dim + 1, dtype=np.int32)
        entity_dofs[0] = self.dim * self.dim
        self.plex.setSection(self.mesh.create_section(entity_dofs))

        # Create separate data structures for the metric determinant
        det_mesh = firedrake.Mesh(mesh.coordinates)
        det_plex = det_mesh.topology_dm
        self.P1 = firedrake.FunctionSpace(det_mesh, "CG", 1)
        entity_dofs[0] = 1
        det_plex.setSection(det_mesh.create_section(entity_dofs))

    @property
    def dat(self):
        return self.function.dat

    @property
    def vec(self):
        with self.dat.vec_ro as v:
            return v

    @property
    @PETSc.Log.EventDecorator("RiemannianMetric.reorder_metric")
    def _reordered(self):
        return dmcommon.to_petsc_local_numbering(self.vec, self.function_space)

    @staticmethod
    def _process_parameters(metric_parameters):
        mp = metric_parameters.copy()
        if "dm_plex_metric" in mp:
            for key, value in mp["dm_plex_metric"].items():
                mp["_".join(["dm_plex_metric", key])] = value
            mp.pop("dm_plex_metric")
        return mp

    def set_parameters(self, metric_parameters={}):
        """
        Set metric parameter values internally.

        :kwarg metric_parameters: a dictionary of metric parameters,
            to be passed to the DMPlex
        """
        mp = self._process_parameters(metric_parameters)
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
        opts = OptionsManager(mp, "")
        with opts.inserted_options():
            self.plex.metricSetFromOptions()
        if self.plex.metricIsIsotropic():
            raise NotImplementedError("Isotropic optimisations are not supported in Firedrake")
        if self.plex.metricIsUniform():
            raise NotImplementedError("Uniform optimisations are not supported in Firedrake")

    @PETSc.Log.EventDecorator("RiemannianMetric.enforce_spd")
    def enforce_spd(self, restrict_sizes=False, restrict_anisotropy=False):
        """
        Enforce that the metric is symmetric positive-definite.

        :kwarg restrict_sizes: should minimum and maximum metric magnitudes
            be enforced?
        :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
        :return: the :class:`RiemannianMetric`.
        """
        kw = {
            "restrictSizes": restrict_sizes,
            "restrictAnisotropy": restrict_anisotropy,
        }
        with firedrake.Function(self.function_space).dat.vec as met:
            with firedrake.Function(self.P1).dat.vec as det:
                det.setDM(self.P1.mesh().topology_dm)
                self.plex.metricEnforceSPD(self.vec, met, det, **kw)
            with self.dat.vec_wo as v:
                v.copy(met)
        return self

    @PETSc.Log.EventDecorator("RiemannianMetric.normalise")
    def normalise(self, global_factor=None, **kwargs):
        """
        Apply :math:`L^p` normalisation to the metric.

        :kwarg global_factor: pre-computed global normalisation factor
        :kwarg restrict_sizes: should minimum and maximum metric magnitudes
            be enforced?
        :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
        :return: the normalised :class:`RiemannianMetric`.
        """
        kwargs.setdefault("restrict_sizes", True)
        kwargs.setdefault("restrict_anisotropy", True)
        d = self.function_space.mesh().topological_dimension()
        p = self.normalisation_order
        target = self.target_complexity
        if target is None:
            raise ValueError("dm_plex_metric_target_complexity must be set.")

        # Compute global normalisation factor
        detM = ufl.det(self.function)
        if global_factor is None:
            dX = ufl.dx(domain=self.mesh)
            exponent = 0.5 if np.isinf(p) else p / (2 * p + d)
            integral = firedrake.assemble(pow(detM, exponent) * dX)
            global_factor = firedrake.Constant(pow(target / integral, 2 / d))

        # Normalise the metric
        determinant = 1 if np.isinf(p) else pow(detM, -1 / (2 * p + d))
        self.interpolate(global_factor * determinant * self.function)

        # Enforce element constraints
        return self.enforce_spd(**kwargs)

    @PETSc.Log.EventDecorator("RiemannianMetric.complexity")
    def complexity(self, boundary=False):
        """
        Compute the metric complexity - the continuous analogue
        of the (inherently discrete) mesh vertex count.

        :kwarg boundary: should the complexity be computed over the
            domain boundary?
        :return: the metric complexity.
        """
        dX = ufl.ds if boundary else ufl.dx
        C = ufl.sqrt(ufl.det(self.function)) * dX
        return firedrake.assemble(C)

    @PETSc.Log.EventDecorator("RiemannianMetric.intersect")
    def intersect(self, *metrics):
        """
        Intersect the metric with other metrics.

        Metric intersection means taking the minimal ellipsoid in the
        direction of each eigenvector at each point in the domain.

        :arg metrics: the metrics to be intersected with
        :return: the intersected :class:`RiemannianMetric`.
        """
        for metric in metrics:
            assert isinstance(metric, RiemannianMetric)
        num_metrics = len(metrics)
        if num_metrics == 0:
            return self
        with firedrake.Function(self.function_space).dat.vec as met:
            if num_metrics == 1:
                self.plex.metricIntersection2(self.vec, metrics[0].vec, met)
            elif num_metrics == 2:
                self.plex.metricIntersection3(self.vec, metrics[0].vec, metrics[1].vec, met)
            else:
                raise NotImplementedError(
                    f"Can only intersect 1, 2 or 3 metrics, not {num_metrics+1}"
                )
            with self.dat.vec_wo as v:
                v.copy(met)
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
