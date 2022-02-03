import abc
from collections.abc import Iterable
import firedrake
import firedrake.cython.dmcommon as dmcommon
import firedrake.function as ffunc
import firedrake.functionspace as ffs
import firedrake.mesh as fmesh
from firedrake.petsc import PETSc, OptionsManager
import firedrake.utils as futils
import numpy as np
import ufl

__all__ = ["RiemannianMetric", "MetricBasedAdaptor", "adapt"]


class RiemannianMetric(ffunc.Function):
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
    @PETSc.Log.EventDecorator()
    def __init__(self, function_space, *args, **kwargs):
        r"""
        :param function_space: the tensor :class:`.FunctionSpace`, on which to build this
            :class:`RiemannianMetric`. Alternatively, another :class:`Function` may be
            passed here and its function space will be used to build this :class:`Function`.
            In this case, the function values are copied. If a :class:`.MeshGeometry` is
            passed here then a tensor :math:`\mathbb P1` space is built on top of it.
        """
        if isinstance(function_space, fmesh.MeshGeometry):
            function_space = ffs.TensorFunctionSpace(function_space, "CG", 1)
        super().__init__(function_space, *args, **kwargs)
        self.metric_parameters = {}
        self.normalisation_order = 1.0
        self.target_complexity = None

        # Check that we have an appropriate tensor P1 function
        fs = self.function_space()
        mesh = fs.mesh()
        tdim = mesh.topological_dimension()
        if tdim not in (2, 3):
            raise ValueError(
                f"Riemannian metric should be 2D or 3D, not {tdim}D"
            )
        el = fs.ufl_element()
        if (el.family(), el.degree()) != ("Lagrange", 1):
            raise ValueError(
                f"Riemannian metric should be in P1 space, not {el}"
            )
        if isinstance(fs.dof_count, Iterable):
            raise ValueError(
                "Riemannian metric cannot be built in a mixed space"
            )
        rank = len(fs.dof_dset.dim)
        if rank != 2:
            raise ValueError(
                "Riemannian metric should be matrix-valued,"
                f" not rank-{rank} tensor-valued"
            )

        # Stash mesh data
        plex = mesh.topology_dm
        self._mesh = mesh
        self._plex = plex
        self._tdim = tdim

        # Ensure DMPlex coordinates are consistent
        self.update_plex_coordinates()

        # Adjust the section
        entity_dofs = np.zeros(tdim + 1, dtype=np.int32)
        entity_dofs[0] = tdim ** 2
        sec = mesh.create_section(entity_dofs)
        off = 0
        for c in range(*plex.getChart()):
            sec.setOffset(c, off)
            off += sec.getDof(c)
        plex.setSection(sec)

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

        :param metric_parameters: a dictionary of parameters to be passed to PETSc's
            Riemmanian metric implementation. All such options have the prefix
            `dm_plex_metric_`.
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
            self._plex.metricSetFromOptions()
        if self._plex.metricIsIsotropic():
            raise NotImplementedError("Isotropic optimisations are not supported in Firedrake")
        if self._plex.metricIsUniform():
            raise NotImplementedError("Uniform optimisations are not supported in Firedrake")
        self.metric_parameters = mp

    # NOTE: This will become redundant at some point
    @PETSc.Log.EventDecorator()
    def update_plex_coordinates(self):
        """
        Ensure that the coordinates of the Firedrake mesh and
        the underlying DMPlex are consistent.
        """
        mesh = self._mesh
        plex = self._plex
        entity_dofs = np.zeros(self._tdim + 1, dtype=np.int32)
        entity_dofs[0] = mesh.geometric_dimension()
        coord_section = mesh.create_section(entity_dofs)
        # NOTE: section doesn't have any fields, but PETSc assumes it to have one
        coord_dm = plex.getCoordinateDM()
        coord_dm.setSection(coord_section)
        coords_local = coord_dm.createLocalVec()
        coords_local.array[:] = np.reshape(
            mesh.coordinates.dat.data_ro_with_halos, coords_local.array.shape
        )
        plex.setCoordinatesLocal(coords_local)

    @property
    @PETSc.Log.EventDecorator("RiemannianMetric.reorder_metric")
    def _reordered(self):
        with self.dat.vec_ro as v:
            return dmcommon.to_petsc_local_numbering(v, self.function_space())

    @PETSc.Log.EventDecorator()
    def enforce_spd(self, restrict_sizes=False, restrict_anisotropy=False):
        """
        Enforce that the metric is symmetric positive-definite.

        :param restrict_sizes: should minimum and maximum metric magnitudes
            be enforced?
        :param restrict_anisotropy: should maximum anisotropy be enforced?
        :return: the :class:`RiemannianMetric`.
        """
        kw = {
            "restrictSizes": restrict_sizes,
            "restrictAnisotropy": restrict_anisotropy,
        }
        plex = self._plex
        with ffunc.Function(self.function_space()).dat.vec as met:
            det = plex.metricDeterminantCreate()
            with self.dat.vec as v:
                plex.metricEnforceSPD(v, met, det, **kw)
                met.copy(v)
        return self

    @PETSc.Log.EventDecorator()
    def normalise(self, global_factor=None, **kwargs):
        """
        Apply :math:`L^p` normalisation to the metric.

        :param global_factor: pre-computed global normalisation factor
        :param restrict_sizes: should minimum and maximum metric magnitudes
            be enforced?
        :param restrict_anisotropy: should maximum anisotropy be enforced?
        :return: the normalised :class:`RiemannianMetric`.
        """
        kwargs.setdefault("restrict_sizes", True)
        kwargs.setdefault("restrict_anisotropy", True)
        d = self._tdim
        p = self.normalisation_order
        target = self.target_complexity
        if target is None:
            raise ValueError("dm_plex_metric_target_complexity must be set.")

        # Compute global normalisation factor
        detM = ufl.det(self)
        if global_factor is None:
            dX = ufl.dx(domain=self._mesh)
            exponent = 0.5 if np.isinf(p) else p / (2 * p + d)
            integral = firedrake.assemble(pow(detM, exponent) * dX)
            global_factor = firedrake.Constant(pow(target / integral, 2 / d))

        # Normalise the metric
        determinant = 1 if np.isinf(p) else pow(detM, -1 / (2 * p + d))
        self.interpolate(global_factor * determinant * self)

        # Enforce element constraints
        return self.enforce_spd(**kwargs)

    @PETSc.Log.EventDecorator()
    def complexity(self, boundary=False):
        """
        Compute the metric complexity - the continuous analogue
        of the (inherently discrete) mesh vertex count.

        :param boundary: should the complexity be computed over the
            domain boundary?
        :return: the metric complexity.
        """
        dX = ufl.ds if boundary else ufl.dx
        return firedrake.assemble(ufl.sqrt(ufl.det(self)) * dX)

    @PETSc.Log.EventDecorator()
    def intersect(self, *metrics):
        """
        Intersect the metric with other metrics.

        Metric intersection means taking the minimal ellipsoid in the
        direction of each eigenvector at each point in the domain.

        :param metrics: the metrics to be intersected with
        :return: the intersected :class:`RiemannianMetric`.
        """
        for metric in metrics:
            assert isinstance(metric, RiemannianMetric)
        num_metrics = len(metrics) + 1
        if num_metrics == 1:
            return self
        plex = self._plex
        with ffunc.Function(self.function_space()).dat.vec as met:
            with self.dat.vec as v1:
                with metrics[0].dat.vec_ro as v2:
                    if num_metrics == 2:
                        plex.metricIntersection2(v1, v2, met)
                    elif num_metrics == 3:
                        with metrics[1].dat.vec_ro as v3:
                            plex.metricIntersection3(v1, v2, v3, met)
                    else:
                        raise NotImplementedError(
                            f"Can only intersect 1, 2 or 3 metrics, not {num_metrics+1}"
                        )
                met.copy(v1)
        return self

    @PETSc.Log.EventDecorator()
    def average(self, *metrics):
        """
        Average the metric with other metrics.

        :param metrics: the metrics to be averaged with
        :return: the averaged :class:`RiemannianMetric`.
        """
        num_metrics = len(metrics) + 1
        weight = 1.0 / num_metrics
        if num_metrics == 1:
            return self
        self *= weight
        for metric in metrics:
            self += weight * metric
        return self

    def copy(self, deepcopy=False):
        """
        Copy the metric and any associated parameters.

        :kwarg deepcopy: If ``True``, the new
            :class:`RiemannianMetric` will allocate new space
            and copy values.  If ``False``, the default, then the new
            :class:`RiemannianMetric` will share the dof values.
        """
        mp = self.metric_parameters.copy()
        func = super().copy(deepcopy=deepcopy)
        metric = RiemannianMetric(func)
        metric.set_parameters(mp)
        return metric


class AdaptorBase(abc.ABC):
    """
    Abstract base class that defines the API for all mesh adaptors.
    """
    def __init__(self, mesh):
        """
        :param mesh: mesh to be adapted
        """
        self.mesh = mesh

    @abc.abstractmethod
    def adapted_mesh(self):
        pass

    @abc.abstractmethod
    def interpolate(self, f):
        """
        Interpolate a field from the initial mesh to the adapted mesh.

        :param f: the field to be interpolated
        """
        pass


class MetricBasedAdaptor(AdaptorBase):
    """
    Class for driving metric-based mesh adaptation.
    """

    @PETSc.Log.EventDecorator("MetricBasedAdaptor.__init__")
    def __init__(self, mesh, metric):
        """
        :param mesh: :class:`MeshGeometry` to be adapted.
        :param metric: Riemannian metric :class:`Function`.
        """
        if metric._mesh is not mesh:
            raise ValueError("The mesh associated with the metric is inconsistent")
        if isinstance(mesh.topology, fmesh.ExtrudedMeshTopology):
            raise NotImplementedError("Cannot adapt extruded meshes")
        coord_fe = mesh.coordinates.ufl_element()
        if (coord_fe.family(), coord_fe.degree()) != ("Lagrange", 1):
            raise NotImplementedError(f"Mesh coordinates must be P1, not {coord_fe}")
        assert isinstance(metric, RiemannianMetric)
        super().__init__(mesh)
        self.metric = metric

    @futils.cached_property
    @PETSc.Log.EventDecorator("MetricBasedAdaptor.adapted_mesh")
    def adapted_mesh(self):
        """
        Adapt the mesh with respect to the provided metric.

        :return: a new :class:`MeshGeometry`.
        """
        self.metric.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
        metric = self.metric._reordered
        newplex = self.mesh.topology_dm.adaptMetric(metric, "Face Sets", "Cell Sets")
        return fmesh.Mesh(newplex, distribution_parameters={"partition": False})

    @PETSc.Log.EventDecorator("MetricBasedAdaptor.interpolate")
    def interpolate(self, f):
        raise NotImplementedError  # TODO: Implement consistent interpolation in parallel


def adapt(mesh, *metrics, **kwargs):
    r"""
    Adapt a mesh with respect to a metric and some adaptor parameters.

    If multiple metrics are provided, then they are intersected.

    :param mesh: :class:`MeshGeometry` to be adapted.
    :param metrics: Riemannian metric :class:`Function`\s.
    :param adaptor_parameters: parameters used to drive
        the metric-based mesh adaptation
    :return: a new :class:`MeshGeometry`.
    """
    num_metrics = len(metrics)
    metric = metrics[0]
    if num_metrics > 1:
        metric.intersect(*metrics[1:])
    adaptor = MetricBasedAdaptor(mesh, metric)
    return adaptor.adapted_mesh
