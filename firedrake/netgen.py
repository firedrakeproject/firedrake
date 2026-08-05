"""Conversion between Netgen meshes and Firedrake meshes."""
import pickle
from functools import cached_property

import numpy as np

from pyop2.mpi import COMM_WORLD, MPI
from firedrake.petsc import PETSc
from firedrake.utils import ScalarType
import firedrake

# Netgen and ngsPETSc are not available when the documentation is getting built
# because they do not have ARM wheels.
try:
    import netgen.meshing as ngm
    from netgen.meshing import MeshingParameters
    from ngsPETSc import MeshMapping, createNetgenMesh
except ImportError:
    pass

try:
    import ngsolve as ngs
except ImportError:
    class ngs:
        "dummy class"
        class comp:
            "dummy class"
            Mesh = type("_MissingNGSolveMesh", (), {})


def _coordinate_finite_element(plex: PETSc.DMPlex,
                               degree: int,
                               is_simplex: bool) -> PETSc.FE:
    """Create the PETSc finite element used for DMPlex coordinates.

    Parameters
    ----------
    plex
        The DMPlex whose dimension and coordinate dimension determine the
        element's reference cell and value shape.
    degree
        Polynomial degree of the coordinate element.
    is_simplex
        Whether ``plex`` is a simplex mesh.

    Returns
    -------
    PETSc.FE
        A Lagrange finite element of the requested degree.
    """
    prefix = "netgen_coordinate_"
    key = f"{prefix}petscspace_degree"
    options = PETSc.Options()
    previous = options.getString(key) if options.hasName(key) else None
    options[key] = degree
    try:
        element = PETSc.FE().createDefault(
            plex.getDimension(),
            plex.getCoordinateDim(),
            is_simplex,
            prefix=prefix,
            comm=plex.comm,
        )
    finally:
        if previous is None:
            del options[key]
        else:
            options[key] = previous
    return element


def _coordinate_reference_points(element: PETSc.FE,
                                 tdim: int,
                                 gdim: int) -> np.ndarray:
    """Return PETSc coordinate nodes in Netgen reference coordinates."""
    dual = element.getDualSpace()
    dimension = dual.getDimension()
    if dimension % gdim:
        raise ValueError(
            f"Coordinate dual dimension {dimension} is not divisible by "
            f"the geometric dimension {gdim}"
        )

    points = []
    for node in range(0, dimension, gdim):
        point, _ = dual.getFunctional(node).getData()
        if point.size != tdim:
            raise ValueError("Netgen coordinates require point-evaluation nodes.")
        for component in range(1, gdim):
            component_point, _ = dual.getFunctional(node + component).getData()
            if not np.array_equal(point, component_point):
                raise ValueError(
                    "Coordinate components use different dual evaluation points."
                )
        points.append(point)

    # Express the dual nodes in barycentric coordinates relative to PETSc's
    # oriented reference-cell closure.  In particular, PETSc's tetrahedron
    # does not order the coordinate axes in the same way as its triangle.
    reference_dm = dual.getDM()
    reference_cell = reference_dm.getHeightStratum(0)[0]
    vertices = reference_dm.getVecClosure(
        reference_dm.getCoordinateSection(),
        reference_dm.getCoordinatesLocal(),
        reference_cell,
    ).reshape(-1, tdim)
    points = np.asarray(points)
    coordinates = np.linalg.solve(
        (vertices[1:] - vertices[0]).T,
        (points - vertices[0]).T,
    ).T
    barycentric = np.column_stack((1 - coordinates.sum(axis=1), coordinates))

    # Netgen's element transformation associates its reference vertices
    # with the cyclically shifted connectivity order (last, first, ...,
    # penultimate).
    netgen_vertex_permutation = np.roll(np.arange(tdim + 1), 1)
    return barycentric[:, netgen_vertex_permutation][:, 1:]


def _is_simplex(plex: PETSc.DMPlex) -> bool:
    """Return whether every cell of a DMPlex, on every rank, is a simplex.

    Parameters
    ----------
    plex
        The DMPlex to test.

    Returns
    -------
    bool
        Whether the mesh is a simplex mesh. Ranks holding no cells abstain.
    """
    cell_start, cell_end = plex.getHeightStratum(0)
    return plex.comm.tompi4py().allreduce(
        cell_start == cell_end or plex.isSimplex(),
        op=MPI.LAND,
    )


def _set_netgen_coordinates(plex: PETSc.DMPlex,
                            ngmesh: object,
                            degree: int,
                            *,
                            root_only: bool = True,
                            reset: bool = False) -> None:
    """Attach Netgen-evaluated high-order coordinates to a DMPlex.

    Parameters
    ----------
    plex
        The DMPlex to curve. Its coordinate discretization is replaced by a
        Lagrange element of the requested degree.
    ngmesh
        The Netgen mesh whose element mapping evaluates the coordinates.
    degree
        Polynomial degree of the new coordinates.
    root_only
        Whether only rank 0 holds a nonempty ``ngmesh`` and ``plex``, as is
        the case before the mesh is distributed. Other ranks then contribute
        no cell coordinates.
    reset
        Whether to first reset ``plex`` to a linear coordinate
        discretization, needed when it already carries a stale
        higher-order one.
    """
    from firedrake.cython import dmcommon

    if not isinstance(degree, int) or degree < 1:
        raise ValueError("The Netgen coordinate degree must be a positive integer.")
    comm = plex.comm.tompi4py()
    is_simplex = _is_simplex(plex)
    if not is_simplex:
        raise NotImplementedError(
            "High-order Netgen coordinates currently require a simplex mesh."
        )

    tdim = plex.getDimension()
    gdim = plex.getCoordinateDim()
    if reset:
        # Refined DMs can carry the parent coordinate PetscFE alongside a
        # linear coordinate section. Restore a consistent linear coordinate
        # DM before installing the requested element.
        linear_element = _coordinate_finite_element(plex, 1, is_simplex)
        plex.setCoordinateDisc(linear_element, False, False)
    element = _coordinate_finite_element(plex, degree, is_simplex)
    points = _coordinate_reference_points(element, tdim, gdim)
    plex.setCoordinateDisc(element, False, True)

    if not root_only or comm.rank == 0:
        elements = {
            1: ngmesh.Elements1D,
            2: ngmesh.Elements2D,
            3: ngmesh.Elements3D,
        }[tdim]()
        values = np.empty((len(elements), len(points), gdim), dtype=np.float64)
        ngmesh.Curve(degree)
        ngmesh.CalcElementMapping(points, values)
        values = np.asarray(values, dtype=ScalarType)
    else:
        values = np.empty((0, len(points), gdim), dtype=ScalarType)
    dmcommon.set_cell_coordinates(plex, values)


def _linearize_coordinate_dm(plex: PETSc.DMPlex) -> None:
    """Replace a simplex coordinate discretization by its linear interpolant.

    Parameters
    ----------
    plex
        The DMPlex whose coordinate discretization is replaced in place by
        a degree-1 Lagrange element.
    """
    is_simplex = _is_simplex(plex)
    if not is_simplex:
        raise NotImplementedError(
            "Linearizing Netgen coordinates currently requires a simplex mesh."
        )
    element = _coordinate_finite_element(plex, 1, is_simplex)
    plex.setCoordinateDisc(element, False, True)


def _mesh_from_coordinate_dm(topology: object,
                             name: str,
                             degree: int) -> object:
    """Construct a Firedrake mesh from a high-order DMPlex coordinate field.

    Parameters
    ----------
    topology
        The mesh topology whose DMPlex already carries a degree-``degree``
        coordinate discretization.
    name
        Name of the returned mesh.
    degree
        Polynomial degree of the coordinate field to read from ``topology``.

    Returns
    -------
    firedrake.mesh.MeshGeometry
        A mesh built from the DMPlex's coordinates, reordered into
        Firedrake's coordinate function space layout.
    """
    import finat.ufl
    from firedrake.function import CoordinatelessFunction
    from firedrake.functionspace import FunctionSpace
    from firedrake.cython import dmcommon
    from firedrake.mesh import make_mesh_from_coordinates

    dm = topology.topology_dm
    element = finat.ufl.VectorElement(
        "Lagrange",
        topology.ufl_cell(),
        degree,
        dim=dm.getCoordinateDim(),
    )
    function_space = FunctionSpace(topology, element)
    section = function_space.dm.getDefaultSection()
    shape = (section.getStorageSize(), dm.getCoordinateDim())
    values = dmcommon.reordered_coords_high_order(dm, section, shape)
    coordinates = CoordinatelessFunction(
        function_space,
        val=values,
        name=f"{name}_coordinates",
    )
    return make_mesh_from_coordinates(coordinates, name)


def _transfer_high_order_coordinates(coarse_mesh, fine_mesh, order):
    """Transfer high-order coordinates from a Netgen geometry to a refined mesh.

    ``fine_mesh`` is a straight-edged (order 1) refinement of ``coarse_mesh``.
    This rebuilds its Netgen mesh from ``coarse_mesh``'s geometry and curves
    it to the requested ``order``, so that the curved fine mesh follows the
    same underlying CAD geometry as the coarse one, rather than just
    interpolating the coarse mesh's straight-edged coordinates.

    Parameters
    ----------
    coarse_mesh : MeshGeometry
        The coarse mesh, carrying the Netgen geometry to curve against.
    fine_mesh : MeshGeometry
        A straight-edged refinement of ``coarse_mesh``. Its Netgen attributes
        are set here, as they are required to curve it.
    order : int
        The polynomial order of the curved coordinate field.

    Returns
    -------
    MeshGeometry
        A mesh sharing ``fine_mesh``'s topology, with coordinates curved to
        ``order`` against ``coarse_mesh``'s geometry.

    """
    fine_mesh.netgen_mesh = createNetgenMesh(fine_mesh.topology_dm, coarse_mesh.netgen_mesh)
    fine_mesh.netgen_flags = getattr(coarse_mesh, "netgen_flags", {})
    cg_field = not coarse_mesh.coordinates.function_space().finat_element.is_dg()
    curved_coordinates = fine_mesh.curve_field(order=order, cg_field=cg_field)
    curved_mesh = firedrake.Mesh(curved_coordinates, name=fine_mesh.name)
    curved_mesh.netgen_mesh = fine_mesh.netgen_mesh
    curved_mesh.netgen_flags = fine_mesh.netgen_flags
    return curved_mesh


def splitToQuads(plex, dim, comm):
    """Split a Netgen mesh into quads using a PETSc transform."""
    # TODO: Improve support quad meshing.
    # @pef  Get netgen to make a quad-dominant mesh, and then only split the triangles.
    #       Current implementation will make for poor-quality meshes.
    if dim == 2:
        transform = PETSc.DMPlexTransform().create(comm=comm)
        transform.setType(PETSc.DMPlexTransformType.REFINETOBOX)
        transform.setDM(plex)
        transform.setUp()
    else:
        raise RuntimeError("Splitting to quads is only possible for 2D meshes.")
    newplex = transform.apply(plex)
    return newplex


splitTypes = {"Alfeld": lambda x: x.SplitAlfeld(),
              "Powell-Sabin": lambda x: x.SplitPowellSabin()}


class NetgenGeometry:
    """A Netgen geometry source associated with a Firedrake mesh.

    Parameters
    ----------
    mesh
        A Netgen or NGSolve mesh.
    options
        Netgen construction and coordinate options.
    user_comm
        The communicator on which to build the DMPlex.
    plex
        An existing DMPlex represented by ``mesh``.  If provided, topology
        conversion is skipped.
    """

    def __init__(self,
                 mesh: object,
                 options: dict | None,
                 user_comm=COMM_WORLD,
                 plex: PETSc.DMPlex | None = None) -> None:
        self.comm = user_comm
        self.options = dict(options) if isinstance(options, dict) else {}
        self.mesh = mesh
        self.mesh_mapping = None
        self.plex = plex
        self._mesh_is_replicated = plex is not None

        if plex is not None:
            return
        self._mesh_is_replicated = self.comm.allreduce(
            isinstance(mesh, (ngs.comp.Mesh, ngm.Mesh)),
            op=MPI.LAND,
        )
        if isinstance(mesh, ngs.comp.Mesh):
            mesh = mesh.ngmesh
        is_netgen = self.comm.bcast(
            isinstance(mesh, ngm.Mesh) if self.comm.rank == 0 else None,
            root=0,
        )
        if is_netgen:
            split2tets = self.options.get("split_to_tets", False)
            split = self.options.get("split", False)
            quad = self.options.get("quad", False)
            opt_moves = self.options.get("optimisation_moves", False)
            degree = self.options.get("degree", 1)
            if split2tets or split or quad or opt_moves:
                self._mesh_is_replicated = False
            if self.comm.rank == 0:
                if split2tets:
                    mesh = mesh.Split2Tets()
                if split:
                    # Split mesh this includes Alfeld and Powell-Sabin
                    splitTypes[split](mesh)
                if opt_moves:
                    # Optimises the mesh, for example smoothing
                    if mesh.dim == 2:
                        mesh.OptimizeMesh2d(MeshingParameters(optimize2d=opt_moves))
                    elif mesh.dim == 3:
                        mesh.OptimizeVolumeMesh(MeshingParameters(optimize3d=opt_moves))
                    else:
                        raise ValueError("Only 2D and 3D meshes can be optimised.")
            self.mesh = mesh
            self.mesh_mapping = MeshMapping(mesh, comm=self.comm)
            self.plex = self.mesh_mapping.petscPlex
            self.mesh = self.mesh_mapping.ngMesh
            if quad:
                dim = self.comm.bcast(mesh.dim if self.comm.rank == 0 else None, root=0)
                self.plex = splitToQuads(self.plex, dim, comm=self.comm)
                self.mesh_mapping = MeshMapping(self.plex)
                self.mesh = self.mesh_mapping.ngMesh
            if degree > 1:
                # The linear coordinates are already those of the Netgen mesh.
                _set_netgen_coordinates(self.plex, self.mesh, degree)
        else:
            raise ValueError("Mesh format not recognised.")

    @cached_property
    def _local_mesh(self) -> object:
        """Return a rank-local copy of the Netgen mesh when it is needed."""
        if self._mesh_is_replicated:
            return self.mesh
        serialized_mesh = self.comm.bcast(
            pickle.dumps(self.mesh) if self.comm.rank == 0 else None,
            root=0,
        )
        return self.mesh if self.comm.rank == 0 else pickle.loads(serialized_mesh)

    def curve_field(self,
                    mesh: object,
                    order: int,
                    **kwargs: object) -> object:
        """Return re-evaluated Netgen coordinates.

        Parameters
        ----------
        mesh
            The Firedrake mesh whose topology and construction options are
            reused.
        order
            Polynomial degree of the new coordinates.
        **kwargs
            Compatibility options accepted by
            :meth:`firedrake.mesh.MeshGeometry.curve_field`.

        Returns
        -------
        firedrake.Function
            The coordinate field of a rebuilt mesh.
        """
        if kwargs.get("cg_field", True) is False:
            raise NotImplementedError(
                "cg_field=False is not supported: DMPlex high-order Netgen "
                "coordinates are always continuous."
            )
        options = {
            key: value for key, value in self.options.items()
            if key not in {
                "split_to_tets", "split", "quad", "optimisation_moves",
                "degree", "permutation_tol", "cg",
            }
        }
        options["degree"] = order
        rebuilt = firedrake.Mesh(
            self.mesh,
            name=mesh.name,
            comm=mesh.comm,
            reorder=mesh._did_reordering,
            distribution_parameters=mesh._distribution_parameters,
            tolerance=mesh.tolerance,
            netgen_flags=options,
        )
        return rebuilt.coordinates

    def snap(self, plex: PETSc.DMPlex) -> None:
        """Snap the vertices of a derived DMPlex onto the geometry.

        Converting a DMPlex back to Netgen projects its boundary onto the
        geometry, so the points of the resulting Netgen mesh are the snapped
        vertices, in the order they were read out of ``plex``. A refined
        DMPlex must be snapped before it is refined again, so that each level
        is a subdivision of the level below it.

        Parameters
        ----------
        plex
            A DMPlex derived from this geometry, with linear coordinates,
            which are snapped in place.
        """
        if not _is_simplex(plex):
            # ngsPETSc only converts simplex DMPlexes back to Netgen.
            return
        ngmesh = createNetgenMesh(plex, self._local_mesh)
        coordinates = plex.getCoordinatesLocal()
        coordinates.array[:] = ngmesh.Coordinates().reshape(-1)
        plex.setCoordinatesLocal(coordinates)

    def recurve(self, fine_mesh: object, order: int) -> object:
        """Re-evaluate coordinates on a derived mesh.

        The derived mesh is converted back to Netgen and its coordinates are
        evaluated from the Netgen element mapping, which follows the geometry.

        Parameters
        ----------
        fine_mesh
            A refined Firedrake mesh with linear, and already snapped,
            DMPlex coordinates.
        order
            Polynomial degree of the new coordinates.

        Returns
        -------
        firedrake.mesh.MeshGeometry
            A mesh using the re-evaluated coordinates.
        """
        if order == 1 or not fine_mesh.ufl_cell().is_simplex:
            # Snapping has already placed the vertices on the geometry.
            fine_mesh._geometry_source = self
            return fine_mesh

        dm = fine_mesh.topology_dm
        fresh_mesh = createNetgenMesh(dm, self._local_mesh)
        _set_netgen_coordinates(
            dm, fresh_mesh, order, root_only=False, reset=True
        )
        curved_mesh = _mesh_from_coordinate_dm(
            fine_mesh.topology, fine_mesh.name, order
        )
        curved_mesh._geometry_source = type(self)(
            fresh_mesh, self.options, fine_mesh.comm, plex=dm
        )
        curved_mesh._distribution_parameters = dict(
            fine_mesh._distribution_parameters
        )
        curved_mesh._did_reordering = fine_mesh._did_reordering
        curved_mesh._tolerance = fine_mesh.tolerance
        return curved_mesh
