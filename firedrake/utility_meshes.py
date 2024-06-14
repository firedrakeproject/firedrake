import numpy as np

import ufl

from pyop2.mpi import COMM_WORLD
from firedrake.utils import IntType, RealType, ScalarType

from firedrake import (
    VectorFunctionSpace,
    Function,
    Constant,
    par_loop,
    dx,
    WRITE,
    READ,
    assemble,
    Interpolate,
    FiniteElement,
    interval,
    tetrahedron,
)
from firedrake.cython import dmcommon
from firedrake import mesh
from firedrake import function
from firedrake import functionspace
from firedrake.petsc import PETSc

from pyadjoint.tape import no_annotations


__all__ = [
    "IntervalMesh",
    "UnitIntervalMesh",
    "PeriodicIntervalMesh",
    "PeriodicUnitIntervalMesh",
    "UnitTriangleMesh",
    "RectangleMesh",
    "TensorRectangleMesh",
    "SquareMesh",
    "UnitSquareMesh",
    "PeriodicRectangleMesh",
    "PeriodicSquareMesh",
    "PeriodicUnitSquareMesh",
    "CircleManifoldMesh",
    "UnitDiskMesh",
    "UnitBallMesh",
    "UnitTetrahedronMesh",
    "TensorBoxMesh",
    "BoxMesh",
    "CubeMesh",
    "UnitCubeMesh",
    "PeriodicBoxMesh",
    "PeriodicUnitCubeMesh",
    "IcosahedralSphereMesh",
    "UnitIcosahedralSphereMesh",
    "OctahedralSphereMesh",
    "UnitOctahedralSphereMesh",
    "CubedSphereMesh",
    "UnitCubedSphereMesh",
    "TorusMesh",
    "AnnulusMesh",
    "SolidTorusMesh",
    "CylinderMesh",
]


distribution_parameters_no_overlap = {"partition": True,
                                      "overlap_type": (mesh.DistributedMeshOverlapType.NONE, 0)}
reorder_noop = False


def _postprocess_periodic_mesh(coords, comm, distribution_parameters, reorder, name, distribution_name, permutation_name):
    dm = coords.function_space().mesh().topology.topology_dm
    dm.removeLabel("pyop2_core")
    dm.removeLabel("pyop2_owned")
    dm.removeLabel("pyop2_ghost")
    dm.removeLabel("exterior_facets")
    dm.removeLabel("interior_facets")
    V = coords.function_space()
    dmcommon._set_dg_coordinates(dm,
                                 V.finat_element,
                                 V.dm.getLocalSection(),
                                 coords.dat._vec)
    return mesh.Mesh(
        dm,
        comm=comm,
        distribution_parameters=distribution_parameters,
        reorder=reorder,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def IntervalMesh(
    ncells,
    length_or_left,
    right=None,
    distribution_parameters=None,
    reorder=False,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """
    Generate a uniform mesh of an interval.

    :arg ncells: The number of the cells over the interval.
    :arg length_or_left: The length of the interval (if ``right``
         is not provided) or else the left hand boundary point.
    :arg right: (optional) position of the right
         boundary point (in which case ``length_or_left`` should
         be the left boundary point).
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    The left hand boundary point has boundary marker 1,
    while the right hand point has marker 2.
    """
    if right is None:
        left = 0
        right = length_or_left
    else:
        left = length_or_left

    if ncells <= 0 or ncells % 1:
        raise ValueError("Number of cells must be a postive integer")
    length = right - left
    if length < 0:
        raise ValueError("Requested mesh has negative length")
    dx = length / ncells
    # This ensures the rightmost point is actually present.
    coords = np.arange(left, right + 0.01 * dx, dx, dtype=np.double).reshape(-1, 1)
    cells = np.dstack(
        (
            np.arange(0, len(coords) - 1, dtype=np.int32),
            np.arange(1, len(coords), dtype=np.int32),
        )
    ).reshape(-1, 2)
    plex = mesh.plex_from_cell_list(
        1, cells, coords, comm, mesh._generate_default_mesh_topology_name(name)
    )
    # Apply boundary IDs
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    coordinates = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    vStart, vEnd = plex.getDepthStratum(0)  # vertices
    for v in range(vStart, vEnd):
        vcoord = plex.vecGetClosure(coord_sec, coordinates, v)
        if vcoord[0] == coords[0]:
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, v, 1)
        if vcoord[0] == coords[-1]:
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, v, 2)

    m = mesh.Mesh(
        plex,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    return m


@PETSc.Log.EventDecorator()
def UnitIntervalMesh(
    ncells,
    distribution_parameters=None,
    reorder=False,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """
    Generate a uniform mesh of the interval [0,1].

    :arg ncells: The number of the cells over the interval.
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=1`) point has marker 2.
    """

    return IntervalMesh(
        ncells,
        length_or_left=1.0,
        distribution_parameters=distribution_parameters,
        reorder=reorder,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def PeriodicIntervalMesh(
    ncells,
    length,
    distribution_parameters=None,
    reorder=False,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a periodic mesh of an interval.

    :arg ncells: The number of cells over the interval.
    :arg length: The length the interval.
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """

    if ncells < 3:
        raise ValueError(
            "1D periodic meshes with fewer than 3 \
cells are not currently supported"
        )
    m = CircleManifoldMesh(
        ncells,
        distribution_parameters=distribution_parameters_no_overlap,
        reorder=reorder_noop,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )
    coord_fs = VectorFunctionSpace(
        m, FiniteElement("DG", interval, 1, variant="equispaced"), dim=1
    )
    old_coordinates = m.coordinates
    new_coordinates = Function(
        coord_fs, name=mesh._generate_default_mesh_coordinates_name(name)
    )

    domain = "{ [i, j] : 0 <= i, j < 2 }"
    instructions = f"""
    <{RealType}> eps = 1e-12
    <{RealType}> pi = 3.141592653589793
    <{RealType}> oc[i, j] = real(old_coords[i, j])
    <{RealType}> a = atan2(oc[0, 1], oc[0, 0]) / (2*pi)
    <{RealType}> b = atan2(oc[1, 1], oc[1, 0]) / (2*pi)
    <{IntType}> swap = 1 if a >= b else 0
    <{RealType}> aa = fmin(a, b)
    <{RealType}> bb = fmax(a, b)
    <{RealType}> bb_abs = abs(bb)
    bb = (1.0 if aa < -eps else bb) if bb_abs < eps else bb
    aa = aa + 1 if aa < -eps else aa
    bb = bb + 1 if bb < -eps else bb
    a = bb if swap == 1 else aa
    b = aa if swap == 1 else bb
    new_coords[0] = a * L[0]
    new_coords[1] = b * L[0]
    """

    cL = Constant(length)

    par_loop(
        (domain, instructions),
        dx,
        {
            "new_coords": (new_coordinates, WRITE),
            "old_coords": (old_coordinates, READ),
            "L": (cL, READ),
        },
    )

    return _postprocess_periodic_mesh(new_coordinates,
                                      comm,
                                      distribution_parameters,
                                      reorder,
                                      name,
                                      distribution_name,
                                      permutation_name)


@PETSc.Log.EventDecorator()
def PeriodicUnitIntervalMesh(
    ncells,
    distribution_parameters=None,
    reorder=False,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a periodic mesh of the unit interval

    :arg ncells: The number of cells in the interval.
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    return PeriodicIntervalMesh(
        ncells,
        length=1.0,
        distribution_parameters=distribution_parameters,
        reorder=reorder,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def OneElementThickMesh(
    ncells,
    Lx,
    Ly,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """
    Generate a rectangular mesh in the domain with corners [0,0]
    and [Lx, Ly] with ncells, that is periodic in the x-direction.

    :arg ncells: The number of cells in the mesh.
    :arg Lx: The width of the domain in the x-direction.
    :arg Ly: The width of the domain in the y-direction.
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """

    left = np.arange(ncells, dtype=np.int32)
    right = np.roll(left, -1)
    cells = np.array([left, left, right, right]).T
    dx = Lx / ncells
    X = np.arange(1.0 * ncells, dtype=np.double) * dx
    Y = 0.0 * X
    coords = np.array([X, Y]).T

    # a line of coordinates, with a looped topology
    plex = mesh.plex_from_cell_list(
        2, cells, coords, comm, mesh._generate_default_mesh_topology_name(name)
    )
    mesh1 = mesh.Mesh(plex, distribution_parameters=distribution_parameters, comm=comm)
    mesh1.topology.init()
    cell_numbering = mesh1._cell_numbering
    cell_range = plex.getHeightStratum(0)
    cell_closure = np.zeros((cell_range[1], 9), dtype=IntType)

    # Get the coordinates for this process
    coords = plex.getCoordinatesLocal().array_r
    # get the PETSc section
    coords_sec = plex.getCoordinateSection()

    for e in range(*cell_range):

        closure, _ = plex.getTransitiveClosure(e)

        # get the row for this cell
        row = cell_numbering.getOffset(e)

        # run some checks
        assert closure[0] == e
        assert len(closure) == 6, closure
        edge_range = plex.getHeightStratum(1)
        assert all(closure[1:4] >= edge_range[0])
        assert all(closure[1:4] < edge_range[1])
        vertex_range = plex.getHeightStratum(2)
        assert all(closure[4:] >= vertex_range[0])
        assert all(closure[4:] < vertex_range[1])

        # enter the cell number
        cell_closure[row][8] = e

        # Get a list of unique edges
        edge_set = list(closure[1:4])

        # there are two vertices in the cell
        cell_vertices = closure[4:]
        cell_X = np.array([0.0, 0.0], dtype=ScalarType)
        for i, v in enumerate(cell_vertices):
            cell_X[i] = coords[coords_sec.getOffset(v)]

        # Add in the edges
        for i in range(3):
            edge_vertex, edge_vertex_ = plex.getCone(edge_set[i])
            if edge_vertex_ != edge_vertex:
                # we have a y-periodic edge
                cell_closure[row][6] = edge_set[i]
                cell_closure[row][7] = edge_set[i]
            else:
                # in this code we check if it is a right edge, or a left edge
                # by inspecting the x coordinates of the edge vertex (1)
                # and comparing with the x coordinates of the cell vertices (2)

                # there is only one vertex on the edge in this case

                # get X coordinate for this edge
                edge_X = coords[coords_sec.getOffset(edge_vertex)]
                # get X coordinates for this cell
                if cell_X.min() < dx / 2:
                    if cell_X.max() < 3 * dx / 2:
                        # We are in the first cell
                        if edge_X.min() < dx / 2:
                            # we are on left hand edge
                            cell_closure[row][4] = edge_set[i]
                        else:
                            # we are on right hand edge
                            cell_closure[row][5] = edge_set[i]
                    else:
                        # We are in the last cell
                        if edge_X.min() < dx / 2:
                            # we are on right hand edge
                            cell_closure[row][5] = edge_set[i]
                        else:
                            # we are on left hand edge
                            cell_closure[row][4] = edge_set[i]
                else:
                    if abs(cell_X.min() - edge_X.min()) < dx / 2:
                        # we are on left hand edge
                        cell_closure[row][4] = edge_set[i]
                    else:
                        # we are on right hand edge
                        cell_closure[row][5] = edge_set[i]

        # Add in the vertices
        vertices = closure[4:]
        v1 = vertices[0]
        v2 = vertices[1]
        x1 = coords[coords_sec.getOffset(v1)]
        x2 = coords[coords_sec.getOffset(v2)]
        # Fix orientations
        if x1 > x2:
            if x1 - x2 < dx * 1.5:
                # we are not on the rightmost cell and need to swap
                v1, v2 = v2, v1
        elif x2 - x1 > dx * 1.5:
            # we are on the rightmost cell and need to swap
            v1, v2 = v2, v1

        cell_closure[row][0:4] = [v1, v1, v2, v2]

    mesh1.topology.cell_closure = np.array(cell_closure, dtype=IntType)

    mesh1.init()

    fe_dg = FiniteElement("DQ", mesh1.ufl_cell(), 1, variant="equispaced")
    Vc = VectorFunctionSpace(mesh1, fe_dg)
    fc = Function(
        Vc, name=mesh._generate_default_mesh_coordinates_name(name)
    ).interpolate(mesh1.coordinates)

    mash = mesh.Mesh(
        fc,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    topverts = Vc.cell_node_list[:, 1::2].flatten()
    mash.coordinates.dat.data_with_halos[topverts, 1] = Ly

    # search for the last cell
    mcoords_ro = mash.coordinates.dat.data_ro_with_halos
    mcoords = mash.coordinates.dat.data_with_halos
    for e in range(*cell_range):
        cell = cell_numbering.getOffset(e)
        cell_nodes = Vc.cell_node_list[cell, :]
        Xvals = mcoords_ro[cell_nodes, 0]
        if Xvals.max() - Xvals.min() > Lx / 2:
            mcoords[cell_nodes[2:], 0] = Lx
        else:
            mcoords

    local_facet_dat = mash.topology.interior_facets.local_facet_dat

    lfd = local_facet_dat.data
    for i in range(lfd.shape[0]):
        if all(lfd[i, :] == np.array([3, 3])):
            lfd[i, :] = [2, 3]

    return mash


@PETSc.Log.EventDecorator()
def UnitTriangleMesh(
    refinement_level=0,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of the reference triangle

    :kwarg refinement_level: Number of uniform refinements to perform
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    cells = [[0, 1, 2]]
    plex = mesh.plex_from_cell_list(2, cells, coords, comm)

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()

    tol = 1e-2  # 0.5 would suffice
    for face in boundary_faces:
        face_coords = plex.vecGetClosure(coord_sec, coords, face)
        # |x+y-1| < eps
        if abs(face_coords[0] + face_coords[1] - 1) < tol and abs(face_coords[2] + face_coords[3] - 1) < tol:
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
        # |x| < eps
        if abs(face_coords[0]) < tol and abs(face_coords[2]) < tol:
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
        # |y| < eps
        if abs(face_coords[1]) < tol and abs(face_coords[3]) < tol:
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
    plex.removeLabel("boundary_faces")
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()

    plex.setName(mesh._generate_default_mesh_topology_name(name))
    return mesh.Mesh(
        plex,
        reorder=False,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )


@PETSc.Log.EventDecorator()
def RectangleMesh(
    nx,
    ny,
    Lx,
    Ly,
    originX=0.,
    originY=0.,
    quadrilateral=False,
    reorder=None,
    diagonal="left",
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a rectangular mesh

    :arg nx: The number of cells in the x direction.
    :arg ny: The number of cells in the y direction.
    :arg Lx: The X coordinates of the upper right corner of the rectangle.
    :arg Ly: The Y coordinates of the upper right corner of the rectangle.
    :arg originX: The X coordinates of the lower left corner of the rectangle.
    :arg originY: The Y coordinates of the lower left corner of the rectangle.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg diagonal: For triangular meshes, should the diagonal got
        from bottom left to top right (``"right"``), or top left to
        bottom right (``"left"``), or put in both diagonals (``"crossed"``).
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == originX
    * 2: plane x == Lx
    * 3: plane y == originY
    * 4: plane y == Ly
    """

    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(originX, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(originY, Ly, ny + 1, dtype=np.double)
    return TensorRectangleMesh(
        xcoords,
        ycoords,
        quadrilateral=quadrilateral,
        reorder=reorder,
        diagonal=diagonal,
        distribution_parameters=distribution_parameters,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


def TensorRectangleMesh(
    xcoords,
    ycoords,
    quadrilateral=False,
    reorder=None,
    diagonal="left",
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a rectangular mesh

    :arg xcoords: mesh points for the x direction
    :arg ycoords: mesh points for the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg diagonal: For triangular meshes, should the diagonal got
        from bottom left to top right (``"right"``), or top left to
        bottom right (``"left"``), or put in both diagonals (``"crossed"``).

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == xcoords[0]
    * 2: plane x == xcoords[-1]
    * 3: plane y == ycoords[0]
    * 4: plane y == ycoords[-1]
    """
    xcoords = np.unique(xcoords)
    ycoords = np.unique(ycoords)
    nx = np.size(xcoords) - 1
    ny = np.size(ycoords) - 1

    for n in (nx, ny):
        if n <= 0:
            raise ValueError("Number of cells must be a postive integer")

    coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
    # cell vertices
    i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))
    if not quadrilateral and diagonal == "crossed":
        xs = 0.5 * (xcoords[1:] + xcoords[:-1])
        ys = 0.5 * (ycoords[1:] + ycoords[:-1])
        extra = np.asarray(np.meshgrid(xs, ys)).swapaxes(0, 2).reshape(-1, 2)
        coords = np.vstack([coords, extra])
        #
        # 2-----3
        # | \ / |
        # |  4  |
        # | / \ |
        # 0-----1
        cells = [
            i * (ny + 1) + j,
            i * (ny + 1) + j + 1,
            (i + 1) * (ny + 1) + j,
            (i + 1) * (ny + 1) + j + 1,
            (nx + 1) * (ny + 1) + i * ny + j,
        ]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 5)
        idx = [0, 1, 4, 0, 2, 4, 2, 3, 4, 3, 1, 4]
        cells = cells[:, idx].reshape(-1, 3)
    else:
        cells = [
            i * (ny + 1) + j,
            i * (ny + 1) + j + 1,
            (i + 1) * (ny + 1) + j + 1,
            (i + 1) * (ny + 1) + j,
        ]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
        if not quadrilateral:
            if diagonal == "left":
                idx = [0, 1, 3, 1, 2, 3]
            elif diagonal == "right":
                idx = [0, 1, 2, 0, 2, 3]
            else:
                raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
            # two cells per cell above...
            cells = cells[:, idx].reshape(-1, 3)

    plex = mesh.plex_from_cell_list(
        2, cells, coords, comm, mesh._generate_default_mesh_topology_name(name)
    )

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = 0.5 * min(xcoords[1] - xcoords[0], xcoords[-1] - xcoords[-2])
        ytol = 0.5 * min(ycoords[1] - ycoords[0], ycoords[-1] - ycoords[-2])
        x0 = xcoords[0]
        x1 = xcoords[-1]
        y0 = ycoords[0]
        y1 = ycoords[-1]
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[0] - x0) < xtol and abs(face_coords[2] - x0) < xtol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            if abs(face_coords[0] - x1) < xtol and abs(face_coords[2] - x1) < xtol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
            if abs(face_coords[1] - y0) < ytol and abs(face_coords[3] - y0) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
            if abs(face_coords[1] - y1) < ytol and abs(face_coords[3] - y1) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 4)
    plex.removeLabel("boundary_faces")
    m = mesh.Mesh(
        plex,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    return m


@PETSc.Log.EventDecorator()
def SquareMesh(
    nx,
    ny,
    L,
    reorder=None,
    quadrilateral=False,
    diagonal="left",
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg L: The extent in the x and y directions
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == L
    * 3: plane y == 0
    * 4: plane y == L
    """
    return RectangleMesh(
        nx,
        ny,
        L,
        L,
        reorder=reorder,
        quadrilateral=quadrilateral,
        diagonal=diagonal,
        distribution_parameters=distribution_parameters,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def UnitSquareMesh(
    nx,
    ny,
    reorder=None,
    diagonal="left",
    quadrilateral=False,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a unit square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    """
    return SquareMesh(
        nx,
        ny,
        1,
        reorder=reorder,
        quadrilateral=quadrilateral,
        diagonal=diagonal,
        distribution_parameters=distribution_parameters,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def PeriodicRectangleMesh(
    nx,
    ny,
    Lx,
    Ly,
    direction="both",
    quadrilateral=False,
    reorder=None,
    distribution_parameters=None,
    diagonal=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a periodic rectangular mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg direction: The direction of the periodicity, one of
        ``"both"``, ``"x"`` or ``"y"``.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: (optional), one of ``"crossed"``, ``"left"``, ``"right"``.
        Not valid for quad meshes. Only used for direction ``"x"`` or direction ``"y"``.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    If direction == "x" the boundary edges in this mesh are numbered as follows:

    * 1: plane y == 0
    * 2: plane y == Ly

    If direction == "y" the boundary edges are:

    * 1: plane x == 0
    * 2: plane x == Lx
    """

    if direction == "both" and ny == 1 and quadrilateral:
        return OneElementThickMesh(
            nx,
            Lx,
            Ly,
            distribution_parameters=distribution_parameters,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
            comm=comm,
        )

    if direction not in ("both", "x", "y"):
        raise ValueError(
            "Cannot have a periodic mesh with periodicity '%s'" % direction
        )
    if direction != "both":
        return PartiallyPeriodicRectangleMesh(
            nx,
            ny,
            Lx,
            Ly,
            direction=direction,
            quadrilateral=quadrilateral,
            reorder=reorder,
            distribution_parameters=distribution_parameters,
            diagonal=diagonal,
            comm=comm,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
        )
    if nx < 3 or ny < 3:
        raise ValueError(
            "2D periodic meshes with fewer than 3 cells in each direction are not currently supported"
        )

    m = TorusMesh(
        nx,
        ny,
        1.0,
        0.5,
        quadrilateral=quadrilateral,
        reorder=reorder_noop,
        distribution_parameters=distribution_parameters_no_overlap,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )
    coord_family = "DQ" if quadrilateral else "DG"
    cell = "quadrilateral" if quadrilateral else "triangle"
    coord_fs = VectorFunctionSpace(
        m, FiniteElement(coord_family, cell, 1, variant="equispaced"), dim=2
    )
    old_coordinates = m.coordinates
    new_coordinates = Function(
        coord_fs, name=mesh._generate_default_mesh_coordinates_name(name)
    )

    domain = "{[i, j, k, l]: 0 <= i, k < old_coords.dofs and 0 <= j < new_coords.dofs and 0 <= l < 3}"
    instructions = f"""
    <{RealType}> pi = 3.141592653589793
    <{RealType}> eps = 1e-12
    <{RealType}> bigeps = 1e-1
    <{RealType}> oc[k, l] = real(old_coords[k, l])
    <{RealType}> Y = 0
    <{RealType}> Z = 0
    for i
        Y = Y + oc[i, 1]
        Z = Z + oc[i, 2]
    end
    for j
        <{RealType}> phi = atan2(oc[j, 1], oc[j, 0])
        <{RealType}> theta1 = atan2(oc[j, 2], oc[j, 1] / sin(phi) - 1)
        <{RealType}> theta2 = atan2(oc[j, 2], oc[j, 0] / cos(phi) - 1)
        <{RealType}> abssin = abs(sin(phi))
        <{RealType}> theta = theta1 if abssin > bigeps else theta2
        <{RealType}> nc0 = phi / (2 * pi)
        <{RealType}> absnc = 0
        nc0 = nc0 + 1 if nc0 < -eps else nc0
        absnc = abs(nc0)
        nc0 = 1 if absnc < eps and Y < 0 else nc0
        <{RealType}> nc1 = theta / (2 * pi)
        nc1 = nc1 + 1 if nc1 < -eps else nc1
        absnc = abs(nc1)
        nc1 = 1 if absnc < eps and Z < 0 else nc1
        new_coords[j, 0] = nc0 * Lx[0]
        new_coords[j, 1] = nc1 * Ly[0]
    end
    """

    cLx = Constant(Lx)
    cLy = Constant(Ly)

    par_loop(
        (domain, instructions),
        dx,
        {
            "new_coords": (new_coordinates, WRITE),
            "old_coords": (old_coordinates, READ),
            "Lx": (cLx, READ),
            "Ly": (cLy, READ),
        },
    )

    return _postprocess_periodic_mesh(new_coordinates,
                                      comm,
                                      distribution_parameters,
                                      reorder,
                                      name,
                                      distribution_name,
                                      permutation_name)


@PETSc.Log.EventDecorator()
def PeriodicSquareMesh(
    nx,
    ny,
    L,
    direction="both",
    quadrilateral=False,
    reorder=None,
    distribution_parameters=None,
    diagonal=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a periodic square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg L: The extent in the x and y directions
    :arg direction: The direction of the periodicity, one of
        ``"both"``, ``"x"`` or ``"y"``.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: (optional), one of ``"crossed"``, ``"left"``, ``"right"``.
        Not valid for quad meshes.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    If direction == "x" the boundary edges in this mesh are numbered as follows:

    * 1: plane y == 0
    * 2: plane y == L

    If direction == "y" the boundary edges are:

    * 1: plane x == 0
    * 2: plane x == L
    """
    return PeriodicRectangleMesh(
        nx,
        ny,
        L,
        L,
        direction=direction,
        quadrilateral=quadrilateral,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        diagonal=diagonal,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def PeriodicUnitSquareMesh(
    nx,
    ny,
    direction="both",
    reorder=None,
    quadrilateral=False,
    distribution_parameters=None,
    diagonal=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a periodic unit square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg direction: The direction of the periodicity, one of
        ``"both"``, ``"x"`` or ``"y"``.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: (optional), one of ``"crossed"``, ``"left"``, ``"right"``.
        Not valid for quad meshes.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    If direction == "x" the boundary edges in this mesh are numbered as follows:

    * 1: plane y == 0
    * 2: plane y == 1

    If direction == "y" the boundary edges are:

    * 1: plane x == 0
    * 2: plane x == 1
    """
    return PeriodicSquareMesh(
        nx,
        ny,
        1.0,
        direction=direction,
        reorder=reorder,
        quadrilateral=quadrilateral,
        distribution_parameters=distribution_parameters,
        diagonal=diagonal,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def CircleManifoldMesh(
    ncells,
    radius=1,
    degree=1,
    distribution_parameters=None,
    reorder=False,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generated a 1D mesh of the circle, immersed in 2D.

    :arg ncells: number of cells the circle should be
         divided into (min 3)
    :kwarg radius: (optional) radius of the circle to approximate.
    :kwarg degree: polynomial degree of coordinate space (e.g.,
           cells are straight line segments if degree=1).
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    if ncells < 3:
        raise ValueError("CircleManifoldMesh must have at least three cells")

    vertices = radius * np.column_stack(
        (
            np.cos(np.arange(ncells, dtype=np.double) * (2 * np.pi / ncells)),
            np.sin(np.arange(ncells, dtype=np.double) * (2 * np.pi / ncells)),
        )
    )

    cells = np.column_stack(
        (
            np.arange(0, ncells, dtype=np.int32),
            np.roll(np.arange(0, ncells, dtype=np.int32), -1),
        )
    )

    plex = mesh.plex_from_cell_list(
        1, cells, vertices, comm, mesh._generate_default_mesh_topology_name(name)
    )
    m = mesh.Mesh(
        plex,
        dim=2,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    if degree > 1:
        new_coords = function.Function(
            functionspace.VectorFunctionSpace(m, "CG", degree)
        )
        new_coords.interpolate(ufl.SpatialCoordinate(m))
        # "push out" to circle
        new_coords.dat.data[:] *= (
            radius / np.linalg.norm(new_coords.dat.data, axis=1)
        ).reshape(-1, 1)
        m = mesh.Mesh(
            new_coords,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
            comm=comm,
        )
    m._radius = radius
    return m


@PETSc.Log.EventDecorator()
def UnitDiskMesh(
    refinement_level=0,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of the unit disk in 2D

    :kwarg refinement_level: optional number of refinements (0 is a diamond)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    vertices = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]],
        dtype=np.double,
    )

    cells = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 7],
            [0, 7, 8],
            [0, 8, 1],
        ],
        np.int32,
    )

    plex = mesh.plex_from_cell_list(
        2, cells, vertices, comm, mesh._generate_default_mesh_topology_name(name)
    )

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        for face in boundary_faces:
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
    plex.removeLabel("boundary_faces")
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()

    coords = plex.getCoordinatesLocal().array.reshape(-1, 2)
    for x in coords:
        norm = np.sqrt(np.dot(x, x))
        if norm > 1.0 / (1 << (refinement_level + 1)):
            t = np.max(np.abs(x)) / norm
            x[:] *= t

    m = mesh.Mesh(
        plex,
        dim=2,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    return m


@PETSc.Log.EventDecorator()
def UnitBallMesh(
    refinement_level=0,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of the unit ball in 3D

    :kwarg refinement_level: optional number of refinements (0 is an octahedron)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional MPI communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
        dtype=np.double,
    )

    cells = np.array(
        [
            [0, 1, 2, 3],
            [0, 2, 4, 3],
            [0, 4, 5, 3],
            [0, 5, 1, 3],
            [0, 2, 1, 6],
            [0, 4, 2, 6],
            [0, 5, 4, 6],
            [0, 1, 5, 6],
        ],
        np.int32,
    )

    plex = mesh.plex_from_cell_list(
        3, cells, vertices, comm, mesh._generate_default_mesh_topology_name(name)
    )

    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        for face in boundary_faces:
            plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
    plex.removeLabel("boundary_faces")
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()

    coords = plex.getCoordinatesLocal().array.reshape(-1, 3)
    for x in coords:
        norm = np.sqrt(np.dot(x, x))
        if norm > 1.0 / (1 << (refinement_level + 1)):
            t = np.sum(np.abs(x)) / norm
            x[:] *= t

    m = mesh.Mesh(
        plex,
        dim=3,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    return m


@PETSc.Log.EventDecorator()
def UnitTetrahedronMesh(
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of the reference tetrahedron.

    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cells = [[0, 1, 2, 3]]
    plex = mesh.plex_from_cell_list(
        3, cells, coords, comm, mesh._generate_default_mesh_topology_name(name)
    )
    m = mesh.Mesh(
        plex,
        reorder=False,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    return m


def TensorBoxMesh(
    xcoords,
    ycoords,
    zcoords,
    reorder=None,
    distribution_parameters=None,
    diagonal="default",
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of a 3D box.

    :arg xcoords: Location of nodes in the x direction
    :arg ycoords: Location of nodes in the y direction
    :arg zcoords: Location of nodes in the z direction
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: Two ways of cutting hexadra, should be cut into 6
        tetrahedra (``"default"``), or 5 tetrahedra thus less biased
        (``"crossed"``)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.

    The boundary surfaces are numbered as follows:

    * 1: plane x == xcoords[0]
    * 2: plane x == xcoords[-1]
    * 3: plane y == ycoords[0]
    * 4: plane y == ycoords[-1]
    * 5: plane z == zcoords[0]
    * 6: plane z == zcoords[-1]
    """
    xcoords = np.unique(xcoords)
    ycoords = np.unique(ycoords)
    zcoords = np.unique(zcoords)
    nx = np.size(xcoords)-1
    ny = np.size(ycoords)-1
    nz = np.size(zcoords)-1

    for n in (nx, ny, nz):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")
    # X moves fastest, then Y, then Z
    coords = (
        np.asarray(np.meshgrid(xcoords, ycoords, zcoords)).swapaxes(0, 3).reshape(-1, 3)
    )
    i, j, k = np.meshgrid(
        np.arange(nx, dtype=np.int32),
        np.arange(ny, dtype=np.int32),
        np.arange(nz, dtype=np.int32),
    )
    if diagonal == "default":
        v0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1) * (ny + 1)
        v5 = v1 + (nx + 1) * (ny + 1)
        v6 = v2 + (nx + 1) * (ny + 1)
        v7 = v3 + (nx + 1) * (ny + 1)

        cells = [
            [v0, v1, v3, v7],
            [v0, v1, v7, v5],
            [v0, v5, v7, v4],
            [v0, v3, v2, v7],
            [v0, v6, v4, v7],
            [v0, v2, v6, v7],
        ]
        cells = np.asarray(cells).reshape(-1, ny, nx, nz).swapaxes(0, 3).reshape(-1, 4)
    elif diagonal == "crossed":
        v0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1) * (ny + 1)
        v5 = v1 + (nx + 1) * (ny + 1)
        v6 = v2 + (nx + 1) * (ny + 1)
        v7 = v3 + (nx + 1) * (ny + 1)

        # There are only five tetrahedra in this cutting of hexahedra
        cells = [
            [v0, v1, v2, v4],
            [v1, v7, v5, v4],
            [v1, v2, v3, v7],
            [v2, v4, v6, v7],
            [v1, v2, v7, v4],
        ]
        cells = np.asarray(cells).reshape(-1, ny, nx, nz).swapaxes(0, 3).reshape(-1, 4)
        raise NotImplementedError(
            "The crossed cutting of hexahedra has a broken connectivity issue for Pk (k>1) elements"
        )
    else:
        raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
    plex = mesh.plex_from_cell_list(
        3, cells, coords, comm, mesh._generate_default_mesh_topology_name(name)
    )
    nvert = 3  # num. vertices on facet

    # Apply boundary IDs
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    cdim = plex.getCoordinateDim()
    assert cdim == 3
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = 0.5 * min(xcoords[1]-xcoords[0], xcoords[-1] - xcoords[-2])
        ytol = 0.5 * min(ycoords[1]-ycoords[0], ycoords[-1] - ycoords[-2])
        ztol = 0.5 * min(zcoords[1]-zcoords[0], zcoords[-1] - zcoords[-2])
        x0 = xcoords[0]
        x1 = xcoords[-1]
        y0 = ycoords[0]
        y1 = ycoords[-1]
        z0 = zcoords[0]
        z1 = zcoords[-1]

        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if all([abs(face_coords[0 + cdim * i] - x0) < xtol for i in range(nvert)]):
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            if all([abs(face_coords[0 + cdim * i] - x1) < xtol for i in range(nvert)]):
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
            if all([abs(face_coords[1 + cdim * i] - y0) < ytol for i in range(nvert)]):
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
            if all([abs(face_coords[1 + cdim * i] - y1) < ytol for i in range(nvert)]):
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 4)
            if all([abs(face_coords[2 + cdim * i] - z0) < ztol for i in range(nvert)]):
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 5)
            if all([abs(face_coords[2 + cdim * i] - z1) < ztol for i in range(nvert)]):
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 6)
    plex.removeLabel("boundary_faces")
    m = mesh.Mesh(
        plex,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    return m


@PETSc.Log.EventDecorator()
def BoxMesh(
    nx,
    ny,
    nz,
    Lx,
    Ly,
    Lz,
    hexahedral=False,
    reorder=None,
    distribution_parameters=None,
    diagonal="default",
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of a 3D box.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :kwarg hexahedral: (optional), creates hexahedral mesh.
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: Two ways of cutting hexadra, should be cut into 6
        tetrahedra (``"default"``), or 5 tetrahedra thus less biased
        (``"crossed"``)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == Lx
    * 3: plane y == 0
    * 4: plane y == Ly
    * 5: plane z == 0
    * 6: plane z == Lz
    """
    for n in (nx, ny, nz):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")
    if hexahedral:
        plex = PETSc.DMPlex().createBoxMesh((nx, ny, nz), lower=(0., 0., 0.), upper=(Lx, Ly, Lz), simplex=False, periodic=False, interpolate=True, comm=comm)
        plex.removeLabel(dmcommon.FACE_SETS_LABEL)
        nvert = 4  # num. vertices on faect

        # Apply boundary IDs
        plex.createLabel(dmcommon.FACE_SETS_LABEL)
        plex.markBoundaryFaces("boundary_faces")
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()
        cdim = plex.getCoordinateDim()
        assert cdim == 3
        if plex.getStratumSize("boundary_faces", 1) > 0:
            boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
            xtol = Lx / (2 * nx)
            ytol = Ly / (2 * ny)
            ztol = Lz / (2 * nz)
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if all([abs(face_coords[0 + cdim * i]) < xtol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
                if all([abs(face_coords[0 + cdim * i] - Lx) < xtol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
                if all([abs(face_coords[1 + cdim * i]) < ytol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
                if all([abs(face_coords[1 + cdim * i] - Ly) < ytol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 4)
                if all([abs(face_coords[2 + cdim * i]) < ztol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 5)
                if all([abs(face_coords[2 + cdim * i] - Lz) < ztol for i in range(nvert)]):
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 6)
        plex.removeLabel("boundary_faces")
        m = mesh.Mesh(
            plex,
            reorder=reorder,
            distribution_parameters=distribution_parameters,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
            comm=comm,
        )
        return m
    else:
        xcoords = np.linspace(0, Lx, nx + 1, dtype=np.double)
        ycoords = np.linspace(0, Ly, ny + 1, dtype=np.double)
        zcoords = np.linspace(0, Lz, nz + 1, dtype=np.double)
        return TensorBoxMesh(
            xcoords,
            ycoords,
            zcoords,
            reorder=reorder,
            distribution_parameters=distribution_parameters,
            diagonal=diagonal,
            comm=comm,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
        )


@PETSc.Log.EventDecorator()
def CubeMesh(
    nx,
    ny,
    nz,
    L,
    hexahedral=False,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of a cube

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg L: The extent in the x, y and z directions
    :kwarg hexahedral: (optional), creates hexahedral mesh.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == L
    * 3: plane y == 0
    * 4: plane y == L
    * 5: plane z == 0
    * 6: plane z == L
    """
    return BoxMesh(
        nx,
        ny,
        nz,
        L,
        L,
        L,
        hexahedral=hexahedral,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def UnitCubeMesh(
    nx,
    ny,
    nz,
    hexahedral=False,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a mesh of a unit cube

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :kwarg hexahedral: (optional), creates hexahedral mesh.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    * 5: plane z == 0
    * 6: plane z == 1
    """
    return CubeMesh(
        nx,
        ny,
        nz,
        1,
        hexahedral=hexahedral,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def PeriodicBoxMesh(
    nx,
    ny,
    nz,
    Lx,
    Ly,
    Lz,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a periodic mesh of a 3D box.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    for n in (nx, ny, nz):
        if n < 3:
            raise ValueError(
                "3D periodic meshes with fewer than 3 cells are not currently supported"
            )

    xcoords = np.arange(0.0, Lx, Lx / nx, dtype=np.double)
    ycoords = np.arange(0.0, Ly, Ly / ny, dtype=np.double)
    zcoords = np.arange(0.0, Lz, Lz / nz, dtype=np.double)
    coords = (
        np.asarray(np.meshgrid(xcoords, ycoords, zcoords)).swapaxes(0, 3).reshape(-1, 3)
    )
    i, j, k = np.meshgrid(
        np.arange(nx, dtype=np.int32),
        np.arange(ny, dtype=np.int32),
        np.arange(nz, dtype=np.int32),
    )
    v0 = k * nx * ny + j * nx + i
    v1 = k * nx * ny + j * nx + (i + 1) % nx
    v2 = k * nx * ny + ((j + 1) % ny) * nx + i
    v3 = k * nx * ny + ((j + 1) % ny) * nx + (i + 1) % nx
    v4 = ((k + 1) % nz) * nx * ny + j * nx + i
    v5 = ((k + 1) % nz) * nx * ny + j * nx + (i + 1) % nx
    v6 = ((k + 1) % nz) * nx * ny + ((j + 1) % ny) * nx + i
    v7 = ((k + 1) % nz) * nx * ny + ((j + 1) % ny) * nx + (i + 1) % nx

    cells = [
        [v0, v1, v3, v7],
        [v0, v1, v7, v5],
        [v0, v5, v7, v4],
        [v0, v3, v2, v7],
        [v0, v6, v4, v7],
        [v0, v2, v6, v7],
    ]
    cells = np.asarray(cells).reshape(-1, ny, nx, nz).swapaxes(0, 3).reshape(-1, 4)
    plex = mesh.plex_from_cell_list(
        3, cells, coords, comm, mesh._generate_default_mesh_topology_name(name)
    )
    m = mesh.Mesh(
        plex,
        reorder=reorder_noop,
        distribution_parameters=distribution_parameters_no_overlap,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )

    old_coordinates = m.coordinates
    new_coordinates = Function(
        VectorFunctionSpace(
            m, FiniteElement("DG", tetrahedron, 1, variant="equispaced")
        ),
        name=mesh._generate_default_mesh_coordinates_name(name),
    )

    domain = ""
    instructions = f"""
    <{RealType}> x0 = real(old_coords[0, 0])
    <{RealType}> x1 = real(old_coords[1, 0])
    <{RealType}> x2 = real(old_coords[2, 0])
    <{RealType}> x3 = real(old_coords[3, 0])
    <{RealType}> x_max = fmax(fmax(fmax(x0, x1), x2), x3)
    <{RealType}> y0 = real(old_coords[0, 1])
    <{RealType}> y1 = real(old_coords[1, 1])
    <{RealType}> y2 = real(old_coords[2, 1])
    <{RealType}> y3 = real(old_coords[3, 1])
    <{RealType}> y_max = fmax(fmax(fmax(y0, y1), y2), y3)
    <{RealType}> z0 = real(old_coords[0, 2])
    <{RealType}> z1 = real(old_coords[1, 2])
    <{RealType}> z2 = real(old_coords[2, 2])
    <{RealType}> z3 = real(old_coords[3, 2])
    <{RealType}> z_max = fmax(fmax(fmax(z0, z1), z2), z3)

    new_coords[0, 0] = x_max+hx[0]  if (x_max > real(1.5*hx[0]) and old_coords[0, 0] == 0.) else old_coords[0, 0]
    new_coords[0, 1] = y_max+hy[0]  if (y_max > real(1.5*hy[0]) and old_coords[0, 1] == 0.) else old_coords[0, 1]
    new_coords[0, 2] = z_max+hz[0]  if (z_max > real(1.5*hz[0]) and old_coords[0, 2] == 0.) else old_coords[0, 2]

    new_coords[1, 0] = x_max+hx[0]  if (x_max > real(1.5*hx[0]) and old_coords[1, 0] == 0.) else old_coords[1, 0]
    new_coords[1, 1] = y_max+hy[0]  if (y_max > real(1.5*hy[0]) and old_coords[1, 1] == 0.) else old_coords[1, 1]
    new_coords[1, 2] = z_max+hz[0]  if (z_max > real(1.5*hz[0]) and old_coords[1, 2] == 0.) else old_coords[1, 2]

    new_coords[2, 0] = x_max+hx[0]  if (x_max > real(1.5*hx[0]) and old_coords[2, 0] == 0.) else old_coords[2, 0]
    new_coords[2, 1] = y_max+hy[0]  if (y_max > real(1.5*hy[0]) and old_coords[2, 1] == 0.) else old_coords[2, 1]
    new_coords[2, 2] = z_max+hz[0]  if (z_max > real(1.5*hz[0]) and old_coords[2, 2] == 0.) else old_coords[2, 2]

    new_coords[3, 0] = x_max+hx[0]  if (x_max > real(1.5*hx[0]) and old_coords[3, 0] == 0.) else old_coords[3, 0]
    new_coords[3, 1] = y_max+hy[0]  if (y_max > real(1.5*hy[0]) and old_coords[3, 1] == 0.) else old_coords[3, 1]
    new_coords[3, 2] = z_max+hz[0]  if (z_max > real(1.5*hz[0]) and old_coords[3, 2] == 0.) else old_coords[3, 2]
    """
    hx = Constant(Lx / nx)
    hy = Constant(Ly / ny)
    hz = Constant(Lz / nz)

    par_loop(
        (domain, instructions),
        dx,
        {
            "new_coords": (new_coordinates, WRITE),
            "old_coords": (old_coordinates, READ),
            "hx": (hx, READ),
            "hy": (hy, READ),
            "hz": (hz, READ),
        },
    )
    return _postprocess_periodic_mesh(new_coordinates,
                                      comm,
                                      distribution_parameters,
                                      reorder,
                                      name,
                                      distribution_name,
                                      permutation_name)


@PETSc.Log.EventDecorator()
def PeriodicUnitCubeMesh(
    nx,
    ny,
    nz,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a periodic mesh of a unit cube

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    return PeriodicBoxMesh(
        nx,
        ny,
        nz,
        1.0,
        1.0,
        1.0,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def IcosahedralSphereMesh(
    radius,
    refinement_level=0,
    degree=1,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate an icosahedral approximation to the surface of the
    sphere.

    :arg radius: The radius of the sphere to approximate.
         For a radius R the edge length of the underlying
         icosahedron will be.

         .. math::

             a = \\frac{R}{\\sin(2 \\pi / 5)}

    :kwarg refinement_level: optional number of refinements (0 is an
        icosahedron).
    :kwarg degree: polynomial degree of coordinate space (e.g.,
           flat triangles if degree=1).
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    if refinement_level < 0 or refinement_level % 1:
        raise RuntimeError("Number of refinements must be a non-negative integer")

    if degree < 1:
        raise ValueError("Mesh coordinate degree must be at least 1")
    from math import sqrt

    phi = (1 + sqrt(5)) / 2
    # vertices of an icosahedron with an edge length of 2
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.double,
    )
    # faces of the base icosahedron
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    plex = mesh.plex_from_cell_list(2, faces, vertices, comm)
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()
    plex.setName(mesh._generate_default_mesh_topology_name(name))

    coords = plex.getCoordinatesLocal().array.reshape(-1, 3)
    scale = (radius / np.linalg.norm(coords, axis=1)).reshape(-1, 1)
    coords *= scale
    m = mesh.Mesh(
        plex,
        dim=3,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    if degree > 1:
        new_coords = function.Function(
            functionspace.VectorFunctionSpace(m, "CG", degree)
        )
        new_coords.interpolate(ufl.SpatialCoordinate(m))
        # "push out" to sphere
        new_coords.dat.data[:] *= (
            radius / np.linalg.norm(new_coords.dat.data, axis=1)
        ).reshape(-1, 1)
        m = mesh.Mesh(
            new_coords,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
            comm=comm,
        )
    m._radius = radius
    return m


@PETSc.Log.EventDecorator()
def UnitIcosahedralSphereMesh(
    refinement_level=0,
    degree=1,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate an icosahedral approximation to the unit sphere.

    :kwarg refinement_level: optional number of refinements (0 is an
        icosahedron).
    :kwarg degree: polynomial degree of coordinate space (e.g.,
           flat triangles if degree=1).
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    return IcosahedralSphereMesh(
        1.0,
        refinement_level=refinement_level,
        degree=degree,
        reorder=reorder,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


# mesh is mainly used as a utility, so it's unnecessary to annotate the construction
# in this case.
@PETSc.Log.EventDecorator()
@no_annotations
def OctahedralSphereMesh(
    radius,
    refinement_level=0,
    degree=1,
    hemisphere="both",
    z0=0.8,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate an octahedral approximation to the surface of the
    sphere.

    :arg radius: The radius of the sphere to approximate.
    :kwarg refinement_level: optional number of refinements (0 is an
        octahedron).
    :kwarg degree: polynomial degree of coordinate space (e.g.,
           flat triangles if degree=1).
    :kwarg hemisphere: One of "both", "north", or "south"
    :kwarg z0: for abs(z/R)>z0, blend from a mesh where the higher-order
        non-vertex nodes are on lines of latitude to a mesh where these nodes
        are just pushed out radially from the equivalent P1 mesh.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    if refinement_level < 0 or refinement_level % 1:
        raise ValueError("Number of refinements must be a non-negative integer")

    if degree < 1:
        raise ValueError("Mesh coordinate degree must be at least 1")
    if hemisphere not in {"both", "north", "south"}:
        raise ValueError("Unhandled hemisphere '%s'" % hemisphere)
    # vertices of an octahedron of radius 1
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 5],
            [0, 2, 4],
            [0, 4, 5],
            [1, 2, 3],
            [1, 3, 5],
            [2, 3, 4],
            [3, 4, 5],
        ],
        dtype=IntType,
    )
    if hemisphere == "north":
        vertices = vertices[[0, 1, 2, 3, 4], ...]
        faces = faces[0::2, ...]
    elif hemisphere == "south":
        indices = [0, 1, 3, 4, 5]
        vertices = vertices[indices, ...]
        faces = faces[1::2, ...]
        for new, idx in enumerate(indices):
            faces[faces == idx] = new

    plex = mesh.plex_from_cell_list(2, faces, vertices, comm)
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()
    plex.setName(mesh._generate_default_mesh_topology_name(name))

    # build the initial mesh
    m = mesh.Mesh(
        plex,
        dim=3,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    if degree > 1:
        # use it to build a higher-order mesh
        m = assemble(Interpolate(ufl.SpatialCoordinate(m), VectorFunctionSpace(m, "CG", degree)))
        m = mesh.Mesh(
            m,
            name=name,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
            comm=comm,
        )

    # remap to a cone
    x, y, z = ufl.SpatialCoordinate(m)
    # This will DTWT on meshes with more than 26 refinement levels.
    # (log_2 1e8 ~= 26.5)
    tol = ufl.real(Constant(1.0e-8))
    rnew = ufl.max_value(1 - abs(z), 0)
    # Avoid division by zero (when rnew is zero, x & y are also zero)
    x0 = ufl.conditional(ufl.lt(ufl.real(rnew), tol), 0, x / rnew)
    y0 = ufl.conditional(ufl.lt(rnew, tol), 0, y / rnew)
    theta = ufl.conditional(
        ufl.ge(ufl.real(y0), 0), ufl.pi / 2 * (1 - x0), ufl.pi / 2.0 * (x0 - 1)
    )
    m.coordinates.interpolate(
        ufl.as_vector([ufl.cos(theta) * rnew, ufl.sin(theta) * rnew, z])
    )

    # push out to a sphere
    phi = ufl.pi * z / 2
    # Avoid division by zero (when rnew is zero, phi is pi/2, so cos(phi) is zero).
    scale = ufl.conditional(
        ufl.lt(ufl.real(rnew), ufl.real(tol)), 0, ufl.cos(phi) / rnew
    )
    znew = ufl.sin(phi)
    # Make a copy of the coordinates so that we can blend two different
    # mappings near the pole
    Vc = m.coordinates.function_space()
    Xlatitudinal = assemble(Interpolate(
        Constant(radius) * ufl.as_vector([x * scale, y * scale, znew]), Vc
    ))
    Vlow = VectorFunctionSpace(m, "CG", 1)
    Xlow = assemble(Interpolate(Xlatitudinal, Vlow))
    r = ufl.sqrt(Xlow[0] ** 2 + Xlow[1] ** 2 + Xlow[2] ** 2)
    Xradial = Constant(radius) * Xlow / r

    s = ufl.real(abs(z) - z0) / (1 - z0)
    exp = ufl.exp
    taper = ufl.conditional(
        ufl.gt(s, 1.0 - tol),
        1.0,
        ufl.conditional(
            ufl.gt(s, tol), exp(-1.0 / s) / (exp(-1.0 / s) + exp(-1.0 / (1.0 - s))), 0.0
        ),
    )
    m.coordinates.interpolate(taper * Xradial + (1 - taper) * Xlatitudinal)
    m._radius = radius
    return m


@PETSc.Log.EventDecorator()
def UnitOctahedralSphereMesh(
    refinement_level=0,
    degree=1,
    hemisphere="both",
    z0=0.8,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate an octahedral approximation to the unit sphere.

    :kwarg refinement_level: optional number of refinements (0 is an
        octahedron).
    :kwarg degree: polynomial degree of coordinate space (e.g.,
           flat triangles if degree=1).
    :kwarg hemisphere: One of "both", "north", or "south"
    :kwarg z0: for abs(z)>z0, blend from a mesh where the higher-order
        non-vertex nodes are on lines of latitude to a mesh where these nodes
        are just pushed out radially from the equivalent P1 mesh.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    return OctahedralSphereMesh(
        1.0,
        refinement_level=refinement_level,
        degree=degree,
        hemisphere=hemisphere,
        z0=z0,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


def _cubedsphere_cells_and_coords(radius, refinement_level):
    """Generate vertex and face lists for cubed sphere"""
    # We build the mesh out of 6 panels of the cube
    # this allows to build the gnonomic cube transformation
    # which is defined separately for each panel

    # Start by making a grid of local coordinates which we use
    # to map to each panel of the cubed sphere under the gnonomic
    # transformation
    dtheta = 2 ** (-refinement_level + 1) * np.arctan(1.0)
    a = 3.0 ** (-0.5) * radius
    theta = np.arange(np.arctan(-1.0), np.arctan(1.0) + dtheta, dtheta, dtype=np.double)
    x = a * np.tan(theta)
    Nx = x.size

    # Compute panel numberings for each panel
    # We use the following "flatpack" arrangement of panels
    #   3
    #  102
    #   4
    #   5

    # 0 is the bottom of the cube, 5 is the top.
    # All panels are numbered from left to right, top to bottom
    # according to this diagram.

    panel_numbering = np.zeros((6, Nx, Nx), dtype=np.int32)

    # Numbering for panel 0
    panel_numbering[0, :, :] = np.arange(Nx**2, dtype=np.int32).reshape(Nx, Nx)
    count = panel_numbering.max() + 1

    # Numbering for panel 5
    panel_numbering[5, :, :] = count + np.arange(Nx**2, dtype=np.int32).reshape(
        Nx, Nx
    )
    count = panel_numbering.max() + 1

    # Numbering for panel 4 - shares top edge with 0 and bottom edge
    #                         with 5
    # interior numbering
    panel_numbering[4, 1:-1, :] = count + np.arange(
        Nx * (Nx - 2), dtype=np.int32
    ).reshape(Nx - 2, Nx)

    # bottom edge
    panel_numbering[4, 0, :] = panel_numbering[5, -1, :]
    # top edge
    panel_numbering[4, -1, :] = panel_numbering[0, 0, :]
    count = panel_numbering.max() + 1

    # Numbering for panel 3 - shares top edge with 5 and bottom edge
    #                         with 0
    # interior numbering
    panel_numbering[3, 1:-1, :] = count + np.arange(
        Nx * (Nx - 2), dtype=np.int32
    ).reshape(Nx - 2, Nx)
    # bottom edge
    panel_numbering[3, 0, :] = panel_numbering[0, -1, :]
    # top edge
    panel_numbering[3, -1, :] = panel_numbering[5, 0, :]
    count = panel_numbering.max() + 1

    # Numbering for panel 1
    # interior numbering
    panel_numbering[1, 1:-1, 1:-1] = count + np.arange(
        (Nx - 2) ** 2, dtype=np.int32
    ).reshape(Nx - 2, Nx - 2)
    # left edge of 1 is left edge of 5 (inverted)
    panel_numbering[1, :, 0] = panel_numbering[5, ::-1, 0]
    # right edge of 1 is left edge of 0
    panel_numbering[1, :, -1] = panel_numbering[0, :, 0]
    # top edge (excluding vertices) of 1 is left edge of 3 (downwards)
    panel_numbering[1, -1, 1:-1] = panel_numbering[3, -2:0:-1, 0]
    # bottom edge (excluding vertices) of 1 is left edge of 4
    panel_numbering[1, 0, 1:-1] = panel_numbering[4, 1:-1, 0]
    count = panel_numbering.max() + 1

    # Numbering for panel 2
    # interior numbering
    panel_numbering[2, 1:-1, 1:-1] = count + np.arange(
        (Nx - 2) ** 2, dtype=np.int32
    ).reshape(Nx - 2, Nx - 2)
    # left edge of 2 is right edge of 0
    panel_numbering[2, :, 0] = panel_numbering[0, :, -1]
    # right edge of 2 is right edge of 5 (inverted)
    panel_numbering[2, :, -1] = panel_numbering[5, ::-1, -1]
    # bottom edge (excluding vertices) of 2 is right edge of 4 (downwards)
    panel_numbering[2, 0, 1:-1] = panel_numbering[4, -2:0:-1, -1]
    # top edge (excluding vertices) of 2 is right edge of 3
    panel_numbering[2, -1, 1:-1] = panel_numbering[3, 1:-1, -1]
    count = panel_numbering.max() + 1

    # That's the numbering done.

    # Set up an array for all of the mesh coordinates
    Npoints = panel_numbering.max() + 1
    coords = np.zeros((Npoints, 3), dtype=np.double)
    lX, lY = np.meshgrid(x, x)
    lX.shape = (Nx**2,)
    lY.shape = (Nx**2,)
    r = (a**2 + lX**2 + lY**2) ** 0.5

    # Now we need to compute the gnonomic transformation
    # for each of the panels
    panel_numbering.shape = (6, Nx**2)

    def coordinates_on_panel(panel_num, X, Y, Z):
        I = panel_numbering[panel_num, :]
        coords[I, 0] = radius / r * X
        coords[I, 1] = radius / r * Y
        coords[I, 2] = radius / r * Z

    coordinates_on_panel(0, lX, lY, -a)
    coordinates_on_panel(1, -a, lY, -lX)
    coordinates_on_panel(2, a, lY, lX)
    coordinates_on_panel(3, lX, a, lY)
    coordinates_on_panel(4, lX, -a, -lY)
    coordinates_on_panel(5, lX, -lY, a)

    # Now we need to build the face numbering
    # in local coordinates
    vertex_numbers = np.arange(Nx**2, dtype=np.int32).reshape(Nx, Nx)
    local_faces = np.zeros(((Nx - 1) ** 2, 4), dtype=np.int32)
    local_faces[:, 0] = vertex_numbers[:-1, :-1].reshape(-1)
    local_faces[:, 1] = vertex_numbers[1:, :-1].reshape(-1)
    local_faces[:, 2] = vertex_numbers[1:, 1:].reshape(-1)
    local_faces[:, 3] = vertex_numbers[:-1, 1:].reshape(-1)

    cells = panel_numbering[:, local_faces].reshape(-1, 4)
    return cells, coords


@PETSc.Log.EventDecorator()
def CubedSphereMesh(
    radius,
    refinement_level=0,
    degree=1,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate an cubed approximation to the surface of the
    sphere.

    :arg radius: The radius of the sphere to approximate.
    :kwarg refinement_level: optional number of refinements (0 is a cube).
    :kwarg degree: polynomial degree of coordinate space (e.g.,
           bilinear quads if degree=1).
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    if refinement_level < 0 or refinement_level % 1:
        raise RuntimeError("Number of refinements must be a non-negative integer")

    if degree < 1:
        raise ValueError("Mesh coordinate degree must be at least 1")

    cells, coords = _cubedsphere_cells_and_coords(radius, refinement_level)
    plex = mesh.plex_from_cell_list(
        2, cells, coords, comm, mesh._generate_default_mesh_topology_name(name)
    )

    m = mesh.Mesh(
        plex,
        dim=3,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )

    if degree > 1:
        new_coords = function.Function(
            functionspace.VectorFunctionSpace(m, "Q", degree)
        )
        new_coords.interpolate(ufl.SpatialCoordinate(m))
        # "push out" to sphere
        new_coords.dat.data[:] *= (
            radius / np.linalg.norm(new_coords.dat.data, axis=1)
        ).reshape(-1, 1)
        m = mesh.Mesh(
            new_coords,
            distribution_name=distribution_name,
            permutation_name=permutation_name,
            comm=comm,
        )
    m._radius = radius
    return m


@PETSc.Log.EventDecorator()
def UnitCubedSphereMesh(
    refinement_level=0,
    degree=1,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a cubed approximation to the unit sphere.

    :kwarg refinement_level: optional number of refinements (0 is a cube).
    :kwarg degree: polynomial degree of coordinate space (e.g.,
           bilinear quads if degree=1).
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """
    return CubedSphereMesh(
        1.0,
        refinement_level=refinement_level,
        degree=degree,
        reorder=reorder,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def TorusMesh(
    nR,
    nr,
    R,
    r,
    quadrilateral=False,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a toroidal mesh

    :arg nR: The number of cells in the major direction (min 3)
    :arg nr: The number of cells in the minor direction (min 3)
    :arg R: The major radius
    :arg r: The minor radius
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.
    """

    if nR < 3 or nr < 3:
        raise ValueError("Must have at least 3 cells in each direction")

    for n in (nR, nr):
        if n % 1:
            raise RuntimeError("Number of cells must be an integer")

    # gives an array [[0, 0], [0, 1], ..., [1, 0], [1, 1], ...]
    idx_temp = (
        np.asarray(np.meshgrid(np.arange(nR), np.arange(nr)))
        .swapaxes(0, 2)
        .reshape(-1, 2)
    )

    # vertices - standard formula for (x, y, z), see Wikipedia
    vertices = np.asarray(
        np.column_stack(
            (
                (R + r * np.cos(idx_temp[:, 1] * (2 * np.pi / nr)))
                * np.cos(idx_temp[:, 0] * (2 * np.pi / nR)),
                (R + r * np.cos(idx_temp[:, 1] * (2 * np.pi / nr)))
                * np.sin(idx_temp[:, 0] * (2 * np.pi / nR)),
                r * np.sin(idx_temp[:, 1] * (2 * np.pi / nr)),
            )
        ),
        dtype=np.double,
    )

    # cell vertices
    i, j = np.meshgrid(np.arange(nR, dtype=np.int32), np.arange(nr, dtype=np.int32))
    i = i.reshape(-1)  # Miklos's suggestion to make the code
    j = j.reshape(-1)  # less impenetrable
    cells = [
        i * nr + j,
        i * nr + (j + 1) % nr,
        ((i + 1) % nR) * nr + (j + 1) % nr,
        ((i + 1) % nR) * nr + j,
    ]
    cells = np.column_stack(cells)
    if not quadrilateral:
        # two cells per cell above...
        cells = cells[:, [0, 1, 3, 1, 2, 3]].reshape(-1, 3)

    plex = mesh.plex_from_cell_list(
        2, cells, vertices, comm, mesh._generate_default_mesh_topology_name(name)
    )
    m = mesh.Mesh(
        plex,
        dim=3,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )
    return m


@PETSc.Log.EventDecorator()
def AnnulusMesh(
    R,
    r,
    nr=4,
    nt=32,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate an annulus mesh periodically extruding an interval mesh

    :arg R: The outer radius
    :arg r: The inner radius
    :kwarg nr: (optional), number of cells in the radial direction
    :kwarg nt: (optional), number of cells in the circumferential direction (min 3)
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if ``None``, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if ``None``, the name is automatically
           generated.
    """
    if nt < 3:
        raise ValueError("Must have at least 3 cells in the circumferential direction")
    base_name = name + "_base"
    base = IntervalMesh(nr,
                        r,
                        right=R,
                        distribution_parameters=distribution_parameters,
                        comm=comm,
                        name=base_name,
                        distribution_name=distribution_name,
                        permutation_name=permutation_name)
    bar = mesh.ExtrudedMesh(base, layers=nt, layer_height=2 * np.pi / nt, extrusion_type="uniform", periodic=True)
    x, y = ufl.SpatialCoordinate(bar)
    V = bar.coordinates.function_space()
    coord = Function(V).interpolate(ufl.as_vector([x * ufl.cos(y), x * ufl.sin(y)]))
    annulus = mesh.make_mesh_from_coordinates(coord.topological, name)
    annulus.topology.name = mesh._generate_default_mesh_topology_name(name)
    annulus._base_mesh = base
    return annulus


@PETSc.Log.EventDecorator()
def SolidTorusMesh(
    R,
    r,
    nR=8,
    refinement_level=0,
    reorder=None,
    distribution_parameters=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generate a solid toroidal mesh (with axis z) periodically extruding a disk mesh

    :arg R: The major radius
    :arg r: The minor radius
    :kwarg nR: (optional), number of cells in the major direction (min 3)
    :kwarg refinement_level: (optional), number of times the base disk mesh is refined.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if ``None``, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if ``None``, the name is automatically
           generated.
    """
    if nR < 3:
        raise ValueError("Must have at least 3 cells in the major direction")
    base_name = name + "_base"
    unit = UnitDiskMesh(refinement_level=refinement_level,
                        reorder=reorder,
                        distribution_parameters=distribution_parameters,
                        comm=comm,
                        distribution_name=distribution_name,
                        permutation_name=permutation_name)
    x, y = ufl.SpatialCoordinate(unit)
    V = unit.coordinates.function_space()
    coord = Function(V).interpolate(ufl.as_vector([r * x + R, r * y]))
    disk = mesh.make_mesh_from_coordinates(coord.topological, base_name)
    disk.topology.name = mesh._generate_default_mesh_topology_name(base_name)
    disk.topology.topology_dm.setName(disk.topology.name)
    bar = mesh.ExtrudedMesh(disk, layers=nR, layer_height=2 * np.pi / nR, extrusion_type="uniform", periodic=True)
    x, y, z = ufl.SpatialCoordinate(bar)
    V = bar.coordinates.function_space()
    coord = Function(V).interpolate(ufl.as_vector([x * ufl.cos(z), x * ufl.sin(z), -y]))
    torus = mesh.make_mesh_from_coordinates(coord.topological, name)
    torus.topology.name = mesh._generate_default_mesh_topology_name(name)
    torus._base_mesh = disk
    return torus


@PETSc.Log.EventDecorator()
def CylinderMesh(
    nr,
    nl,
    radius=1,
    depth=1,
    longitudinal_direction="z",
    quadrilateral=False,
    reorder=None,
    distribution_parameters=None,
    diagonal=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generates a cylinder mesh.

    :arg nr: number of cells the cylinder circumference should be
         divided into (min 3)
    :arg nl: number of cells along the longitudinal axis of the cylinder
    :kwarg radius: (optional) radius of the cylinder to approximate.
    :kwarg depth: (optional) depth of the cylinder to approximate.
    :kwarg longitudinal_direction: (option) direction for the
         longitudinal axis of the cylinder.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: (optional), one of ``"crossed"``, ``"left"``, ``"right"``.
        Not valid for quad meshes.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    The boundary edges in this mesh are numbered as follows:

    * 1: plane l == 0 (bottom)
    * 2: plane l == depth (top)
    """
    if nr < 3:
        raise ValueError("CylinderMesh must have at least three cells")
    if quadrilateral and diagonal is not None:
        raise ValueError("Cannot specify slope of diagonal on quad meshes")
    if not quadrilateral and diagonal is None:
        diagonal = "left"

    coord_xy = radius * np.column_stack(
        (
            np.cos(np.arange(nr) * (2 * np.pi / nr)),
            np.sin(np.arange(nr) * (2 * np.pi / nr)),
        )
    )
    coord_z = depth * np.linspace(0.0, 1.0, nl + 1).reshape(-1, 1)
    vertices = np.asarray(
        np.column_stack(
            (np.tile(coord_xy, (nl + 1, 1)), np.tile(coord_z, (1, nr)).reshape(-1, 1))
        ),
        dtype=np.double,
    )

    # intervals on circumference
    ring_cells = np.column_stack(
        (
            np.arange(0, nr, dtype=np.int32),
            np.roll(np.arange(0, nr, dtype=np.int32), -1),
        )
    )
    # quads in the first layer
    ring_cells = np.column_stack((ring_cells, np.roll(ring_cells, 1, axis=1) + nr))

    if not quadrilateral and diagonal == "crossed":
        dxy = np.pi / nr
        Lxy = 2 * np.pi
        extra_uv = np.linspace(dxy, Lxy - dxy, nr, dtype=np.double)
        extra_xy = radius * np.column_stack((np.cos(extra_uv), np.sin(extra_uv)))
        dz = 1 * 0.5 / nl
        extra_z = depth * np.linspace(dz, 1 - dz, nl).reshape(-1, 1)
        extras = np.asarray(
            np.column_stack(
                (np.tile(extra_xy, (nl, 1)), np.tile(extra_z, (1, nr)).reshape(-1, 1))
            ),
            dtype=np.double,
        )
        origvertices = vertices
        vertices = np.vstack([vertices, extras])
        #
        # 2-----3
        # | \ / |
        # |  4  |
        # | / \ |
        # 0-----1

        offset = np.arange(nl, dtype=np.int32) * nr
        origquads = np.row_stack(tuple(ring_cells + i for i in offset))
        cells = np.zeros((origquads.shape[0] * 4, 3), dtype=np.int32)
        cellidx = 0
        newvertices = range(len(origvertices), len(origvertices) + len(extras))
        for (origquad, extravertex) in zip(origquads, newvertices):
            cells[cellidx + 0, :] = [origquad[0], origquad[1], extravertex]
            cells[cellidx + 1, :] = [origquad[0], origquad[3], extravertex]
            cells[cellidx + 2, :] = [origquad[3], origquad[2], extravertex]
            cells[cellidx + 3, :] = [origquad[2], origquad[1], extravertex]
            cellidx += 4

    else:
        offset = np.arange(nl, dtype=np.int32) * nr
        cells = np.row_stack(tuple(ring_cells + i for i in offset))
        if not quadrilateral:
            if diagonal == "left":
                idx = [0, 1, 3, 1, 2, 3]
            elif diagonal == "right":
                idx = [0, 1, 2, 0, 2, 3]
            else:
                raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
            # two cells per cell above...
            cells = cells[:, idx].reshape(-1, 3)

    if longitudinal_direction == "x":
        rotation = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.double)
        vertices = np.dot(vertices, rotation.T)
    elif longitudinal_direction == "y":
        rotation = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.double)
        vertices = np.dot(vertices, rotation.T)
    elif longitudinal_direction != "z":
        raise ValueError("Unknown longitudinal direction '%s'" % longitudinal_direction)

    plex = mesh.plex_from_cell_list(
        2, cells, vertices, comm, mesh._generate_default_mesh_topology_name(name)
    )

    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        eps = depth / (2 * nl)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            # index of x/y/z coordinates of the face element
            axis_ix = {"x": 0, "y": 1, "z": 2}
            i = axis_ix[longitudinal_direction]
            j = i + 3
            if abs(face_coords[i]) < eps and abs(face_coords[j]) < eps:
                # bottom of cylinder
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            if abs(face_coords[i] - depth) < eps and abs(face_coords[j] - depth) < eps:
                # top of cylinder
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
    plex.removeLabel("boundary_faces")

    return mesh.Mesh(
        plex,
        dim=3,
        reorder=reorder,
        distribution_parameters=distribution_parameters,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
        comm=comm,
    )


@PETSc.Log.EventDecorator()
def PartiallyPeriodicRectangleMesh(
    nx,
    ny,
    Lx,
    Ly,
    direction="x",
    quadrilateral=False,
    reorder=None,
    distribution_parameters=None,
    diagonal=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generates RectangleMesh that is periodic in the x or y direction.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg direction: The direction of the periodicity.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: (optional), one of ``"crossed"``, ``"left"``, ``"right"``.
        Not valid for quad meshes.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    If direction == "x" the boundary edges in this mesh are numbered as follows:

    * 1: plane y == 0
    * 2: plane y == Ly

    If direction == "y" the boundary edges are:

    * 1: plane x == 0
    * 2: plane x == Lx
    """

    if direction not in ("x", "y"):
        raise ValueError("Unsupported periodic direction '%s'" % direction)

    # handle x/y directions: na, La are for the periodic axis
    na, nb, La, Lb = nx, ny, Lx, Ly
    if direction == "y":
        na, nb, La, Lb = ny, nx, Ly, Lx

    if na < 3:
        raise ValueError(
            "2D periodic meshes with fewer than 3 cells in each direction are not currently supported"
        )

    m = CylinderMesh(
        na,
        nb,
        1.0,
        1.0,
        longitudinal_direction="z",
        quadrilateral=quadrilateral,
        reorder=reorder_noop,
        distribution_parameters=distribution_parameters_no_overlap,
        diagonal=diagonal,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )
    coord_family = "DQ" if quadrilateral else "DG"
    cell = "quadrilateral" if quadrilateral else "triangle"
    coord_fs = VectorFunctionSpace(
        m, FiniteElement(coord_family, cell, 1, variant="equispaced"), dim=2
    )
    old_coordinates = m.coordinates
    new_coordinates = Function(
        coord_fs, name=mesh._generate_default_mesh_coordinates_name(name)
    )

    # make x-periodic mesh
    # unravel x coordinates like in periodic interval
    # set y coordinates to z coordinates
    domain = "{[i, j, k, l]: 0 <= i, k < old_coords.dofs and 0 <= j < new_coords.dofs and 0 <= l < 3}"
    instructions = f"""
    <{RealType}> Y = 0
    <{RealType}> pi = 3.141592653589793
    <{RealType}> oc[k, l] = real(old_coords[k, l])
    for i
        Y = Y + oc[i, 1]
    end
    for j
        <{RealType}> nc0 = atan2(oc[j, 1], oc[j, 0]) / (pi* 2)
        nc0 = nc0 + 1 if nc0 < 0 else nc0
        nc0 = 1 if nc0 == 0 and Y < 0 else nc0
        new_coords[j, 0] = nc0 * Lx[0]
        new_coords[j, 1] = old_coords[j, 2] * Ly[0]
    end
    """

    cLx = Constant(La)
    cLy = Constant(Lb)

    par_loop(
        (domain, instructions),
        dx,
        {
            "new_coords": (new_coordinates, WRITE),
            "old_coords": (old_coordinates, READ),
            "Lx": (cLx, READ),
            "Ly": (cLy, READ),
        },
    )

    if direction == "y":
        # flip x and y coordinates
        operator = np.asarray([[0, 1], [1, 0]])
        new_coordinates.dat.data[:] = np.dot(new_coordinates.dat.data, operator.T)

    return _postprocess_periodic_mesh(new_coordinates,
                                      comm,
                                      distribution_parameters,
                                      reorder,
                                      name,
                                      distribution_name,
                                      permutation_name)
