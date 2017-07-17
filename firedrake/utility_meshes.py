from __future__ import absolute_import, print_function, division
import numpy as np

import ufl

from pyop2.mpi import COMM_WORLD
from pyop2.datatypes import IntType

from firedrake import VectorFunctionSpace, Function, Constant, \
    par_loop, dx, WRITE, READ
from firedrake import mesh
from firedrake import function
from firedrake import functionspace


__all__ = ['IntervalMesh', 'UnitIntervalMesh',
           'PeriodicIntervalMesh', 'PeriodicUnitIntervalMesh',
           'UnitTriangleMesh',
           'RectangleMesh', 'SquareMesh', 'UnitSquareMesh',
           'PeriodicRectangleMesh', 'PeriodicSquareMesh',
           'PeriodicUnitSquareMesh',
           'CircleManifoldMesh',
           'UnitTetrahedronMesh',
           'BoxMesh', 'CubeMesh', 'UnitCubeMesh',
           'IcosahedralSphereMesh', 'UnitIcosahedralSphereMesh',
           'CubedSphereMesh', 'UnitCubedSphereMesh',
           'TorusMesh', 'CylinderMesh']


def IntervalMesh(ncells, length_or_left, right=None, comm=COMM_WORLD):
    """
    Generate a uniform mesh of an interval.

    :arg ncells: The number of the cells over the interval.
    :arg length_or_left: The length of the interval (if ``right``
         is not provided) or else the left hand boundary point.
    :arg right: (optional) position of the right
         boundary point (in which case ``length_or_left`` should
         be the left boundary point).
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

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
    cells = np.dstack((np.arange(0, len(coords) - 1, dtype=np.int32),
                       np.arange(1, len(coords), dtype=np.int32))).reshape(-1, 2)
    plex = mesh._from_cell_list(1, cells, coords, comm)
    # Apply boundary IDs
    plex.createLabel("boundary_ids")
    coordinates = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    vStart, vEnd = plex.getDepthStratum(0)  # vertices
    for v in range(vStart, vEnd):
        vcoord = plex.vecGetClosure(coord_sec, coordinates, v)
        if vcoord[0] == coords[0]:
            plex.setLabelValue("boundary_ids", v, 1)
        if vcoord[0] == coords[-1]:
            plex.setLabelValue("boundary_ids", v, 2)

    return mesh.Mesh(plex, reorder=False)


def UnitIntervalMesh(ncells, comm=COMM_WORLD):
    """
    Generate a uniform mesh of the interval [0,1].

    :arg ncells: The number of the cells over the interval.
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=1`) point has marker 2.
    """

    return IntervalMesh(ncells, length_or_left=1.0, comm=comm)


def PeriodicIntervalMesh(ncells, length, comm=COMM_WORLD):
    """Generate a periodic mesh of an interval.

    :arg ncells: The number of cells over the interval.
    :arg length: The length the interval.
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """

    if ncells < 3:
        raise ValueError("1D periodic meshes with fewer than 3 \
cells are not currently supported")

    m = CircleManifoldMesh(ncells, comm=comm)
    coord_fs = VectorFunctionSpace(m, 'DG', 1, dim=1)
    old_coordinates = m.coordinates
    new_coordinates = Function(coord_fs)

    periodic_kernel = """
    const double pi = 3.141592653589793;
    const double eps = 1e-12;
    double a = atan2(old_coords[0][1], old_coords[0][0]) / (2*pi);
    double b = atan2(old_coords[1][1], old_coords[1][0]) / (2*pi);
    int swap = 0;
    if ( a >= b ) {
        const double tmp = b;
        b = a;
        a = tmp;
        swap = 1;
    }
    if ( fabs(b) < eps && a < -eps ) {
        b = 1.0;
    }
    if ( a < -eps ) {
        a += 1;
    }
    if ( b < -eps ) {
        b += 1;
    }
    if ( swap ) {
        const double tmp = b;
        b = a;
        a = tmp;
    }
    new_coords[0][0] = a * L[0];
    new_coords[1][0] = b * L[0];
    """

    cL = Constant(length)

    par_loop(periodic_kernel, dx,
             {"new_coords": (new_coordinates, WRITE),
              "old_coords": (old_coordinates, READ),
              "L": (cL, READ)})

    return mesh.Mesh(new_coordinates)


def PeriodicUnitIntervalMesh(ncells, comm=COMM_WORLD):
    """Generate a periodic mesh of the unit interval

    :arg ncells: The number of cells in the interval.
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    return PeriodicIntervalMesh(ncells, length=1.0, comm=comm)


def OneElementThickMesh(ncells, Lx, Ly, comm=COMM_WORLD):
    """
    Generate a rectangular mesh in the domain with corners [0,0]
    and [Lx, Ly] with ncells, that is periodic in the x-direction.

    :arg ncells: The number of cells in the mesh.
    :arg Lx: The width of the domain in the x-direction.
    :arg Ly: The width of the domain in the y-direction.
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """

    left = np.arange(ncells, dtype=np.int32)
    right = np.roll(left, -1)
    cells = np.array([left, left, right, right]).T
    dx = Lx/ncells
    X = np.arange(1.0*ncells, dtype=np.double)*dx
    Y = 0.*X
    coords = np.array([X, Y]).T

    # a line of coordinates, with a looped topology
    plex = mesh._from_cell_list(2, cells, coords, comm)
    mesh1 = mesh.Mesh(plex)
    mesh1.topology.init()
    cell_numbering = mesh1._cell_numbering
    cell_range = plex.getHeightStratum(0)
    cell_closure = np.zeros((cell_range[1], 9), dtype=IntType)

    # Get the coordinates for this process
    coords = plex.getCoordinatesLocal().array_r
    # get the PETSc section
    coords_sec = plex.getCoordinateSection()

    for e in range(*cell_range):

        closure, orient = plex.getTransitiveClosure(e)

        # get the row for this cell
        row = cell_numbering.getOffset(e)

        # run some checks
        assert(closure[0] == e)
        assert len(closure) == 7, closure
        edge_range = plex.getHeightStratum(1)
        assert(all(closure[1:5] >= edge_range[0]))
        assert(all(closure[1:5] < edge_range[1]))
        vertex_range = plex.getHeightStratum(2)
        assert(all(closure[5:] >= vertex_range[0]))
        assert(all(closure[5:] < vertex_range[1]))

        # enter the cell number
        cell_closure[row][8] = e

        # Get a list of unique edges
        edge_set = list(set(closure[1:5]))

        # there are two vertices in the cell
        cell_vertices = closure[5:]
        cell_X = np.array([0., 0.])
        for i, v in enumerate(cell_vertices):
            cell_X[i] = coords[coords_sec.getOffset(v)]

        # Add in the edges
        for i in range(3):
            # count up how many times each edge is repeated
            repeats = list(closure[1:5]).count(edge_set[i])
            if repeats == 2:
                # we have a y-periodic edge
                cell_closure[row][6] = edge_set[i]
                cell_closure[row][7] = edge_set[i]
            elif repeats == 1:
                # in this code we check if it is a right edge, or a left edge
                # by inspecting the x coordinates of the edge vertex (1)
                # and comparing with the x coordinates of the cell vertices (2)

                # there is only one vertex on the edge in this case
                edge_vertex = plex.getCone(edge_set[i])[0]

                # get X coordinate for this edge
                edge_X = coords[coords_sec.getOffset(edge_vertex)]
                # get X coordinates for this cell
                if(cell_X.min() < dx/2):
                    if cell_X.max() < 3*dx/2:
                        # We are in the first cell
                        if(edge_X.min() < dx/2):
                            # we are on left hand edge
                            cell_closure[row][4] = edge_set[i]
                        else:
                            # we are on right hand edge
                            cell_closure[row][5] = edge_set[i]
                    else:
                        # We are in the last cell
                        if(edge_X.min() < dx/2):
                            # we are on right hand edge
                            cell_closure[row][5] = edge_set[i]
                        else:
                            # we are on left hand edge
                            cell_closure[row][4] = edge_set[i]
                else:
                    if(abs(cell_X.min()-edge_X.min()) < dx/2):
                        # we are on left hand edge
                        cell_closure[row][4] = edge_set[i]
                    else:
                        # we are on right hand edge
                        cell_closure[row][5] = edge_set[i]

        # Add in the vertices
        vertices = closure[5:]
        v1 = vertices[0]
        v2 = vertices[1]
        x1 = coords[coords_sec.getOffset(v1)]
        x2 = coords[coords_sec.getOffset(v2)]
        # Fix orientations
        if(x1 > x2):
            if(x1 - x2 < dx*1.5):
                # we are not on the rightmost cell and need to swap
                v1, v2 = v2, v1
        elif(x2 - x1 > dx*1.5):
            # we are on the rightmost cell and need to swap
            v1, v2 = v2, v1

        cell_closure[row][0:4] = [v1, v1, v2, v2]

    mesh1.topology.cell_closure = np.array(cell_closure, dtype=IntType)

    mesh1.init()

    Vc = VectorFunctionSpace(mesh1, 'DQ', 1)
    fc = Function(Vc).interpolate(mesh1.coordinates)

    mash = mesh.Mesh(fc)
    topverts = Vc.cell_node_list[:, 1::2].flatten()
    mash.coordinates.dat.data_with_halos[topverts, 1] = Ly

    # search for the last cell
    mcoords_ro = mash.coordinates.dat.data_ro_with_halos
    mcoords = mash.coordinates.dat.data_with_halos
    for e in range(*cell_range):
        cell = cell_numbering.getOffset(e)
        cell_nodes = Vc.cell_node_list[cell, :]
        Xvals = mcoords_ro[cell_nodes, 0]
        if(Xvals.max() - Xvals.min() > Lx/2):
            mcoords[cell_nodes[2:], 0] = Lx
        else:
            mcoords

    local_facet_dat = mash.topology.interior_facets.local_facet_dat
    local_facet_number = mash.topology.interior_facets.local_facet_number

    lfd_ro = local_facet_dat.data_ro
    for i in range(lfd_ro.shape[0]):
        if all(lfd_ro[i, :] == np.array([3, 3])):
            local_facet_dat.data[i, :] = [2, 3]
            local_facet_number[i, :] = [2, 3]

    return mash


def UnitTriangleMesh(comm=COMM_WORLD):
    """Generate a mesh of the reference triangle

    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    coords = [[0., 0.], [1., 0.], [0., 1.]]
    cells = [[0, 1, 2]]
    plex = mesh._from_cell_list(2, cells, coords, comm)
    return mesh.Mesh(plex, reorder=False)


def RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False, reorder=None,
                  diagonal="left", comm=COMM_WORLD):
    """Generate a rectangular mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    :kwarg diagonal: For triangular meshes, should the diagonal got
        from bottom left to top right (``"right"``), or top left to
        bottom right (``"left"``).

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == Lx
    * 3: plane y == 0
    * 4: plane y == Ly
    """

    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(0.0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0.0, Ly, ny + 1, dtype=np.double)
    coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)

    # cell vertices
    i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))
    cells = [i*(ny+1) + j, i*(ny+1) + j+1, (i+1)*(ny+1) + j+1, (i+1)*(ny+1) + j]
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

    plex = mesh._from_cell_list(2, cells, coords, comm)

    # mark boundary facets
    plex.createLabel("boundary_ids")
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[0]) < xtol and abs(face_coords[2]) < xtol:
                plex.setLabelValue("boundary_ids", face, 1)
            if abs(face_coords[0] - Lx) < xtol and abs(face_coords[2] - Lx) < xtol:
                plex.setLabelValue("boundary_ids", face, 2)
            if abs(face_coords[1]) < ytol and abs(face_coords[3]) < ytol:
                plex.setLabelValue("boundary_ids", face, 3)
            if abs(face_coords[1] - Ly) < ytol and abs(face_coords[3] - Ly) < ytol:
                plex.setLabelValue("boundary_ids", face, 4)

    return mesh.Mesh(plex, reorder=reorder)


def SquareMesh(nx, ny, L, reorder=None, quadrilateral=False, comm=COMM_WORLD):
    """Generate a square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg L: The extent in the x and y directions
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == L
    * 3: plane y == 0
    * 4: plane y == L
    """
    return RectangleMesh(nx, ny, L, L, reorder=reorder,
                         quadrilateral=quadrilateral,
                         comm=comm)


def UnitSquareMesh(nx, ny, reorder=None, quadrilateral=False, comm=COMM_WORLD):
    """Generate a unit square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    """
    return SquareMesh(nx, ny, 1, reorder=reorder,
                      quadrilateral=quadrilateral,
                      comm=comm)


def PeriodicRectangleMesh(nx, ny, Lx, Ly, direction="both",
                          quadrilateral=False, reorder=None, comm=COMM_WORLD):
    """Generate a periodic rectangular mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg direction: The direction of the periodicity, one of
        ``"both"``, ``"x"`` or ``"y"``.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    If direction == "x" the boundary edges in this mesh are numbered as follows:

    * 1: plane y == 0
    * 2: plane y == Ly

    If direction == "y" the boundary edges are:

    * 1: plane x == 0
    * 2: plane x == Lx
    """

    if direction == "both" and ny == 1 and quadrilateral:
        return OneElementThickMesh(nx, Lx, Ly)

    if direction not in ("both", "x", "y"):
        raise ValueError("Cannot have a periodic mesh with periodicity '%s'" % direction)
    if direction != "both":
        return PartiallyPeriodicRectangleMesh(nx, ny, Lx, Ly, direction=direction,
                                              quadrilateral=quadrilateral,
                                              reorder=reorder,
                                              comm=comm)
    if nx < 3 or ny < 3:
        raise ValueError("2D periodic meshes with fewer than 3 \
cells in each direction are not currently supported")

    m = TorusMesh(nx, ny, 1.0, 0.5, quadrilateral=quadrilateral, reorder=reorder,
                  comm=comm)
    coord_fs = VectorFunctionSpace(m, 'DG', 1, dim=2)
    old_coordinates = m.coordinates
    new_coordinates = Function(coord_fs)

    periodic_kernel = """
double pi = 3.141592653589793;
double eps = 1e-12;
double bigeps = 1e-1;
double phi, theta, Y, Z;
Y = 0.0;
Z = 0.0;

for(int i=0; i<old_coords.dofs; i++) {
    Y += old_coords[i][1];
    Z += old_coords[i][2];
}

for(int i=0; i<new_coords.dofs; i++) {
    phi = atan2(old_coords[i][1], old_coords[i][0]);
    if (fabs(sin(phi)) > bigeps)
        theta = atan2(old_coords[i][2], old_coords[i][1]/sin(phi) - 1.0);
    else
        theta = atan2(old_coords[i][2], old_coords[i][0]/cos(phi) - 1.0);

    new_coords[i][0] = phi/(2.0*pi);
    if(new_coords[i][0] < -eps) {
        new_coords[i][0] += 1.0;
    }
    if(fabs(new_coords[i][0]) < eps && Y < 0.0) {
        new_coords[i][0] = 1.0;
    }

    new_coords[i][1] = theta/(2.0*pi);
    if(new_coords[i][1] < -eps) {
        new_coords[i][1] += 1.0;
    }
    if(fabs(new_coords[i][1]) < eps && Z < 0.0) {
        new_coords[i][1] = 1.0;
    }

    new_coords[i][0] *= Lx[0];
    new_coords[i][1] *= Ly[0];
}
"""

    cLx = Constant(Lx)
    cLy = Constant(Ly)

    par_loop(periodic_kernel, dx,
             {"new_coords": (new_coordinates, WRITE),
              "old_coords": (old_coordinates, READ),
              "Lx": (cLx, READ),
              "Ly": (cLy, READ)})

    return mesh.Mesh(new_coordinates)


def PeriodicSquareMesh(nx, ny, L, direction="both", quadrilateral=False, reorder=None,
                       comm=COMM_WORLD):
    """Generate a periodic square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg L: The extent in the x and y directions
    :arg direction: The direction of the periodicity, one of
        ``"both"``, ``"x"`` or ``"y"``.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    If direction == "x" the boundary edges in this mesh are numbered as follows:

    * 1: plane y == 0
    * 2: plane y == L

    If direction == "y" the boundary edges are:

    * 1: plane x == 0
    * 2: plane x == L
    """
    return PeriodicRectangleMesh(nx, ny, L, L, direction=direction,
                                 quadrilateral=quadrilateral, reorder=reorder,
                                 comm=comm)


def PeriodicUnitSquareMesh(nx, ny, direction="both", reorder=None,
                           quadrilateral=False, comm=COMM_WORLD):
    """Generate a periodic unit square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg direction: The direction of the periodicity, one of
        ``"both"``, ``"x"`` or ``"y"``.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    If direction == "x" the boundary edges in this mesh are numbered as follows:

    * 1: plane y == 0
    * 2: plane y == 1

    If direction == "y" the boundary edges are:

    * 1: plane x == 0
    * 2: plane x == 1
    """
    return PeriodicSquareMesh(nx, ny, 1.0, direction=direction,
                              reorder=reorder, quadrilateral=quadrilateral,
                              comm=comm)


def CircleManifoldMesh(ncells, radius=1, comm=COMM_WORLD):
    """Generated a 1D mesh of the circle, immersed in 2D.

    :arg ncells: number of cells the circle should be
         divided into (min 3)
    :kwarg radius: (optional) radius of the circle to approximate
           (defaults to 1).
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    if ncells < 3:
        raise ValueError("CircleManifoldMesh must have at least three cells")

    vertices = radius*np.column_stack((np.cos(np.arange(ncells, dtype=np.double)*(2*np.pi/ncells)),
                                       np.sin(np.arange(ncells, dtype=np.double)*(2*np.pi/ncells))))

    cells = np.column_stack((np.arange(0, ncells, dtype=np.int32),
                             np.roll(np.arange(0, ncells, dtype=np.int32), -1)))

    plex = mesh._from_cell_list(1, cells, vertices, comm)
    m = mesh.Mesh(plex, dim=2, reorder=False)
    m._radius = radius
    return m


def UnitTetrahedronMesh(comm=COMM_WORLD):
    """Generate a mesh of the reference tetrahedron.

    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    coords = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    cells = [[0, 1, 2, 3]]
    plex = mesh._from_cell_list(3, cells, coords, comm)
    return mesh.Mesh(plex, reorder=False)


def BoxMesh(nx, ny, nz, Lx, Ly, Lz, reorder=None, comm=COMM_WORLD):
    """Generate a mesh of a 3D box.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

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

    xcoords = np.linspace(0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0, Ly, ny + 1, dtype=np.double)
    zcoords = np.linspace(0, Lz, nz + 1, dtype=np.double)
    # X moves fastest, then Y, then Z
    coords = np.asarray(np.meshgrid(xcoords, ycoords, zcoords)).swapaxes(0, 3).reshape(-1, 3)
    i, j, k = np.meshgrid(np.arange(nx, dtype=np.int32),
                          np.arange(ny, dtype=np.int32),
                          np.arange(nz, dtype=np.int32))
    v0 = k*(nx + 1)*(ny + 1) + j*(nx + 1) + i
    v1 = v0 + 1
    v2 = v0 + (nx + 1)
    v3 = v1 + (nx + 1)
    v4 = v0 + (nx + 1)*(ny + 1)
    v5 = v1 + (nx + 1)*(ny + 1)
    v6 = v2 + (nx + 1)*(ny + 1)
    v7 = v3 + (nx + 1)*(ny + 1)

    cells = [v0, v1, v3, v7,
             v0, v1, v7, v5,
             v0, v5, v7, v4,
             v0, v3, v2, v7,
             v0, v6, v4, v7,
             v0, v2, v6, v7]
    cells = np.asarray(cells).swapaxes(0, 3).reshape(-1, 4)

    plex = mesh._from_cell_list(3, cells, coords, comm)

    # Apply boundary IDs
    plex.createLabel("boundary_ids")
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)
        ztol = Lz/(2*nz)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[0]) < xtol and abs(face_coords[3]) < xtol and abs(face_coords[6]) < xtol:
                plex.setLabelValue("boundary_ids", face, 1)
            if abs(face_coords[0] - Lx) < xtol and abs(face_coords[3] - Lx) < xtol and abs(face_coords[6] - Lx) < xtol:
                plex.setLabelValue("boundary_ids", face, 2)
            if abs(face_coords[1]) < ytol and abs(face_coords[4]) < ytol and abs(face_coords[7]) < ytol:
                plex.setLabelValue("boundary_ids", face, 3)
            if abs(face_coords[1] - Ly) < ytol and abs(face_coords[4] - Ly) < ytol and abs(face_coords[7] - Ly) < ytol:
                plex.setLabelValue("boundary_ids", face, 4)
            if abs(face_coords[2]) < ztol and abs(face_coords[5]) < ztol and abs(face_coords[8]) < ztol:
                plex.setLabelValue("boundary_ids", face, 5)
            if abs(face_coords[2] - Lz) < ztol and abs(face_coords[5] - Lz) < ztol and abs(face_coords[8] - Lz) < ztol:
                plex.setLabelValue("boundary_ids", face, 6)

    return mesh.Mesh(plex, reorder=reorder)


def CubeMesh(nx, ny, nz, L, reorder=None, comm=COMM_WORLD):
    """Generate a mesh of a cube

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg L: The extent in the x, y and z directions
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == L
    * 3: plane y == 0
    * 4: plane y == L
    * 5: plane z == 0
    * 6: plane z == L
    """
    return BoxMesh(nx, ny, nz, L, L, L, reorder=reorder, comm=comm)


def UnitCubeMesh(nx, ny, nz, reorder=None, comm=COMM_WORLD):
    """Generate a mesh of a unit cube

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    * 5: plane z == 0
    * 6: plane z == 1
    """
    return CubeMesh(nx, ny, nz, 1, reorder=reorder, comm=comm)


def IcosahedralSphereMesh(radius, refinement_level=0, degree=1, reorder=None,
                          comm=COMM_WORLD):
    """Generate an icosahedral approximation to the surface of the
    sphere.

    :arg radius: The radius of the sphere to approximate.
         For a radius R the edge length of the underlying
         icosahedron will be.

         .. math::

             a = \\frac{R}{\\sin(2 \\pi / 5)}

    :kwarg refinement_level: optional number of refinements (0 is an
        icosahedron).
    :kwarg degree: polynomial degree of coordinate space (defaults
        to 1: flat triangles)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    if refinement_level < 0 or refinement_level % 1:
            raise RuntimeError("Number of refinements must be a non-negative integer")

    if degree < 1:
        raise ValueError("Mesh coordinate degree must be at least 1")
    from math import sqrt
    phi = (1 + sqrt(5)) / 2
    # vertices of an icosahedron with an edge length of 2
    vertices = np.array([[-1, phi, 0],
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
                         [-phi, 0, 1]],
                        dtype=np.double)
    # faces of the base icosahedron
    faces = np.array([[0, 11, 5],
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
                      [9, 8, 1]], dtype=np.int32)

    plex = mesh._from_cell_list(2, faces, vertices, comm)
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()

    coords = plex.getCoordinatesLocal().array.reshape(-1, 3)
    scale = (radius / np.linalg.norm(coords, axis=1)).reshape(-1, 1)
    coords *= scale
    m = mesh.Mesh(plex, dim=3, reorder=reorder)
    if degree > 1:
        new_coords = function.Function(functionspace.VectorFunctionSpace(m, "CG", degree))
        new_coords.interpolate(ufl.SpatialCoordinate(m))
        # "push out" to sphere
        new_coords.dat.data[:] *= (radius / np.linalg.norm(new_coords.dat.data, axis=1)).reshape(-1, 1)
        m = mesh.Mesh(new_coords)
    m._radius = radius
    return m


def UnitIcosahedralSphereMesh(refinement_level=0, degree=1, reorder=None,
                              comm=COMM_WORLD):
    """Generate an icosahedral approximation to the unit sphere.

    :kwarg refinement_level: optional number of refinements (0 is an
        icosahedron).
    :kwarg degree: polynomial degree of coordinate space (defaults
        to 1: flat triangles)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    return IcosahedralSphereMesh(1.0, refinement_level=refinement_level,
                                 degree=degree, reorder=reorder,
                                 comm=comm)


def _cubedsphere_cells_and_coords(radius, refinement_level):
    """Generate vertex and face lists for cubed sphere """
    # We build the mesh out of 6 panels of the cube
    # this allows to build the gnonomic cube transformation
    # which is defined separately for each panel

    # Start by making a grid of local coordinates which we use
    # to map to each panel of the cubed sphere under the gnonomic
    # transformation
    dtheta = 2**(-refinement_level+1)*np.arctan(1.0)
    a = 3.0**(-0.5)*radius
    theta = np.arange(np.arctan(-1.0), np.arctan(1.0)+dtheta, dtheta, dtype=np.double)
    x = a*np.tan(theta)
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
    count = panel_numbering.max()+1

    # Numbering for panel 5
    panel_numbering[5, :, :] = count + np.arange(Nx**2, dtype=np.int32).reshape(Nx, Nx)
    count = panel_numbering.max()+1

    # Numbering for panel 4 - shares top edge with 0 and bottom edge
    #                         with 5
    # interior numbering
    panel_numbering[4, 1:-1, :] = count + np.arange(Nx*(Nx-2),
                                                    dtype=np.int32).reshape(Nx-2, Nx)

    # bottom edge
    panel_numbering[4, 0, :] = panel_numbering[5, -1, :]
    # top edge
    panel_numbering[4, -1, :] = panel_numbering[0, 0, :]
    count = panel_numbering.max()+1

    # Numbering for panel 3 - shares top edge with 5 and bottom edge
    #                         with 0
    # interior numbering
    panel_numbering[3, 1:-1, :] = count + np.arange(Nx*(Nx-2),
                                                    dtype=np.int32).reshape(Nx-2, Nx)
    # bottom edge
    panel_numbering[3, 0, :] = panel_numbering[0, -1, :]
    # top edge
    panel_numbering[3, -1, :] = panel_numbering[5, 0, :]
    count = panel_numbering.max()+1

    # Numbering for panel 1
    # interior numbering
    panel_numbering[1, 1:-1, 1:-1] = count + np.arange((Nx-2)**2,
                                                       dtype=np.int32).reshape(Nx-2, Nx-2)
    # left edge of 1 is left edge of 5 (inverted)
    panel_numbering[1, :, 0] = panel_numbering[5, ::-1, 0]
    # right edge of 1 is left edge of 0
    panel_numbering[1, :, -1] = panel_numbering[0, :, 0]
    # top edge (excluding vertices) of 1 is left edge of 3 (downwards)
    panel_numbering[1, -1, 1:-1] = panel_numbering[3, -2:0:-1, 0]
    # bottom edge (excluding vertices) of 1 is left edge of 4
    panel_numbering[1, 0, 1:-1] = panel_numbering[4, 1:-1, 0]
    count = panel_numbering.max()+1

    # Numbering for panel 2
    # interior numbering
    panel_numbering[2, 1:-1, 1:-1] = count + np.arange((Nx-2)**2,
                                                       dtype=np.int32).reshape(Nx-2, Nx-2)
    # left edge of 2 is right edge of 0
    panel_numbering[2, :, 0] = panel_numbering[0, :, -1]
    # right edge of 2 is right edge of 5 (inverted)
    panel_numbering[2, :, -1] = panel_numbering[5, ::-1, -1]
    # bottom edge (excluding vertices) of 2 is right edge of 4 (downwards)
    panel_numbering[2, 0, 1:-1] = panel_numbering[4, -2:0:-1, -1]
    # top edge (excluding vertices) of 2 is right edge of 3
    panel_numbering[2, -1, 1:-1] = panel_numbering[3, 1:-1, -1]
    count = panel_numbering.max()+1

    # That's the numbering done.

    # Set up an array for all of the mesh coordinates
    Npoints = panel_numbering.max()+1
    coords = np.zeros((Npoints, 3), dtype=np.double)
    lX, lY = np.meshgrid(x, x)
    lX.shape = (Nx**2,)
    lY.shape = (Nx**2,)
    r = (a**2 + lX**2 + lY**2)**0.5

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
    local_faces = np.zeros(((Nx-1)**2, 4), dtype=np.int32)
    local_faces[:, 0] = vertex_numbers[:-1, :-1].reshape(-1)
    local_faces[:, 1] = vertex_numbers[1:, :-1].reshape(-1)
    local_faces[:, 2] = vertex_numbers[1:, 1:].reshape(-1)
    local_faces[:, 3] = vertex_numbers[:-1, 1:].reshape(-1)

    cells = panel_numbering[:, local_faces].reshape(-1, 4)
    return cells, coords


def CubedSphereMesh(radius, refinement_level=0, degree=1,
                    reorder=None, comm=COMM_WORLD):
    """Generate an cubed approximation to the surface of the
    sphere.

    :arg radius: The radius of the sphere to approximate.
    :kwarg refinement_level: optional number of refinements (0 is a cube).
    :kwarg degree: polynomial degree of coordinate space (defaults
        to 1: bilinear quads)
    :kwarg reorder: (optional), should the mesh be reordered?
    """
    if refinement_level < 0 or refinement_level % 1:
            raise RuntimeError("Number of refinements must be a non-negative integer")

    if degree < 1:
        raise ValueError("Mesh coordinate degree must be at least 1")

    cells, coords = _cubedsphere_cells_and_coords(radius, refinement_level)
    plex = mesh._from_cell_list(2, cells, coords, comm)

    m = mesh.Mesh(plex, dim=3, reorder=reorder)

    if degree > 1:
        new_coords = function.Function(functionspace.VectorFunctionSpace(m, "Q", degree))
        new_coords.interpolate(ufl.SpatialCoordinate(m))
        # "push out" to sphere
        new_coords.dat.data[:] *= (radius / np.linalg.norm(new_coords.dat.data, axis=1)).reshape(-1, 1)
        m = mesh.Mesh(new_coords)
    m._radius = radius
    return m


def UnitCubedSphereMesh(refinement_level=0, degree=1, reorder=None, comm=COMM_WORLD):
    """Generate a cubed approximation to the unit sphere.

    :kwarg refinement_level: optional number of refinements (0 is a cube).
    :kwarg degree: polynomial degree of coordinate space (defaults
        to 1: bilinear quads)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    return CubedSphereMesh(1.0, refinement_level=refinement_level,
                           degree=degree, reorder=reorder, comm=comm)


def TorusMesh(nR, nr, R, r, quadrilateral=False, reorder=None, comm=COMM_WORLD):
    """Generate a toroidal mesh

    :arg nR: The number of cells in the major direction (min 3)
    :arg nr: The number of cells in the minor direction (min 3)
    :arg R: The major radius
    :arg r: The minor radius
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """

    if nR < 3 or nr < 3:
        raise ValueError("Must have at least 3 cells in each direction")

    for n in (nR, nr):
        if n % 1:
            raise RuntimeError("Number of cells must be an integer")

    # gives an array [[0, 0], [0, 1], ..., [1, 0], [1, 1], ...]
    idx_temp = np.asarray(np.meshgrid(np.arange(nR), np.arange(nr))).swapaxes(0, 2).reshape(-1, 2)

    # vertices - standard formula for (x, y, z), see Wikipedia
    vertices = np.asarray(np.column_stack((
        (R + r*np.cos(idx_temp[:, 1]*(2*np.pi/nr)))*np.cos(idx_temp[:, 0]*(2*np.pi/nR)),
        (R + r*np.cos(idx_temp[:, 1]*(2*np.pi/nr)))*np.sin(idx_temp[:, 0]*(2*np.pi/nR)),
        r*np.sin(idx_temp[:, 1]*(2*np.pi/nr)))), dtype=np.double)

    # cell vertices
    i, j = np.meshgrid(np.arange(nR, dtype=np.int32), np.arange(nr, dtype=np.int32))
    i = i.reshape(-1)  # Miklos's suggestion to make the code
    j = j.reshape(-1)  # less impenetrable
    cells = [i*nr + j, i*nr + (j+1) % nr, ((i+1) % nR)*nr + (j+1) % nr, ((i+1) % nR)*nr + j]
    cells = np.column_stack(cells)
    if not quadrilateral:
        # two cells per cell above...
        cells = cells[:, [0, 1, 3, 1, 2, 3]].reshape(-1, 3)

    plex = mesh._from_cell_list(2, cells, vertices, comm)
    m = mesh.Mesh(plex, dim=3, reorder=reorder)
    return m


def CylinderMesh(nr, nl, radius=1, depth=1, longitudinal_direction="z",
                 quadrilateral=False, reorder=None, comm=COMM_WORLD):
    """Generates a cylinder mesh.

    :arg nr: number of cells the cylinder circumference should be
         divided into (min 3)
    :arg nl: number of cells along the longitudinal axis of the cylinder
    :kwarg radius: (optional) radius of the cylinder to approximate
         (default 1).
    :kwarg depth: (optional) depth of the cylinder to approximate
         (default 1).
    :kwarg longitudinal_direction: (option) direction for the
         longitudinal axis of the cylinder.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    The boundary edges in this mesh are numbered as follows:

    * 1: plane l == 0 (bottom)
    * 2: plane l == depth (top)
    """
    if nr < 3:
        raise ValueError("CylinderMesh must have at least three cells")

    coord_xy = radius*np.column_stack((np.cos(np.arange(nr)*(2*np.pi/nr)),
                                       np.sin(np.arange(nr)*(2*np.pi/nr))))
    coord_z = depth*np.linspace(0.0, 1.0, nl + 1).reshape(-1, 1)
    vertices = np.asarray(np.column_stack((np.tile(coord_xy, (nl + 1, 1)),
                                           np.tile(coord_z, (1, nr)).reshape(-1, 1))),
                          dtype=np.double)

    # intervals on circumference
    ring_cells = np.column_stack((np.arange(0, nr, dtype=np.int32),
                                  np.roll(np.arange(0, nr, dtype=np.int32), -1)))
    # quads in the first layer
    ring_cells = np.column_stack((ring_cells, np.roll(ring_cells, 1, axis=1) + nr))
    offset = np.arange(nl, dtype=np.int32)*nr
    cells = np.row_stack((ring_cells + i for i in offset))
    if not quadrilateral:
        # two cells per cell above...
        cells = cells[:, [0, 1, 3, 1, 2, 3]].reshape(-1, 3)

    if longitudinal_direction == "x":
        rotation = np.asarray([[0, 0, 1],
                               [0, 1, 0],
                               [-1, 0, 0]], dtype=np.double)
        vertices = np.dot(vertices, rotation.T)
    elif longitudinal_direction == "y":
        rotation = np.asarray([[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0]], dtype=np.double)
        vertices = np.dot(vertices, rotation.T)
    elif longitudinal_direction != "z":
        raise ValueError("Unknown longitudinal direction '%s'" % longitudinal_direction)
    plex = mesh._from_cell_list(2, cells, vertices, comm)

    plex.createLabel("boundary_ids")
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        eps = depth/(2*nl)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            # index of x/y/z coordinates of the face element
            axis_ix = {"x": 0, "y": 1, "z": 2}
            i = axis_ix[longitudinal_direction]
            j = i + 3
            if abs(face_coords[i]) < eps and abs(face_coords[j]) < eps:
                # bottom of cylinder
                plex.setLabelValue("boundary_ids", face, 1)
            if abs(face_coords[i] - depth) < eps and abs(face_coords[j] - depth) < eps:
                # top of cylinder
                plex.setLabelValue("boundary_ids", face, 2)

    m = mesh.Mesh(plex, dim=3, reorder=reorder)
    return m


def PartiallyPeriodicRectangleMesh(nx, ny, Lx, Ly, direction="x", quadrilateral=False, reorder=None, comm=COMM_WORLD):
    """Generates RectangleMesh that is periodic in the x or y direction.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg direction: The direction of the periodicity (default x).
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

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
        raise ValueError("2D periodic meshes with fewer than 3 \
cells in each direction are not currently supported")

    m = CylinderMesh(na, nb, 1.0, 1.0, longitudinal_direction="z",
                     quadrilateral=quadrilateral, reorder=reorder, comm=comm)
    coord_fs = VectorFunctionSpace(m, 'DG', 1, dim=2)
    old_coordinates = m.coordinates
    new_coordinates = Function(coord_fs)

    # make x-periodic mesh
    # unravel x coordinates like in periodic interval
    # set y coordinates to z coordinates
    periodic_kernel = """double Y,pi;
            Y = 0.0;
            for(int i=0; i<old_coords.dofs; i++) {
                Y += old_coords[i][1];
            }

            pi=3.141592653589793;
            for(int i=0;i<new_coords.dofs;i++){
            new_coords[i][0] = atan2(old_coords[i][1],old_coords[i][0])/pi/2;
            if(new_coords[i][0]<0.) new_coords[i][0] += 1;
            if(new_coords[i][0]==0 && Y<0.) new_coords[i][0] = 1.0;
            new_coords[i][0] *= Lx[0];
            new_coords[i][1] = old_coords[i][2]*Ly[0];
            }"""

    cLx = Constant(La)
    cLy = Constant(Lb)

    par_loop(periodic_kernel, dx,
             {"new_coords": (new_coordinates, WRITE),
              "old_coords": (old_coordinates, READ),
              "Lx": (cLx, READ),
              "Ly": (cLy, READ)})

    if direction == "y":
        # flip x and y coordinates
        operator = np.asarray([[0, 1],
                               [1, 0]])
        new_coordinates.dat.data[:] = np.dot(new_coordinates.dat.data, operator.T)

    return mesh.Mesh(new_coordinates)
