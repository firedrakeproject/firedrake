import numpy as np
import os
import tempfile
from shutil import rmtree

from pyop2.mpi import MPI
from pyop2.profiling import profile

import mesh
from petsc import PETSc


__all__ = ['IntervalMesh', 'UnitIntervalMesh',
           'PeriodicIntervalMesh', 'PeriodicUnitIntervalMesh',
           'UnitTriangleMesh',
           'RectangleMesh', 'SquareMesh', 'UnitSquareMesh',
           'CircleMesh', 'UnitCircleMesh',
           'CircleManifoldMesh',
           'UnitTetrahedronMesh',
           'BoxMesh', 'CubeMesh', 'UnitCubeMesh',
           'IcosahedralSphereMesh', 'UnitIcosahedralSphereMesh']


_cachedir = os.path.join(tempfile.gettempdir(),
                         'firedrake-mesh-cache-uid%d' % os.getuid())


def _ensure_cachedir():
    if MPI.comm.rank == 0 and not os.path.exists(_cachedir):
        os.makedirs(_cachedir)

_ensure_cachedir()


def _clear_cachedir():
    if MPI.comm.rank == 0 and os.path.exists(_cachedir):
        rmtree(_cachedir, ignore_errors=True)
        _ensure_cachedir()


def _msh_exists(name):
    f = os.path.join(_cachedir, name)
    return os.path.exists(f + '.msh')


def _build_msh_file(input, output, dimension):
    try:
        # Must occur after mpi4py import due to:
        # 1) MPI initialisation issues
        # 2) LD_PRELOAD issues
        import gmshpy
        gmshpy.Msg.SetVerbosity(-1)
        # We've got the gmsh python interface available, so
        # use that, rather than spawning the gmsh binary.
        m = gmshpy.GModel()
        m.readGEO(input)
        m.mesh(dimension)
        m.writeMSH(output + ".msh")
        return
    except ImportError:
        raise RuntimeError('Creation of gmsh meshes requires gmshpy')


def _get_msh_file(source, name, dimension, meshed=False):
    """Given a source code, name and dimension  of the mesh,
    returns the name of the file that contains necessary information to build
    a mesh class. The mesh class would call _from_file method on this file
    to contruct itself.
    """

    if MPI.comm.rank == 0:
        input = os.path.join(_cachedir, name + '.geo')
        if not meshed:
            if not os.path.exists(input):
                with open(input, 'w') as f:
                    f.write(source)

        output = os.path.join(_cachedir, name)

        if not _msh_exists(name):
            if meshed:
                with file(output + '.msh', 'w') as f:
                    f.write(source)
            else:
                _build_msh_file(input, output, dimension)
        MPI.comm.bcast(output, root=0)
    else:
        output = MPI.comm.bcast(None, root=0)
    return output + '.msh'


def _from_cell_list(dim, cells, coords, comm=None):
    """
    Create a DMPlex from a list of cells and coords.

    :arg dim: The topological dimension of the mesh
    :arg cells: The vertices of each cell
    :arg coords: The coordinates of each vertex
    :arg comm: An optional MPI communicator to build the plex on
         (defaults to ``COMM_WORLD``)
    """

    if comm is None:
        comm = MPI.comm
    if comm.rank == 0:
        cells = np.asarray(cells, dtype=PETSc.IntType)
        coords = np.asarray(coords, dtype=float)
        comm.bcast(cells.shape, root=0)
        comm.bcast(coords.shape, root=0)
        # Provide the actual data on rank 0.
        return PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=comm)

    cell_shape = list(comm.bcast(None, root=0))
    coord_shape = list(comm.bcast(None, root=0))
    cell_shape[0] = 0
    coord_shape[0] = 0
    # Provide empty plex on other ranks
    # A subsequent call to plex.distribute() takes care of parallel partitioning
    return PETSc.DMPlex().createFromCellList(dim,
                                             np.zeros(cell_shape, dtype=PETSc.IntType),
                                             np.zeros(coord_shape, dtype=float),
                                             comm=comm)


@profile
def IntervalMesh(ncells, length):
    """
    Generate a uniform mesh of the interval [0,L].

    :arg ncells: The number of the cells over the interval.
    :arg length: The length of the interval.

    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=L`) point has marker 2.
    """
    dx = float(length) / ncells
    # This ensures the rightmost point is actually present.
    coords = np.arange(0, length + 0.01 * dx, dx).reshape(-1, 1)
    cells = np.dstack((np.arange(0, len(coords) - 1, dtype=np.int32),
                       np.arange(1, len(coords), dtype=np.int32))).reshape(-1, 2)
    plex = _from_cell_list(1, cells, coords)
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


def UnitIntervalMesh(ncells):
    """
    Generate a uniform mesh of the interval [0,1].

    :arg ncells: The number of the cells over the interval.

    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=1`) point has marker 2.
    """

    return IntervalMesh(ncells, length=1.0)


@profile
def PeriodicIntervalMesh(ncells, length):
    """Generate a periodic mesh of an interval.

    :arg ncells: The number of cells over the interval.
    :arg length: The length the interval."""

    if MPI.comm.size > 1:
        raise NotImplementedError("Periodic intervals not yet implemented in parallel")
    nvert = ncells
    nedge = ncells
    plex = PETSc.DMPlex().create()
    plex.setDimension(1)
    plex.setChart(0, nvert+nedge)
    for e in range(nedge):
        plex.setConeSize(e, 2)
    plex.setUp()
    for e in range(nedge-1):
        plex.setCone(e, [nedge+e, nedge+e+1])
        plex.setConeOrientation(e, [0, 0])
    # Connect v_(n-1) with v_0
    plex.setCone(nedge-1, [nedge+nvert-1, nedge])
    plex.setConeOrientation(nedge-1, [0, 0])
    plex.symmetrize()
    plex.stratify()

    # Build coordinate section
    dx = float(length) / ncells
    coords = [x for x in np.arange(0, length + 0.01 * dx, dx)]

    coordsec = plex.getCoordinateSection()
    coordsec.setChart(nedge, nedge+nvert)
    for v in range(nedge, nedge+nvert):
        coordsec.setDof(v, 1)
    coordsec.setUp()
    size = coordsec.getStorageSize()
    coordvec = PETSc.Vec().createWithArray(coords, size=size)
    plex.setCoordinatesLocal(coordvec)

    dx = length / ncells
    # HACK ALERT!
    # Almost certainly not right when symbolic geometry stuff lands.
    # Hopefully DMPlex will eventually give us a DG coordinate
    # field.  Until then, we build one by hand.
    coords = np.dstack((np.arange(dx, length + dx*0.01, dx),
                        np.arange(0, length - dx*0.01, dx))).flatten()
    # Last cell is back to front.
    coords[-2:] = coords[-2:][::-1]
    return mesh.Mesh(plex, periodic_coords=coords, reorder=False)


def PeriodicUnitIntervalMesh(ncells):
    """Generate a periodic mesh of the unit interval

    :arg ncells: The number of cells in the interval.
    """
    return PeriodicIntervalMesh(ncells, length=1.0)


def UnitTriangleMesh():
    """Generate a mesh of the reference triangle"""
    coords = [[0., 0.], [1., 0.], [0., 1.]]
    cells = [[0, 1, 2]]
    plex = _from_cell_list(2, cells, coords)
    return mesh.Mesh(plex, reorder=False)


@profile
def RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False, reorder=None):
    """Generate a rectangular mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == Lx
    * 3: plane y == 0
    * 4: plane y == Ly
    """
    if quadrilateral:
        dx = float(Lx) / nx
        dy = float(Ly) / ny
        xcoords = np.arange(0.0, Lx + 0.01 * dx, dx)
        ycoords = np.arange(0.0, Ly + 0.01 * dy, dy)
        coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)

        # cell vertices
        i, j = np.meshgrid(np.arange(nx), np.arange(ny))
        cells = [i*(ny+1) + j, i*(ny+1) + j+1, (i+1)*(ny+1) + j+1, (i+1)*(ny+1) + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)

        plex = _from_cell_list(2, cells, coords)
    else:
        boundary = PETSc.DMPlex().create(MPI.comm)
        boundary.setDimension(1)
        boundary.createSquareBoundary([0., 0.], [float(Lx), float(Ly)], [nx, ny])
        boundary.setTriangleOptions("pqezQYSl")

        plex = PETSc.DMPlex().generate(boundary)

    # mark boundary facets
    plex.createLabel("boundary_ids")
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = float(Lx)/(2*nx)
        ytol = float(Ly)/(2*ny)
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


def SquareMesh(nx, ny, L, reorder=None, quadrilateral=False):
    """Generate a square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg L: The extent in the x and y directions
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == L
    * 3: plane y == 0
    * 4: plane y == L
    """
    return RectangleMesh(nx, ny, L, L, reorder=reorder, quadrilateral=quadrilateral)


def UnitSquareMesh(nx, ny, reorder=None, quadrilateral=False):
    """Generate a unit square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered

    The boundary edges in this mesh are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    """
    return SquareMesh(nx, ny, 1, reorder=reorder, quadrilateral=quadrilateral)


@profile
def CircleMesh(radius, resolution, reorder=None):
    """Generate a structured triangular mesh of a circle.

    :arg radius: The radius of the circle.
    :arg resolution: The number of cells lying along the radius and
         the arc of the quadrant.
    :kwarg reorder: (optional), should the mesh be reordered?
    """
    source = """
    lc = %g;
    Point(1) = {0, -0.5, 0, lc};
    Point(2) = {0, 0.5, 0, lc};
    Line(1) = {1, 2};
    surface[] = Extrude{{0, 0, %g},{0, 0, 0}, 0.9999 * Pi}{
    Line{1};Layers{%d};
    };
    Physical Surface(2) = { surface[1] };
    """ % (0.5 / resolution, radius, resolution * 4)

    output = _get_msh_file(source, "circle_%g_%d" % (radius, resolution), 2)
    return mesh.Mesh(output, reorder=reorder)


def UnitCircleMesh(resolution, reorder=None):
    """Generate a structured triangular mesh of a unit circle.

    :arg resolution: The number of cells lying along the radius and
         the arc of the quadrant.
    :kwarg reorder: (optional), should the mesh be reordered?
    """
    return CircleMesh(1.0, resolution, reorder=reorder)


def CircleManifoldMesh(ncells, radius=1):
    """Generated a 1D mesh of the circle, immersed in 2D.

    :arg ncells: number of cells the circle should be
         divided into (min 3)
    :kwarg radius: (optional) radius of the circle to approximate
           (defaults to 1).
    """
    if ncells < 3:
        raise ValueError("CircleManifoldMesh must have at least three cells")

    vertices = radius*np.column_stack((np.cos(np.arange(ncells)*(2*np.pi/ncells)),
                                       np.sin(np.arange(ncells)*(2*np.pi/ncells))))

    cells = np.column_stack((np.arange(0, ncells, dtype=np.int32),
                             np.roll(np.arange(0, ncells, dtype=np.int32), -1)))

    plex = _from_cell_list(1, cells, vertices)
    return mesh.Mesh(plex, dim=2, reorder=False)


def UnitTetrahedronMesh():
    """Generate a mesh of the reference tetrahedron"""
    coords = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    cells = [[0, 1, 2, 3]]
    plex = _from_cell_list(3, cells, coords)
    return mesh.Mesh(plex, reorder=False)


@profile
def BoxMesh(nx, ny, nz, Lx, Ly, Lz, reorder=None):
    """Generate a mesh of a 3D box.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :kwarg reorder: (optional), should the mesh be reordered?

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == Lx
    * 3: plane y == 0
    * 4: plane y == Ly
    * 5: plane z == 0
    * 6: plane z == Lz
    """
    # Create mesh from DMPlex
    boundary = PETSc.DMPlex().create(MPI.comm)
    boundary.setDimension(2)
    boundary.createCubeBoundary([0., 0., 0.], [Lx, Ly, Lz], [nx, ny, nz])
    plex = PETSc.DMPlex().generate(boundary)

    # Apply boundary IDs
    plex.createLabel("boundary_ids")
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = float(Lx)/(2*nx)
        ytol = float(Ly)/(2*ny)
        ztol = float(Lz)/(2*nz)
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


def CubeMesh(nx, ny, nz, L, reorder=None):
    """Generate a mesh of a cube

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg L: The extent in the x, y and z directions
    :kwarg reorder: (optional), should the mesh be reordered?

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == L
    * 3: plane y == 0
    * 4: plane y == L
    * 5: plane z == 0
    * 6: plane z == L
    """
    return BoxMesh(nx, ny, nz, L, L, L, reorder=reorder)


def UnitCubeMesh(nx, ny, nz, reorder=None):
    """Generate a mesh of a unit cube

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :kwarg reorder: (optional), should the mesh be reordered?

    The boundary surfaces are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    * 5: plane z == 0
    * 6: plane z == 1
    """
    return CubeMesh(nx, ny, nz, 1, reorder=reorder)


@profile
def IcosahedralSphereMesh(radius, refinement_level=0, reorder=None):
    """Generate an icosahedral approximation to the surface of the
    sphere.

    :arg radius: The radius of the sphere to approximate.
         For a radius R the edge length of the underlying
         icosahedron will be.

         .. math::

             a = \\frac{R}{\\sin(2 \\pi / 5)}

    :kwarg refinement_level: optional number of refinements (0 is an
        icosahedron).
    :kwarg reorder: (optional), should the mesh be reordered?
    """
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
                         [-phi, 0, 1]])
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

    plex = _from_cell_list(2, faces, vertices)
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()

    vStart, vEnd = plex.getDepthStratum(0)
    nvertices = vEnd - vStart
    coords = plex.getCoordinatesLocal().array.reshape(nvertices, 3)
    scale = (radius / np.linalg.norm(coords, axis=1)).reshape(-1, 1)
    coords *= scale
    return mesh.Mesh(plex, dim=3, reorder=reorder)


def UnitIcosahedralSphereMesh(refinement_level=0, reorder=None):
    """Generate an icosahedral approximation to the unit sphere.

    :kwarg refinement_level: optional number of refinements (0 is an
        icosahedron).
    :kwarg reorder: (optional), should the mesh be reordered?
    """
    return IcosahedralSphereMesh(1.0, refinement_level=refinement_level,
                                 reorder=reorder)
