from __future__ import absolute_import
import numpy as np
import os
import tempfile
from shutil import rmtree

from pyop2.mpi import MPI
from pyop2.profiling import profile

from firedrake import VectorFunctionSpace, Function, Constant, \
    par_loop, dx, WRITE, READ
from firedrake import mesh
from firedrake import expression
from firedrake import function
from firedrake import functionspace
from firedrake.petsc import PETSc


__all__ = ['IntervalMesh', 'UnitIntervalMesh',
           'PeriodicIntervalMesh', 'PeriodicUnitIntervalMesh',
           'UnitTriangleMesh',
           'RectangleMesh', 'SquareMesh', 'UnitSquareMesh',
           'PeriodicRectangleMesh', 'PeriodicSquareMesh',
           'PeriodicUnitSquareMesh',
           'CircleMesh', 'UnitCircleMesh',
           'CircleManifoldMesh',
           'UnitTetrahedronMesh',
           'BoxMesh', 'CubeMesh', 'UnitCubeMesh',
           'IcosahedralSphereMesh', 'UnitIcosahedralSphereMesh',
           'CubedSphereMesh', 'UnitCubedSphereMesh',
           'TorusMesh']


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


@profile
def IntervalMesh(ncells, length_or_left, right=None):
    """
    Generate a uniform mesh of an interval.

    :arg ncells: The number of the cells over the interval.
    :arg length_or_left: The length of the interval (if :data:`right`
         is not provided) or else the left hand boundary point.
    :arg right: (optional) position of the right
         boundary point (in which case :data:`length_or_left` should
         be the left boundary point).

    The left hand boundary point has boundary marker 1,
    while the right hand point has marker 2.
    """
    if right is None:
        left = 0
        right = length_or_left
    else:
        left = length_or_left

    length = right - left
    if length < 0:
        raise RuntimeError("Requested mesh has negative length")
    dx = float(length) / ncells
    # This ensures the rightmost point is actually present.
    coords = np.arange(left, right + 0.01 * dx, dx).reshape(-1, 1)
    cells = np.dstack((np.arange(0, len(coords) - 1, dtype=np.int32),
                       np.arange(1, len(coords), dtype=np.int32))).reshape(-1, 2)
    plex = mesh._from_cell_list(1, cells, coords)
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

    return IntervalMesh(ncells, length_or_left=1.0)


@profile
def PeriodicIntervalMesh(ncells, length):
    """Generate a periodic mesh of an interval.

    :arg ncells: The number of cells over the interval.
    :arg length: The length the interval."""

    if ncells < 3:
        raise ValueError("1D periodic meshes with fewer than 3 \
cells are not currently supported")

    m = CircleManifoldMesh(ncells)
    coord_fs = VectorFunctionSpace(m, 'DG', 1, dim=1)
    old_coordinates = m.coordinates
    new_coordinates = Function(coord_fs)

    periodic_kernel = """double Y,pi;
            Y = 0.5*(old_coords[0][1]-old_coords[1][1]);
            pi=3.141592653589793;
            for(int i=0;i<2;i++){
            new_coords[i][0] = atan2(old_coords[i][1],old_coords[i][0])/pi/2;
            if(new_coords[i][0]<0.) new_coords[i][0] += 1;
            if(new_coords[i][0]==0 && Y<0.) new_coords[i][0] = 1.0;
            new_coords[i][0] *= L[0];
            }"""

    cL = Constant(length)

    par_loop(periodic_kernel, dx,
             {"new_coords": (new_coordinates, WRITE),
              "old_coords": (old_coordinates, READ),
              "L": (cL, READ)})

    return new_coordinates.as_coordinates()


def PeriodicUnitIntervalMesh(ncells):
    """Generate a periodic mesh of the unit interval

    :arg ncells: The number of cells in the interval.
    """
    return PeriodicIntervalMesh(ncells, length=1.0)


def UnitTriangleMesh():
    """Generate a mesh of the reference triangle"""
    coords = [[0., 0.], [1., 0.], [0., 1.]]
    cells = [[0, 1, 2]]
    plex = mesh._from_cell_list(2, cells, coords)
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

        plex = mesh._from_cell_list(2, cells, coords)
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
def PeriodicRectangleMesh(nx, ny, Lx, Ly, quadrilateral=False, reorder=None):
    """Generate a periodic rectangular mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    """

    if nx < 3 or ny < 3:
        raise ValueError("2D periodic meshes with fewer than 3 \
cells in each direction are not currently supported")

    m = TorusMesh(nx, ny, 1.0, 0.5, quadrilateral=quadrilateral, reorder=reorder)
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

    return new_coordinates.as_coordinates()


def PeriodicSquareMesh(nx, ny, L, quadrilateral=False, reorder=None):
    """Generate a periodic square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg L: The extent in the x and y directions
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    """
    return PeriodicRectangleMesh(nx, ny, L, L, quadrilateral=quadrilateral, reorder=reorder)


def PeriodicUnitSquareMesh(nx, ny, reorder=None, quadrilateral=False):
    """Generate a periodic unit square mesh

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    """
    return PeriodicSquareMesh(nx, ny, 1.0, reorder=reorder, quadrilateral=quadrilateral)


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


@profile
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

    plex = mesh._from_cell_list(1, cells, vertices)
    m = mesh.Mesh(plex, dim=2, reorder=False)
    m._circle_manifold = radius
    return m


def UnitTetrahedronMesh():
    """Generate a mesh of the reference tetrahedron"""
    coords = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    cells = [[0, 1, 2, 3]]
    plex = mesh._from_cell_list(3, cells, coords)
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
def IcosahedralSphereMesh(radius, refinement_level=0, degree=1, reorder=None):
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
    """
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

    plex = mesh._from_cell_list(2, faces, vertices)
    plex.setRefinementUniform(True)
    for i in range(refinement_level):
        plex = plex.refine()

    coords = plex.getCoordinatesLocal().array.reshape(-1, 3)
    scale = (radius / np.linalg.norm(coords, axis=1)).reshape(-1, 1)
    coords *= scale
    m = mesh.Mesh(plex, dim=3, reorder=reorder)
    if degree > 1:
        new_coords = function.Function(functionspace.VectorFunctionSpace(m, "CG", degree))
        new_coords.interpolate(expression.Expression(("x[0]", "x[1]", "x[2]")))
        # "push out" to sphere
        new_coords.dat.data[:] *= (radius / np.linalg.norm(new_coords.dat.data, axis=1)).reshape(-1, 1)
        m = new_coords.as_coordinates()
    m._icosahedral_sphere = radius
    return m


def UnitIcosahedralSphereMesh(refinement_level=0, degree=1, reorder=None):
    """Generate an icosahedral approximation to the unit sphere.

    :kwarg refinement_level: optional number of refinements (0 is an
        icosahedron).
    :kwarg degree: polynomial degree of coordinate space (defaults
        to 1: flat triangles)
    :kwarg reorder: (optional), should the mesh be reordered?
    """
    return IcosahedralSphereMesh(1.0, refinement_level=refinement_level,
                                 degree=degree, reorder=reorder)


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
    theta = np.arange(np.arctan(-1.0), np.arctan(1.0)+dtheta, dtheta)
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
    coords = np.zeros((Npoints, 3), dtype=float)
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
    vertex_numbers = np.arange(Nx**2).reshape(Nx, Nx)
    local_faces = np.zeros(((Nx-1)**2, 4), dtype=np.int32)
    local_faces[:, 0] = vertex_numbers[:-1, :-1].reshape(-1)
    local_faces[:, 1] = vertex_numbers[1:, :-1].reshape(-1)
    local_faces[:, 2] = vertex_numbers[1:, 1:].reshape(-1)
    local_faces[:, 3] = vertex_numbers[:-1, 1:].reshape(-1)

    cells = panel_numbering[:, local_faces].reshape(-1, 4)
    return cells, coords


@profile
def CubedSphereMesh(radius, refinement_level=0, degree=1,
                    reorder=None, use_dmplex_refinement=False):
    """Generate an cubed approximation to the surface of the
    sphere.

    :arg radius: The radius of the sphere to approximate.
    :kwarg refinement_level: optional number of refinements (0 is a cube).
    :kwarg degree: polynomial degree of coordinate space (defaults
        to 1: bilinear quads)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg use_dmplex_refinement: (optional), use dmplex to apply
        the refinement.
    """
    if degree < 1:
        raise ValueError("Mesh coordinate degree must be at least 1")

    if use_dmplex_refinement:
        # vertices of a cube with an edge length of 2
        vertices = np.array([[-1., -1., -1.],
                             [1., -1., -1.],
                             [-1., 1., -1.],
                             [1., 1., -1.],
                             [-1., -1., 1.],
                             [1., -1., 1.],
                             [-1., 1., 1.],
                             [1., 1., 1.]])
        # faces of the base cube
        # bottom face viewed from above
        # 2 3
        # 0 1
        # top face viewed from above
        # 6 7
        # 4 5
        faces = np.array([[0, 1, 3, 2],  # bottom
                          [4, 5, 7, 6],  # top
                          [0, 1, 5, 4],
                          [2, 3, 7, 6],
                          [0, 2, 6, 4],
                          [1, 3, 7, 5]], dtype=np.int32)

        plex = mesh._from_cell_list(2, faces, vertices)
        plex.setRefinementUniform(True)
        for i in range(refinement_level):
            plex = plex.refine()

        # rescale points to the sphere
        # this is not the same as the gnonomic transformation
        coords = plex.getCoordinatesLocal().array.reshape(-1, 3)
        scale = (radius / np.linalg.norm(coords, axis=1)).reshape(-1, 1)
        coords *= scale
    else:
        cells, coords = _cubedsphere_cells_and_coords(radius, refinement_level)
        plex = mesh._from_cell_list(2, cells, coords)

    m = mesh.Mesh(plex, dim=3, reorder=reorder)

    if degree > 1:
        new_coords = function.Function(functionspace.VectorFunctionSpace(m, "Q", degree))
        new_coords.interpolate(expression.Expression(("x[0]", "x[1]", "x[2]")))
        # "push out" to sphere
        new_coords.dat.data[:] *= (radius / np.linalg.norm(new_coords.dat.data, axis=1)).reshape(-1, 1)
        m = new_coords.as_coordinates()

    return m


def UnitCubedSphereMesh(refinement_level=0, degree=1, reorder=None):
    """Generate a cubed approximation to the unit sphere.

    :kwarg refinement_level: optional number of refinements (0 is a cube).
    :kwarg degree: polynomial degree of coordinate space (defaults
        to 1: bilinear quads)
    :kwarg reorder: (optional), should the mesh be reordered?
    """
    return CubedSphereMesh(1.0, refinement_level=refinement_level,
                           degree=degree, reorder=reorder)


@profile
def TorusMesh(nR, nr, R, r, quadrilateral=False, reorder=None):
    """Generate a toroidal mesh

    :arg nR: The number of cells in the major direction (min 3)
    :arg nr: The number of cells in the minor direction (min 3)
    :arg R: The major radius
    :arg r: The minor radius
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    """
    if nR < 3 or nr < 3:
        raise ValueError("Must have at least 3 cells in each direction")

    # gives an array [[0, 0], [0, 1], ..., [1, 0], [1, 1], ...]
    idx_temp = np.asarray(np.meshgrid(np.arange(nR), np.arange(nr))).swapaxes(0, 2).reshape(-1, 2)

    # vertices - standard formula for (x, y, z), see Wikipedia
    vertices = np.column_stack((
        (R + r*np.cos(idx_temp[:, 1]*(2*np.pi/nr)))*np.cos(idx_temp[:, 0]*(2*np.pi/nR)),
        (R + r*np.cos(idx_temp[:, 1]*(2*np.pi/nr)))*np.sin(idx_temp[:, 0]*(2*np.pi/nR)),
        r*np.sin(idx_temp[:, 1]*(2*np.pi/nr))))

    # cell vertices
    i, j = np.meshgrid(np.arange(nR), np.arange(nr))
    i = i.reshape(-1)  # Miklos's suggestion to make the code
    j = j.reshape(-1)  # less impenetrable
    cells = [i*nr + j, i*nr + (j+1) % nr, ((i+1) % nR)*nr + (j+1) % nr, ((i+1) % nR)*nr + j]
    cells = np.column_stack(cells)
    if not quadrilateral:
        # two cells per cell above...
        cells = cells[:, [0, 1, 3, 1, 2, 3]].reshape(-1, 3)

    plex = mesh._from_cell_list(2, cells, vertices)
    m = mesh.Mesh(plex, dim=3, reorder=reorder)
    return m
