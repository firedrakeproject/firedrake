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
           'UnitSquareMesh',
           'CircleManifoldMesh',
           'UnitTetrahedronMesh',
           'UnitCubeMesh',
           'IcosahedralSphereMesh',
           'UnitIcosahedralSphereMesh']


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


class IntervalMesh(mesh.Mesh):
    """
    Generate a uniform mesh of the interval [0,L] for user specified L.

    :arg ncells: The number of the cells over the interval.
    :arg length: The length of the interval.

    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=L`) point has marker 2.
    """

    @profile
    def __init__(self, ncells, length):
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

        super(IntervalMesh, self).__init__(plex, reorder=False)


class UnitIntervalMesh(IntervalMesh):
    """
    Generate a uniform mesh of the interval [0,1].

    :arg ncells: The number of the cells over the interval.
    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=1`) point has marker 2.
    """

    @profile
    def __init__(self, ncells):
        super(UnitIntervalMesh, self).__init__(ncells, length=1.0)


class PeriodicIntervalMesh(mesh.Mesh):
    """Generate a periodic uniform mesh of the interval [0, L], for
    user specified L.

    :arg ncells: The number of cells over the interval.
    :arg length: The length the interval."""

    @profile
    def __init__(self, ncells, length):
        """Build the periodic Plex by hand"""

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
        super(PeriodicIntervalMesh, self).__init__(plex,
                                                   periodic_coords=coords,
                                                   reorder=False)


class PeriodicUnitIntervalMesh(PeriodicIntervalMesh):
    """Generate a periodic uniform mesh of the interval [0, 1].
    :arg ncells: The number of cells over the interval."""

    @profile
    def __init__(self, ncells):
        super(PeriodicUnitIntervalMesh, self).__init__(ncells, length=1.0)


class UnitTriangleMesh(mesh.Mesh):

    """Class that represents a triangle mesh composed of one element."""

    def __init__(self):
        coords = [[0., 0.], [1., 0.], [0., 1.]]
        cells = [[0, 1, 2]]
        plex = _from_cell_list(2, cells, coords)
        super(UnitTriangleMesh, self).__init__(plex, reorder=False)


class UnitSquareMesh(mesh.Mesh):

    """Class that represents a structured triangular mesh of a 2D square whose
    edge is a unit length.

    :arg nx: The number of the cells in the x direction.
    :arg ny: The number of the cells in the y direction.
    :arg reorder: Should the mesh be reordered?

    The number of the elements in a mesh can be computed from 2 * nx * ny,
    and the number of vertices from (nx+1) * (ny+1).

    The boundary edges are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    """

    @profile
    def __init__(self, nx, ny, reorder=None):
        # Create mesh from DMPlex
        boundary = PETSc.DMPlex().create(MPI.comm)
        boundary.setDimension(1)
        boundary.createSquareBoundary([0., 0.], [1., 1.], [nx, ny])
        plex = PETSc.DMPlex().generate(boundary)

        # Apply boundary IDs
        plex.createLabel("boundary_ids")
        plex.markBoundaryFaces("boundary_faces")
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()
        if plex.getStratumSize("boundary_faces", 1) > 0:
            boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
            xtol = 1./(2*nx)
            ytol = 1./(2*ny)
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if abs(face_coords[0]) < xtol and abs(face_coords[2]) < xtol:
                    plex.setLabelValue("boundary_ids", face, 1)
                if abs(face_coords[0] - 1.) < xtol and abs(face_coords[2] - 1.) < xtol:
                    plex.setLabelValue("boundary_ids", face, 2)
                if abs(face_coords[1]) < ytol and abs(face_coords[3]) < ytol:
                    plex.setLabelValue("boundary_ids", face, 3)
                if abs(face_coords[1] - 1.) < ytol and abs(face_coords[3] - 1.) < ytol:
                    plex.setLabelValue("boundary_ids", face, 4)

        super(UnitSquareMesh, self).__init__(plex, reorder=reorder)


class UnitCircleMesh(mesh.Mesh):

    """Class that represents a structured triangle mesh of a 2D circle of an
    unit circle.

    :arg resolution: The number of cells lying along the radius and the arc of
      the quadrant.
    :arg reorder: Should the mesh be reordered?
    """

    @profile
    def __init__(self, resolution, reorder=None):
        source = """
            lc = %g;
            Point(1) = {0, -0.5, 0, lc};
            Point(2) = {0, 0.5, 0, lc};
            Line(1) = {1, 2};
            surface[] = Extrude{{0, 0, 1},{0, 0, 0}, 0.9999 * Pi}{
                    Line{1};Layers{%d};
            };
            Physical Surface(2) = { surface[1] };
            """ % (0.5 / resolution, resolution * 4)

        output = _get_msh_file(source, "unitcircle_%d" % resolution, 2)
        super(UnitCircleMesh, self).__init__(output, reorder=reorder)


class CircleManifoldMesh(mesh.Mesh):

    """A 1D mesh of the circle, immersed in 2D"""

    @profile
    def __init__(self, ncells, radius=1):
        """
        :arg ncells: number of cells the circle should be
             divided into (min 3)

        :arg radius: the radius of the circle to approximate
        """

        if ncells < 3:
            raise ValueError("CircleManifoldMesh must have at least three cells")

        self._R = radius
        self._ncells = ncells

        self._vertices = radius*np.column_stack((np.cos(np.arange(ncells)*(2*np.pi/ncells)),
                                                 np.sin(np.arange(ncells)*(2*np.pi/ncells))))

        self._cells = np.column_stack((np.arange(0, ncells, dtype=np.int32),
                                       np.roll(np.arange(0, ncells, dtype=np.int32), -1)))

        plex = _from_cell_list(1, self._cells, self._vertices)
        super(CircleManifoldMesh, self).__init__(plex, dim=2, reorder=False)


class UnitTetrahedronMesh(mesh.Mesh):

    """Class that represents a tetrahedron mesh that is composed of one
    element.
    """

    def __init__(self):
        coords = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        cells = [[0, 1, 2, 3]]
        plex = _from_cell_list(3, cells, coords)
        super(UnitTetrahedronMesh, self).__init__(plex, reorder=False)


class UnitCubeMesh(mesh.Mesh):

    """Class that represents a structured tetrahedron mesh of a 3D cube whose
    edge is a unit length.

    :arg nx: The number of the cells in the x direction.
    :arg ny: The number of the cells in the y direction.
    :arg nx: The number of the cells in the z direction.
    :arg reorder: Should the mesh be reordered?

    The number of the elements in a mesh can be computed from 6 * nx * ny * nz,
    and the number of the vertices from (nx+1) * (ny+1) * (nz+1).

    The boundary surface are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    * 5: plane z == 0
    * 6: plane z == 1
    """

    @profile
    def __init__(self, nx, ny, nz, reorder=None):

        # Create mesh from DMPlex
        boundary = PETSc.DMPlex().create(MPI.comm)
        boundary.setDimension(2)
        boundary.createCubeBoundary([0., 0., 0.], [1., 1., 1.], [nx, ny, nz])
        plex = PETSc.DMPlex().generate(boundary)

        # Apply boundary IDs
        plex.createLabel("boundary_ids")
        plex.markBoundaryFaces("boundary_faces")
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()
        if plex.getStratumSize("boundary_faces", 1) > 0:
            boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
            xtol = 1./(2*nx)
            ytol = 1./(2*ny)
            ztol = 1./(2*nz)
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if abs(face_coords[0]) < xtol and abs(face_coords[3]) < xtol and abs(face_coords[6]) < xtol:
                    plex.setLabelValue("boundary_ids", face, 1)
                if abs(face_coords[0] - 1.) < xtol and abs(face_coords[3] - 1.) < xtol and abs(face_coords[6] - 1.) < xtol:
                    plex.setLabelValue("boundary_ids", face, 2)
                if abs(face_coords[1]) < ytol and abs(face_coords[4]) < ytol and abs(face_coords[7]) < ytol:
                    plex.setLabelValue("boundary_ids", face, 3)
                if abs(face_coords[1] - 1.) < ytol and abs(face_coords[4] - 1.) < ytol and abs(face_coords[7] - 1.) < ytol:
                    plex.setLabelValue("boundary_ids", face, 4)
                if abs(face_coords[2]) < ztol and abs(face_coords[5]) < ztol and abs(face_coords[8]) < ztol:
                    plex.setLabelValue("boundary_ids", face, 5)
                if abs(face_coords[2] - 1.) < ztol and abs(face_coords[5] - 1.) < ztol and abs(face_coords[8] - 1.) < ztol:
                    plex.setLabelValue("boundary_ids", face, 6)

        super(UnitCubeMesh, self).__init__(plex, reorder=reorder)


class IcosahedralSphereMesh(mesh.Mesh):

    from math import sqrt

    """An icosahedral mesh of the surface of the sphere"""
    phi = (1 + sqrt(5)) / 2
    del sqrt
    # vertices of an icosahedron with an edge length of 2
    _base_vertices = np.array([[-1, phi, 0],
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
    del phi
    # faces of the base icosahedron
    _base_faces = np.array([[0, 11, 5],
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

    @profile
    def __init__(self, radius=1, refinement_level=0, reorder=None):
        """
        :arg radius: the radius of the sphere to approximate.
             For a radius R the edge length of the underlying
             icosahedron will be.

             .. math::

                a = \\frac{R}{\\sin(2 \\pi / 5)}

        :arg refinement_level: how many levels of refinement, zero
                               corresponds to an icosahedron.
        :arg reorder: Should the mesh be reordered?
        """

        self._R = radius
        self._refinement = refinement_level

        self._vertices = np.empty_like(IcosahedralSphereMesh._base_vertices)
        self._faces = np.copy(IcosahedralSphereMesh._base_faces)
        # Rescale so that vertices live on sphere of specified radius
        for i, vtx in enumerate(IcosahedralSphereMesh._base_vertices):
            self._vertices[i] = self._force_to_sphere(vtx)

        for i in range(refinement_level):
            self._refine()

        plex = _from_cell_list(2, self._faces, self._vertices)
        super(IcosahedralSphereMesh, self).__init__(plex, dim=3, reorder=reorder)

    def _force_to_sphere(self, vtx):
        """
        Scale `vtx` such that it sits on surface of the sphere this mesh
        represents.

        """
        scale = self._R / np.linalg.norm(vtx)
        return vtx * scale

    def _refine(self):
        """Refine mesh by one level.

        This increases the number of faces in the mesh by a factor of four."""
        cache = {}
        new_faces = np.empty((4 * len(self._faces), 3), dtype=np.int32)
        # Dividing each face adds 1.5 extra vertices (each vertex on
        # the midpoint is shared two ways).
        new_vertices = np.empty((len(self._vertices) + 3 * len(self._faces) / 2, 3))
        f_idx = 0
        v_idx = len(self._vertices)
        new_vertices[:v_idx] = self._vertices

        def midpoint(v1, v2):
            return self._force_to_sphere((self._vertices[v1] + self._vertices[v2])/2)

        # Walk old faces, splitting into 4
        for (v1, v2, v3) in self._faces:
            a = midpoint(v1, v2)
            b = midpoint(v2, v3)
            c = midpoint(v3, v1)
            ka = tuple(sorted((v1, v2)))
            kb = tuple(sorted((v2, v3)))
            kc = tuple(sorted((v3, v1)))
            if ka not in cache:
                cache[ka] = v_idx
                new_vertices[v_idx] = a
                v_idx += 1
            va = cache[ka]
            if kb not in cache:
                cache[kb] = v_idx
                new_vertices[v_idx] = b
                v_idx += 1
            vb = cache[kb]
            if kc not in cache:
                cache[kc] = v_idx
                new_vertices[v_idx] = c
                v_idx += 1
            vc = cache[kc]
            #
            #         v1
            #        /  \
            #       /    \
            #      v2----v3
            #
            #         v1
            #        /  \
            #       a--- c
            #      / \  / \
            #     /   \/   \
            #   v2----b----v3
            #
            new_faces[f_idx][:] = (v1, va, vc)
            new_faces[f_idx+1][:] = (v2, vb, va)
            new_faces[f_idx+2][:] = (v3, vc, vb)
            new_faces[f_idx+3][:] = (va, vb, vc)
            f_idx += 4
        self._vertices = new_vertices
        self._faces = new_faces


class UnitIcosahedralSphereMesh(IcosahedralSphereMesh):
    """An icosahedral approximation to the unit sphere."""

    @profile
    def __init__(self, refinement_level=0, reorder=None):
        """
        :arg refinement_level: how many levels to refine the mesh.
        :arg reorder: Should the mesh be reordered?
        """
        super(UnitIcosahedralSphereMesh, self).__init__(1, refinement_level, reorder=reorder)
