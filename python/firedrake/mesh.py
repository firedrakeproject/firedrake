import tempfile
from core_types import Mesh
from interval import periodic_interval_mesh
import subprocess
from pyop2.mpi import MPI
import os
from shutil import rmtree
import numpy as np

import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

import firedrake

try:
    # Must occur after mpi4py import due to:
    # 1) MPI initialisation issues
    # 2) LD_PRELOAD issues
    import gmshpy
    gmshpy.Msg.SetVerbosity(-1)
except ImportError:
    gmshpy = None

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

_exts = [".node", ".ele"]
_2dexts = [".edge"]
_3dexts = [".face"]
_pexts = [".halo"]


def _msh_exists(name):
    f = os.path.join(_cachedir, name)
    return os.path.exists(f + '.msh')


def _build_msh_file(input, output, dimension):
    if gmshpy:
        # We've got the gmsh python interface available, so
        # use that, rather than spawning the gmsh binary.
        m = gmshpy.GModel()
        m.readGEO(input)
        m.mesh(dimension)
        m.writeMSH(output + ".msh")
        return
    # Writing of the output file.
    from mpi4py import MPI as _MPI
    # We must use MPI's process spawning functionality because if Gmsh
    # has been compiled with MPI and linked against the library then
    # just running it as a subprocess doesn't work.
    _MPI.COMM_SELF.Spawn('gmsh', args=[input, "-" + str(dimension),
                                       '-o', output + '.msh'])
    # Hideous: MPI_Comm_spawn returns as soon as the child calls
    # MPI_Init.  So to wait for the gmsh process to complete we ought
    # to call MPI_Comm_disconnect.  However, that's collective over
    # the intercommunicator and gmsh doesn't call it, so we deadlock.
    # Instead, sit spinning on the output file until gmsh has finished
    # writing it before proceeding to the next step.
    oldsize = 0
    import time
    while True:
        try:
            statinfo = os.stat(output + '.msh')
            newsize = statinfo.st_size
            if newsize == 0 or newsize != oldsize:
                oldsize = newsize
                # Sleep so we don't restat too soon.
                time.sleep(1)
            else:
                # Gmsh has finished writing the output
                # file, we hope, so break the loop.
                break
        except OSError as e:
            if e.errno == 2:
                # file didn't exist
                pass
            else:
                raise e


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
        if MPI.parallel:
            if dimension == 2:
                exts = _exts + _2dexts
            else:
                exts = _exts + _3dexts
            if not _triangled(output, exts):
                gmsh2triangle = os.path.split(firedrake.__file__)[0] +\
                    "/../../bin/gmsh2triangle"
                if not os.path.exists(gmsh2triangle):
                    raise OSError(
                        "gmsh2triangle not found. Did you make fltools?")
                args = [gmsh2triangle, output + '.msh']
                if dimension == 2:
                    args.append('--2d')
                subprocess.call(args, cwd=_cachedir)

            basename = output + "_" + str(MPI.comm.size)
            # Deal with decomposition.
            # fldecomp would always name the decomposed triangle files
            # in a same way.(meshname_rank.node, rather than
            # meshname_size_rank.node).
            # To go around this without creating triangle files everytime,
            # we can make a simlink meshname_size.node which points to
            # the file meshname.node.
            for ext in exts:
                if os.path.exists(output + ext) \
                        and not os.path.lexists(basename + ext):
                    os.symlink(output + ext, basename + ext)
            pexts = exts + _pexts
            if not all([_triangled(basename + '_' + str(r), pexts)
                        for r in xrange(MPI.comm.size)]):
                fldecomp = os.path.split(firedrake.__file__)[0] +\
                    "/../../bin/fldecomp"
                if not os.path.exists(fldecomp):
                    raise OSError("fldecomp not found. Did you make fltools?")

                subprocess.call([fldecomp, '-n', str(MPI.comm.size), '-m',
                                 'triangle', basename])

            output = basename + ".node"
            MPI.comm.bcast(output, root=0)

    # Not processor-0
    else:
        output = MPI.comm.bcast(None, root=0)

    return output if MPI.parallel else output + '.msh'


def _triangled(basename, exts):
    """ Checks if the mesh of the given basename has already been decomposed.
    """
    return all(map(lambda ext: os.path.exists(basename + ext), exts))


class UnitSquareMesh(Mesh):

    """Class that represents a structured triangular mesh of a 2D square whose
    edge is a unit length.

    :arg nx: The number of the cells in the x direction.
    :arg ny: The number of the cells in the y direction.

    The number of the elements in a mesh can be computed from 2 * nx * ny,
    and the number of vertices from (nx+1) * (ny+1).

    The boundary edges are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    """

    def __init__(self, nx, ny):
        self.name = "unitsquare_%d_%d" % (nx, ny)

        # Create mesh from DMPlex
        boundary = PETSc.DMPlex().create(MPI.comm)
        boundary.setDimension(1)
        boundary.createSquareBoundary([0., 0.], [1., 1.], [nx, ny])
        dmplex = PETSc.DMPlex().generate(boundary)

        # Apply boundary IDs
        dmplex.createLabel("boundary_ids")
        dmplex.markBoundaryFaces("boundary_faces")
        coords = dmplex.getCoordinates()
        coord_sec = dmplex.getCoordinateSection()
        if dmplex.getStratumSize("boundary_faces", 1) > 0:
            boundary_faces = dmplex.getStratumIS("boundary_faces", 1).getIndices()
            for face in boundary_faces:
                face_coords = dmplex.vecGetClosure(coord_sec, coords, face)
                if face_coords[0] == 0. and face_coords[2] == 0.:
                    dmplex.setLabelValue("boundary_ids", face, 1)
                if face_coords[0] == 1. and face_coords[2] == 1.:
                    dmplex.setLabelValue("boundary_ids", face, 2)
                if face_coords[1] == 0. and face_coords[3] == 0.:
                    dmplex.setLabelValue("boundary_ids", face, 3)
                if face_coords[1] == 1. and face_coords[3] == 1.:
                    dmplex.setLabelValue("boundary_ids", face, 4)

        super(UnitSquareMesh, self).__init__(self.name, plex=dmplex)


class UnitCubeMesh(Mesh):

    """Class that represents a structured tetrahedron mesh of a 3D cube whose
    edge is a unit length.

    :arg nx: The number of the cells in the x direction.
    :arg ny: The number of the cells in the y direction.
    :arg nx: The number of the cells in the z direction.

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

    def __init__(self, nx, ny, nz):
        self.name = "unitcube_%d_%d_%d" % (nx, ny, nz)

        # Create mesh from DMPlex
        boundary = PETSc.DMPlex().create(MPI.comm)
        boundary.setDimension(2)
        boundary.createCubeBoundary([0., 0., 0.], [1., 1., 1.], [nx, ny, nz])
        dmplex = PETSc.DMPlex().generate(boundary)

        # Apply boundary IDs
        dmplex.createLabel("boundary_ids")
        dmplex.markBoundaryFaces("boundary_faces")
        coords = dmplex.getCoordinates()
        coord_sec = dmplex.getCoordinateSection()
        if dmplex.getStratumSize("boundary_faces", 1) > 0:
            boundary_faces = dmplex.getStratumIS("boundary_faces", 1).getIndices()
            for face in boundary_faces:
                face_coords = dmplex.vecGetClosure(coord_sec, coords, face)
                if face_coords[0] == 0. and face_coords[3] == 0. and face_coords[6] == 0.:
                    dmplex.setLabelValue("boundary_ids", face, 1)
                if face_coords[0] == 1. and face_coords[3] == 1. and face_coords[6] == 1.:
                    dmplex.setLabelValue("boundary_ids", face, 2)
                if face_coords[1] == 0. and face_coords[4] == 0. and face_coords[7] == 0.:
                    dmplex.setLabelValue("boundary_ids", face, 3)
                if face_coords[1] == 1. and face_coords[4] == 1. and face_coords[7] == 1.:
                    dmplex.setLabelValue("boundary_ids", face, 4)
                if face_coords[2] == 0. and face_coords[5] == 0. and face_coords[8] == 0.:
                    dmplex.setLabelValue("boundary_ids", face, 5)
                if face_coords[2] == 1. and face_coords[5] == 1. and face_coords[8] == 1.:
                    dmplex.setLabelValue("boundary_ids", face, 6)

        super(UnitCubeMesh, self).__init__(self.name, plex=dmplex)


class UnitCircleMesh(Mesh):

    """Class that represents a structured triangle mesh of a 2D circle of an
    unit circle.

    :arg resolution: The number of cells lying along the radius and the arc of
      the quadrant.
    """

    def __init__(self, resolution):
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
        self.name = "unitcircle_%d" % resolution

        output = _get_msh_file(source, self.name, 2)
        super(UnitCircleMesh, self).__init__(output)


class IntervalMesh(Mesh):
    """
    Generate a uniform mesh of the interval [0,L] for user specified L.

    :arg ncells: The number of the cells over the interval.
    :arg length: The length of the interval.

    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=L`) point has marker 2.
    """
    def __init__(self, ncells, length):
        self.name = "interval"
        dx = length / ncells
        # This ensures the rightmost point is actually present.
        coords = [[x] for x in np.arange(0, length + 0.01 * dx, dx)]
        cells = [[c, c+1] for c in range(len(coords)-1)]
        dmplex = PETSc.DMPlex().createFromCellList(1, cells, coords)

        # Apply boundary IDs
        dmplex.createLabel("boundary_ids")
        coords = dmplex.getCoordinates()
        coord_sec = dmplex.getCoordinateSection()
        vStart, vEnd = dmplex.getDepthStratum(0)  # vertices
        for v in range(vStart, vEnd):
            vcoord = dmplex.vecGetClosure(coord_sec, coords, v)
            if vcoord[0] == 0.:
                dmplex.setLabelValue("boundary_ids", v, 1)
            if vcoord[0] == 1.:
                dmplex.setLabelValue("boundary_ids", v, 2)

        super(IntervalMesh, self).__init__(self.name, plex=dmplex)


class UnitIntervalMesh(IntervalMesh):
    """
    Generate a uniform mesh of the interval [0,1].

    :arg ncells: The number of the cells over the interval.

    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=1`) point has marker 2.
    """
    def __init__(self, ncells):
        self.name = "unitinterval"
        IntervalMesh.__init__(self, ncells, length=1.0)


class PeriodicIntervalMesh(Mesh):
    """Generate a periodic uniform mesh of the interval [0, L], for
    user specified L.

    :arg ncells: The number of cells over the interval.
    :arg length: The length the interval."""
    def __init__(self, ncells, length):
        with periodic_interval_mesh(ncells) as output:
            # Coordinate values need to be replaced by the appropriate
            # DG coordinate field.
            dx = length / ncells
            # Two per cell
            coords = np.empty(2 * ncells, dtype=float)
            # For an interval
            #
            # 0---1---2---3 ... n-1---n
            # |                       |
            # `-----------------------'
            #
            # The element (0,1) is numbered first
            coords[0] = 0.0
            coords[1] = dx
            # Then the element (n, 0)
            coords[2] = length
            coords[3] = length - dx
            # Then the rest in order (1, 2), (2, 3) ... (n-1, n)
            if len(coords) > 4:
                coords[4] = dx
                coords[5:] = np.repeat(np.arange(dx * 2, length - dx + dx*0.01, dx), 2)[:-1]

            Mesh.__init__(self, output,
                          periodic_coords=coords)


class PeriodicUnitIntervalMesh(PeriodicIntervalMesh):
    """Generate a periodic uniform mesh of the interval [0, 1].
    :arg ncells: The number of cells over the interval."""
    def __init__(self, ncells):
        PeriodicIntervalMesh.__init__(self, ncells, length=1.0)


class UnitTetrahedronMesh(Mesh):

    """Class that represents a tetrahedron mesh that is composed of one
    element.
    """

    def __init__(self):
        self.name = "unittetra"
        coords = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        cells = [[1, 0, 3, 2]]
        dmplex = PETSc.DMPlex().createFromCellList(3, cells, coords)
        super(UnitTetrahedronMesh, self).__init__(self.name, plex=dmplex)


class UnitTriangleMesh(Mesh):

    """Class that represents a triangle mesh composed of one element."""

    def __init__(self):
        self.name = "unittri"
        coords = [[0., 0.], [1., 0.], [0., 1.]]
        cells = [[1, 2, 0]]
        dmplex = PETSc.DMPlex().createFromCellList(2, cells, coords)
        super(UnitTriangleMesh, self).__init__(self.name, plex=dmplex)


class IcosahedralSphereMesh(Mesh):

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

    def __init__(self, radius=1, refinement_level=0):
        """
        :arg radius: the radius of the sphere to approximate.
             For a radius R the edge length of the underlying
             icosahedron will be.

             .. math::

                a = \\frac{R}{\\sin(2 \\pi / 5)}

        :arg refinement_level: how many levels of refinement, zero
                               corresponds to an icosahedron.
        """

        self.name = "icosahedralspheremesh_%d_%g" % (refinement_level, radius)

        self._R = radius
        self._refinement = refinement_level

        self._vertices = np.empty_like(IcosahedralSphereMesh._base_vertices)
        self._faces = np.copy(IcosahedralSphereMesh._base_faces)
        # Rescale so that vertices live on sphere of specified radius
        for i, vtx in enumerate(IcosahedralSphereMesh._base_vertices):
            self._vertices[i] = self._force_to_sphere(vtx)

        for i in range(refinement_level):
            self._refine()

        dmplex = PETSc.DMPlex().createFromCellList(2, self._faces, self._vertices)
        super(IcosahedralSphereMesh, self).__init__(self.name, plex=dmplex, dim=3)

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
    def __init__(self, refinement_level=0):
        """
        :arg refinement_level: how many levels to refine the mesh.
        """
        super(UnitIcosahedralSphereMesh, self).__init__(1, refinement_level)
