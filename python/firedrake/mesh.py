import tempfile
from core_types import Mesh
from interval import get_interval_mesh
import subprocess
from pyop2.mpi import MPI
import os
from shutil import rmtree

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


def _get_msh_file(source, name, dimension):
    """Given a source code, name and dimension  of the mesh,
    returns the name of the file that contains necessary information to build
    a mesh class. The mesh class would call _from_file method on this file
    to contruct itself.
    """

    if MPI.comm.rank == 0:
        input = os.path.join(_cachedir, name + '.geo')
        if not os.path.exists(input):
            with open(input, 'w') as f:
                f.write(source)

        output = os.path.join(_cachedir, name)

        if not os.path.exists(output + '.msh'):
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
    """

    def __init__(self, nx, ny):
        source = """
            lc = 1e-2;
            Point(1) = {0, 0, 0, lc};
            line[] = Extrude {1, 0, 0}{
                Point{1}; Layers{%d};
            };
            extrusion[] = Extrude {0, 1, 0}{
                Line{1}; Layers{%d};
            };
            Physical Line(1) = { line[1] };
            Physical Line(2) = { extrusion[0] };
            Physical Line(3) = { extrusion[2] };
            Physical Line(4) = { extrusion[3] };
            Physical Surface(1) = { extrusion[1] };
            """ % (nx, ny)
        name = "unitsquare_%d_%d" % (nx, ny)

        output = _get_msh_file(source, name, 2)
        super(UnitSquareMesh, self).__init__(output)


class UnitCubeMesh(Mesh):

    """Class that represents a structured tetrahedron mesh of a 3D cube whose
    edge is a unit length.

    :arg nx: The number of the cells in the x direction.
    :arg ny: The number of the cells in the y direction.
    :arg nx: The number of the cells in the z direction.

    The number of the elements in a mesh can be computed from 6 * nx * ny * nz,
    and the number of the vertices from (nx+1) * (ny+1) * (nz+1).
    """

    def __init__(self, nx, ny, nz):
        source = """
            lc = 1e-2;
            Point(1) = {0, 0, 0, lc};
            Extrude {1, 0, 0}{
                Point{1}; Layers{%d};
            };
            face[] = Extrude {0, 1, 0}{
                Line{1}; Layers{%d};
            };
            extrusion[] = Extrude {0, 0, 1}{
                Surface{5}; Layers{%d};
            };
            Physical Surface(1) = { face[1] };
            Physical Surface(2) = { extrusion[0] };
            Physical Surface(3) = { extrusion[2] };
            Physical Surface(4) = { extrusion[3] };
            Physical Surface(5) = { extrusion[4] };
            Physical Surface(6) = { extrusion[5] };
            Physical Volume(1) = { extrusion[1] };
            """ % (nx, ny, nz)
        name = "unitcube_%d_%d_%d" % (nx, ny, nz)

        output = _get_msh_file(source, name, 3)
        super(UnitCubeMesh, self).__init__(output)


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
        name = "unitcircle_%d" % resolution

        output = _get_msh_file(source, name, 2)
        super(UnitCircleMesh, self).__init__(output)


class UnitIntervalMesh(Mesh):

    """Class that represents a structured interval mesh a line.

    :arg nx: The number of the cells over the interval.
    """

    def __init__(self, nx):
        with get_interval_mesh(nx) as output:
            super(UnitIntervalMesh, self).__init__(output)


class UnitTetrahedronMesh(Mesh):

    """Class that represents a tetrahedron mesh that is composed of one
    element.
    """

    def __init__(self):
        source = """
            cl1 = 1;
            Point(1) = {0, 0, 0, 1};
            Point(2) = {1, 0, 0, 1};
            Point(3) = {0, 1, 0, 1};
            Point(4) = {0, 0, 1, 1};
            Line(1) = {3, 1};
            Line(2) = {1, 4};
            Line(3) = {4, 3};
            Line(4) = {3, 2};
            Line(5) = {2, 4};
            Line(6) = {2, 1};
            Line Loop(8) = {4, 5, 3};
            Plane Surface(8) = {8};
            Line Loop(10) = {1, 2, 3};
            Plane Surface(10) = {10};
            Line Loop(12) = {6, -1, 4};
            Plane Surface(12) = {12};
            Line Loop(14) = {6, 2, -5};
            Plane Surface(14) = {14};
            Surface Loop(16) = {12, 14, 10, 8};
            Volume(16) = {16};
            """
        name = "unittetra"

        output = _get_msh_file(source, name, 3)
        super(UnitTetrahedronMesh, self).__init__(output)


class UnitTriangleMesh(Mesh):

    """Class that represents a triangle mesh composed of one element."""

    def __init__(self):
        source = """
            cl1 = 1;
            Point(1) = {0, 0, 0, 1};
            Point(2) = {1, 0, 0, 1};
            Point(3) = {0, 1, 0, 1};
            Line(1) = {2, 3};
            Line(2) = {3, 1};
            Line(3) = {1, 2};
            Line Loop(5) = {1, 2, 3};
            Plane Surface(5) = {5};
            """
        name = "unittri"
        output = _get_msh_file(source, name, 2)
        super(UnitTriangleMesh, self).__init__(output)
