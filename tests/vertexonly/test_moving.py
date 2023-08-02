from firedrake import *
import pytest
import numpy as np
import subprocess
from mpi4py import MPI


# Utility Functions and Fixtures


@pytest.fixture(
    params=[
        "interval",
        "square",
        "squarequads",
        "extruded",
        pytest.param(
            "extrudedvariablelayers",
            marks=pytest.mark.skip(
                reason="Extruded meshes with variable layers not supported and will hang when created in parallel"
            ),
        ),
        "cube",
        "tetrahedron",
        pytest.param(
            "immersedsphere",
            # CalledProcessError is so the parallel tests correctly xfail
            marks=pytest.mark.xfail(
                raises=(subprocess.CalledProcessError, NotImplementedError),
                reason="immersed parent meshes not supported",
            ),
        ),
        "periodicrectangle",
        "shiftedmesh",
    ],
    ids=lambda x: f"{x}-mesh",
)
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "squarequads":
        return UnitSquareMesh(2, 2, quadrilateral=True)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(2, 2), 3)
    elif request.param == "extrudedvariablelayers":
        return ExtrudedMesh(
            UnitIntervalMesh(3), np.array([[0, 3], [0, 3], [0, 2]]), np.array([3, 3, 2])
        )
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        return UnitIcosahedralSphereMesh()
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)
    elif request.param == "shiftedmesh":
        m = UnitSquareMesh(10, 10)
        m.coordinates.dat.data[:] -= 0.5
        return m


@pytest.fixture(params=[0, 1, 100], ids=lambda x: f"{x}-coords")
def vertexcoords(request, parentmesh):
    size = (request.param, parentmesh.geometric_dimension())
    return pseudo_random_coords(size)


@pytest.fixture(
    params=[("CG", 2, FunctionSpace), ("DG", 2, FunctionSpace)],
    ids=lambda x: f"{x[2].__name__}({x[0]}{x[1]})",
)
def fs(request):
    return request.param


@pytest.fixture(
    params=[
        ("CG", 2, VectorFunctionSpace),
        ("N1curl", 2, FunctionSpace),
        ("N2curl", 2, FunctionSpace),
        ("N1div", 2, FunctionSpace),
        ("N2div", 2, FunctionSpace),
        pytest.param(
            ("RTCE", 2, FunctionSpace),
            marks=pytest.mark.xfail(
                raises=(subprocess.CalledProcessError, NotImplementedError),
                reason="EnrichedElement dual basis not yet defined and FIAT duals don't have a point_dict",
            ),
        ),
        pytest.param(
            ("RTCF", 2, FunctionSpace),
            marks=pytest.mark.xfail(
                raises=(subprocess.CalledProcessError, NotImplementedError),
                reason="EnrichedElement dual basis not yet defined and FIAT duals don't have a point_dict",
            ),
        ),
    ],
    ids=lambda x: f"{x[2].__name__}({x[0]}{x[1]})",
)
def vfs(request, parentmesh):
    family = request.param[0]
    # skip where the element doesn't support the cell type
    if family != "CG":
        if parentmesh.ufl_cell().cellname() == "quadrilateral":
            if not (family == "RTCE" or family == "RTCF"):
                pytest.skip(f"{family} does not support {parentmesh.ufl_cell()} cells")
        elif (
            parentmesh.ufl_cell().cellname() == "triangle"
            or parentmesh.ufl_cell().cellname() == "tetrahedron"
        ):
            if not (
                family == "N1curl"
                or family == "N2curl"
                or family == "N1div"
                or family == "N2div"
            ):
                pytest.skip(f"{family} does not support {parentmesh.ufl_cell()} cells")
        else:
            pytest.skip(f"{family} does not support {parentmesh.ufl_cell()} cells")
    return request.param


@pytest.fixture(
    params=[
        ("CG", 2, TensorFunctionSpace),
        ("BDM", 2, VectorFunctionSpace),
        ("Regge", 2, FunctionSpace),
    ],
    ids=lambda x: f"{x[2].__name__}({x[0]}{x[1]})",
)
def tfs(request, parentmesh):
    family = request.param[0]
    # skip where the element doesn't support the cell type
    if (
        family != "CG"
        and parentmesh.ufl_cell().cellname() != "triangle"
        and parentmesh.ufl_cell().cellname() != "tetrahedron"
    ):
        pytest.skip(f"{family} does not support {parentmesh.ufl_cell()} cells")
    return request.param


def pseudo_random_coords(size):
    """
    Get an array of pseudo random coordinates with coordinate elements
    between -0.5 and 1.5. The random numbers are consistent for any
    given `size` since `numpy.random.seed(0)` is called each time this
    is used.
    """
    np.random.seed(0)
    a, b = -0.5, 1.5
    return (b - a) * np.random.random_sample(size=size) + a


def cell_midpoints(m):
    """Get the coordinates of the midpoints of every cell in mesh `m`.

    :param m: The mesh to generate cell midpoints for.

    :returns: A tuple of numpy arrays `(midpoints, local_midpoints)` where
    `midpoints` are the midpoints for the entire mesh even if the mesh is
    distributed and `local_midpoints` are the midpoints of only the
    rank-local non-ghost cells."""
    m.init()
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(SpatialCoordinate(m))
    # since mesh may be distributed, the number of cells on the MPI rank
    # may not be the same on all ranks (note we exclude ghost cells
    # hence using num_cells_local = m.cell_set.size). Below local means
    # MPI rank local.
    num_cells_local = len(f.dat.data_ro)
    num_cells = MPI.COMM_WORLD.allreduce(num_cells_local, op=MPI.SUM)
    # reshape is for 1D case where f.dat.data_ro has shape (num_cells_local,)
    local_midpoints = f.dat.data_ro.reshape(
        num_cells_local, m.ufl_cell().geometric_dimension()
    )
    local_midpoints_size = np.array(local_midpoints.size)
    local_midpoints_sizes = np.empty(MPI.COMM_WORLD.size, dtype=int)
    MPI.COMM_WORLD.Allgatherv(local_midpoints_size, local_midpoints_sizes)
    midpoints = np.empty(
        (num_cells, m.ufl_cell().geometric_dimension()), dtype=local_midpoints.dtype
    )
    MPI.COMM_WORLD.Allgatherv(local_midpoints, (midpoints, local_midpoints_sizes))
    assert len(np.unique(midpoints, axis=0)) == len(midpoints)
    return midpoints, local_midpoints


def test_missing_points_behaviour_persists(parentmesh):
    """
    Generate points inside the parentmesh and check that the missing_points_behaviour
    argument persists when we move the points outside the parentmesh.
    """
    inputcoords, inputcoordslocal = cell_midpoints(parentmesh)
    # error by default
    vm = VertexOnlyMesh(parentmesh, inputcoords)
    # note we don't get an error until we try to use the coordinates
    vm.coordinates.dat.data[:] = np.full(
        (len(vm.coordinates.dat.data_ro), parentmesh.geometric_dimension()), np.inf
    )
    with pytest.raises(ValueError):
        vm.coordinates
    # error if specified
    vm = VertexOnlyMesh(parentmesh, inputcoords, missing_points_behaviour="error")
    vm.coordinates.dat.data[:] = np.full(
        (len(vm.coordinates.dat.data_ro), parentmesh.geometric_dimension()), np.inf
    )
    with pytest.raises(ValueError):
        vm.coordinates
    # warning if specified
    vm = VertexOnlyMesh(parentmesh, inputcoords, missing_points_behaviour="warn")
    vm.coordinates.dat.data[:] = np.full(
        (len(vm.coordinates.dat.data_ro), parentmesh.geometric_dimension()), np.inf
    )
    with pytest.warns(UserWarning):
        vm.coordinates
    # can surpress error
    vm = VertexOnlyMesh(parentmesh, inputcoords, missing_points_behaviour=None)
    vm.coordinates.dat.data[:] = np.full(
        (len(vm.coordinates.dat.data_ro), parentmesh.geometric_dimension()), np.inf
    )
    vm.coordinates
