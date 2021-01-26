from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI


# Utility Functions

@pytest.fixture(params=[pytest.param("interval", marks=pytest.mark.xfail(reason="swarm not implemented in 1d")),
                        "square",
                        pytest.param("extruded", marks=pytest.mark.xfail(reason="extruded meshes not supported")),
                        "cube",
                        "tetrahedron",
                        pytest.param("immersedsphere", marks=pytest.mark.xfail(reason="immersed parent meshes not supported")),
                        pytest.param("periodicrectangle", marks=pytest.mark.xfail(reason="meshes made from coordinate fields are not supported"))])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(1, 1), 1)
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        return UnitIcosahedralSphereMesh()
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)


@pytest.fixture(params=[0, 1, 100], ids=lambda x: f"{x}-coords")
def vertexcoords(request, parentmesh):
    size = (request.param, parentmesh.geometric_dimension())
    return pseudo_random_coords(size)


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


# Function Space Generation Tests

def functionspace_tests(vm):
    # Prep
    num_cells = vm.num_cells()
    num_cells_mpi_global = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM)
    # Can create DG0 function space
    V = FunctionSpace(vm, "DG", 0)
    # Can't create with degree > 0
    with pytest.raises(ValueError):
        V = FunctionSpace(vm, "DG", 1)
    # Can create function on function spaces
    f = Function(V)
    g = Function(V)
    # Make expr which is x in 1D, x*y in 2D, x*y*z in 3D
    from functools import reduce
    from operator import mul
    expr = reduce(mul, SpatialCoordinate(vm))
    # Can interpolate and Galerkin project expressions onto functions
    f.interpolate(expr)
    g.project(expr)
    # Should have 1 DOF per cell so check DOF DataSet
    assert f.dof_dset.sizes == g.dof_dset.sizes == (num_cells, num_cells, num_cells)
    # Empty halos for functions on vertex only mesh
    assert np.allclose(f.dat.data_ro, f.dat.data_ro_with_halos)
    assert np.allclose(g.dat.data_ro, g.dat.data_ro_with_halos)
    # The function should take on the value of the expression applied to
    # the vertex only mesh coordinates (with no change to coordinate ordering)
    assert np.allclose(f.dat.data_ro, np.prod(vm.coordinates.dat.data_ro, axis=1))
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)
    # Assembly works as expected - global assembly (integration) of a
    # constant on a vertex only mesh is evaluation of that constant
    # num_vertices (globally) times
    f.interpolate(Constant(2))
    assert np.isclose(assemble(f*dx), 2*num_cells_mpi_global)


def vectorfunctionspace_tests(vm):
    # Prep
    gdim = vm.geometric_dimension()
    num_cells = vm.num_cells()
    num_cells_mpi_global = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM)
    # Can create DG0 function space
    V = VectorFunctionSpace(vm, "DG", 0)
    # Can't create with degree > 0
    with pytest.raises(ValueError):
        V = VectorFunctionSpace(vm, "DG", 1)
    # Can create functions on function spaces
    f = Function(V)
    g = Function(V)
    # Can interpolate and Galerkin project onto functions
    x = SpatialCoordinate(vm)
    f.interpolate(2*x)
    g.project(2*x)
    # Should have 1 DOF per cell so check DOF DataSet
    assert f.dof_dset.sizes == g.dof_dset.sizes == (num_cells, num_cells, num_cells)
    # Empty halos for functions on vertex only mesh
    assert np.allclose(f.dat.data_ro, f.dat.data_ro_with_halos)
    assert np.allclose(g.dat.data_ro, g.dat.data_ro_with_halos)
    # The function should take on the value of the expression applied to
    # the vertex only mesh coordinates (with no change to coordinate ordering)
    assert np.allclose(f.dat.data_ro, 2*vm.coordinates.dat.data_ro)
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)
    # Assembly works as expected - global assembly (integration) of a
    # constant on a vertex only mesh is evaluation of that constant
    # num_vertices (globally) times. Note that we get a vertex cell for
    # each geometric dimension so we have to sum over geometric
    # dimension too.
    f.interpolate(Constant([1] * gdim))
    assert np.isclose(assemble(inner(f, f)*dx), num_cells_mpi_global*gdim)


def test_functionspaces(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    functionspace_tests(vm)
    vectorfunctionspace_tests(vm)


@pytest.mark.parallel
def test_functionspaces_parallel(parentmesh, vertexcoords):
    test_functionspaces(parentmesh, vertexcoords)
