from firedrake import *
from firedrake.__future__ import *
from firedrake.utils import IntType
import pytest
import numpy


@pytest.mark.parametrize("measure",
                         [dx, ds_t, ds_b, ds_tb, ds_v])
@pytest.mark.parametrize("subdomain",
                         ["top", "bottom", 1, 2])
def test_variable_layers_bcs_application(measure, subdomain):
    # 3----7              14---17
    # |    |              |    |
    # |    |              |    |
    # 2----6----9----11---13---16
    # |    |    |    |    |    |
    # |    |    |    |    |    |
    # 1----5----8----10---12---15
    # |    |
    # |    |
    # 0----4
    mesh = UnitIntervalMesh(5)
    V = VectorFunctionSpace(mesh, "DG", 0, dim=2)

    x, = SpatialCoordinate(mesh)

    selector = assemble(interpolate(
        conditional(
            real(x) < 0.2,
            as_vector([0, 3]),
            conditional(real(x) > 0.8,
                        as_vector([1, 2]),
                        as_vector([1, 1]))),
        V))

    layers = numpy.empty((5, 2), dtype=IntType)

    layers[:] = selector.dat.data_ro.real

    extmesh = ExtrudedMesh(mesh, layers=layers,
                           layer_height=0.25)

    V = FunctionSpace(extmesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*measure

    bcs = DirichletBC(V, 0, subdomain)

    A = assemble(a, bcs=bcs).M.values

    Abc = A[bcs.nodes, :][:, bcs.nodes]

    rows, cols = Abc.shape

    assert rows == cols
    assert numpy.allclose(Abc, numpy.eye(rows))

    assert numpy.allclose(numpy.unique(A[bcs.nodes, :]), [0, 1])
    assert numpy.allclose(numpy.unique(A[:, bcs.nodes]), [0, 1])


@pytest.mark.parametrize("measure",
                         [dS_h, dS_v])
@pytest.mark.parametrize("subdomain",
                         ["top", "bottom", 1, 2])
def test_variable_layers_bcs_application_interior(measure, subdomain):
    # 3----7              14---17
    # |    |              |    |
    # |    |              |    |
    # 2----6----9----11---13---16
    # |    |    |    |    |    |
    # |    |    |    |    |    |
    # 1----5----8----10---12---15
    # |    |
    # |    |
    # 0----4
    mesh = UnitIntervalMesh(5)
    V = VectorFunctionSpace(mesh, "DG", 0, dim=2)

    x, = SpatialCoordinate(mesh)

    selector = assemble(interpolate(
        conditional(
            real(x) < 0.2,
            as_vector([0, 3]),
            conditional(real(x) > 0.8,
                        as_vector([1, 2]),
                        as_vector([1, 1]))),
        V))

    layers = numpy.empty((5, 2), dtype=IntType)

    layers[:] = selector.dat.data_ro.real

    extmesh = ExtrudedMesh(mesh, layers=layers,
                           layer_height=0.25)

    V = FunctionSpace(extmesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(avg(grad(u)), avg(grad(v)))*measure

    bcs = DirichletBC(V, 0, subdomain)

    A = assemble(a, bcs=bcs).M.values

    Abc = A[bcs.nodes, :][:, bcs.nodes]

    rows, cols = Abc.shape

    assert rows == cols
    assert numpy.allclose(Abc, numpy.eye(rows))

    assert numpy.allclose(numpy.unique(A[bcs.nodes, :]), [0, 1])
    assert numpy.allclose(numpy.unique(A[:, bcs.nodes]), [0, 1])
