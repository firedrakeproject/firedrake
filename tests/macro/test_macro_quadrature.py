import pytest
import numpy
from firedrake import *
from firedrake.mesh import plex_from_cell_list


def alfeld_split(msh):
    dim = msh.geometric_dimension()
    coords = msh.coordinates.dat.data.reshape((-1, dim))
    coords = numpy.row_stack((coords, numpy.average(coords, 0)))
    cells = [list(map(lambda i: dim+1 if i == j else i, range(dim+1))) for j in range(dim+1)]
    plex = plex_from_cell_list(dim, cells, coords, msh.comm)
    return Mesh(plex)


@pytest.fixture(params=(1, 2, 3))
def base_mesh(request):
    if request.param == 1:
        return UnitIntervalMesh(1)
    elif request.param == 2:
        return UnitTriangleMesh()
    elif request.param == 3:
        return UnitTetrahedronMesh()


@pytest.fixture(params=("iso", "alfeld"))
def variant(request):
    return request.param


@pytest.fixture
def meshes(variant, base_mesh):
    if variant == "iso":
        return tuple(MeshHierarchy(base_mesh, 1))
    elif variant == "alfeld":
        return (base_mesh, alfeld_split(base_mesh))


@pytest.mark.parametrize("degree", (1, 4,))
def test_macro_quadrature_monomial(degree, variant, meshes):
    msh = meshes[0]
    gdim = msh.geometric_dimension()
    x = SpatialCoordinate(msh)
    c = Constant(numpy.arange(1, gdim+1))
    expr = dot(c, x) ** degree
    results = [assemble(expr * dx)]

    Q = FunctionSpace(msh, "DG", 0, variant=variant)
    q = TestFunction(Q)
    f = assemble(inner(expr, q)*dx(degree=degree))
    with f.dat.vec_ro as fv:
        result = fv.sum()

    results.append(result)
    assert numpy.isclose(*results)


@pytest.mark.parametrize("degree", (1, 4,))
def test_macro_quadrature_piecewise(degree, variant, meshes):
    results = []
    for msh, v in zip(meshes, (variant, None)):
        gdim = msh.geometric_dimension()
        x = SpatialCoordinate(msh)

        if variant == "alfeld":
            vol = assemble(1*dx(domain=msh))
            x0 = Constant([(1/vol) * assemble(x[j] * dx) for j in range(gdim)])
            c = Constant(numpy.arange(1, gdim+1))
            expr = abs(dot(c, x - x0)) ** degree
        elif variant == "iso":
            vecs = list(map(Constant, numpy.row_stack([numpy.eye(gdim),
                                                       numpy.ones((max(degree-gdim, 0), gdim))])))
            expr = numpy.prod([abs(dot(c, x)) for c in vecs[:degree]])
        else:
            raise ValueError("Unexpected variant")

        Q = FunctionSpace(msh, "DG", 0, variant=v)
        q = TestFunction(Q)
        f = assemble(inner(expr, q)*dx(degree=degree))
        with f.dat.vec_ro as fv:
            result = fv.sum()

        results.append(result)
    assert numpy.isclose(*results)
