import pytest
import numpy
from firedrake import *
from firedrake.mesh import plex_from_cell_list


def alfeld_split(msh):
    dim = msh.geometric_dimension()
    coords = msh.coordinates.dat.data
    coords = numpy.row_stack((coords, numpy.average(coords, 0)))
    cells = [list(map(lambda i: dim+1 if i == j else i, range(dim+1))) for j in range(dim+1)]
    plex = plex_from_cell_list(dim, cells, coords, msh.comm)
    return Mesh(plex)


@pytest.fixture
def base_mesh():
    return UnitTriangleMesh(0)


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
def test_macro_quadrature(degree, variant, meshes):
    results = []
    for msh, v in zip(meshes, (variant, None)):
        gdim = msh.geometric_dimension()
        x = SpatialCoordinate(msh)

        if variant == "alfeld":
            vol = assemble(1*dx(domain=msh))
            x0 = Constant([(1/vol) * assemble(x[j] * dx) for j in range(gdim)])
            a = Constant(numpy.arange(1, gdim+1))
            expr = abs(dot(a, x - x0)) ** degree
        elif variant == "iso":
            if gdim == 2:
                vecs = list(map(Constant, [(1, 0), (0, 1), (1, 1), (1, 0)]))

            expr = numpy.prod([abs(dot(a, x)) for a in vecs[:degree]])
        else:
            raise ValueError("Unexpected variant")

        Q = FunctionSpace(msh, "DG", 0, variant=v)
        q = TestFunction(Q)
        c = assemble(inner(q, expr)*dx(degree=degree))
        with c.dat.vec_ro as cv:
            result = cv.sum()

        results.append(result)
    assert numpy.isclose(*results)
