from firedrake import *
import pytest


@pytest.fixture(params=("square", "cube"))
def mesh(request):
    n = 2
    if request.param == "square":
        return UnitSquareMesh(n, n)
    elif request.param == "cube":
        return UnitCubeMesh(n, n, n)


@pytest.fixture(params=("SV", "GN", "GN2", "AS"))
def space(request, mesh):
    family = request.param
    dim = mesh.topological_dimension()
    if family == "GN":
        V = FunctionSpace(mesh, "GN", 1)
        Q = FunctionSpace(mesh, "DG", 0)
    elif family == "GN2":
        V = FunctionSpace(mesh, "GN2", 1)
        Q = FunctionSpace(mesh, "DG", 0, variant="alfeld")
    elif family == "AS":
        if dim == 2:
            V = FunctionSpace(mesh, "AS", 2)
        else:
            V = FunctionSpace(mesh, "GNH1div", dim)
        Q = FunctionSpace(mesh, "CG", 1, variant="alfeld")
    elif family == "SV":
        V = VectorFunctionSpace(mesh, "CG", dim, variant="alfeld")
        Q = FunctionSpace(mesh, "DG", dim-1, variant="alfeld")
    return V * Q


# Test that div(V) is contained in Q
def test_stokes_complex(mesh, space):
    Z = space
    z = Function(Z)
    u, p = z.subfunctions

    for k in range(len(u.dat.data)):
        u.dat.data_wo[k] = 1
        p.interpolate(div(u))
        assert norm(div(u) - p) < 1E-10
        u.dat.data_wo[k] = 0
