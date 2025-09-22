import pytest
import numpy
from firedrake import *


@pytest.fixture(params=("square", "cube"))
def mesh(request):
    n = 2
    if request.param == "square":
        return UnitSquareMesh(n, n)
    elif request.param == "cube":
        return UnitCubeMesh(n, n, n)


@pytest.fixture(params=("SV", "GN", "GN2", "GNH1div"))
def space(request, mesh):
    family = request.param
    dim = mesh.topological_dimension()
    if family == "GN":
        V = FunctionSpace(mesh, "GN", 1)
        Q = FunctionSpace(mesh, "DG", 0)
    elif family == "GN2":
        V = FunctionSpace(mesh, "GN2", 1)
        Q = FunctionSpace(mesh, "DG", 0, variant="alfeld")
    elif family == "GNH1div":
        V = FunctionSpace(mesh, "GNH1div", dim)
        Q = FunctionSpace(mesh, "CG", 1, variant="alfeld")
    elif family == "SV":
        V = VectorFunctionSpace(mesh, "CG", dim, variant="alfeld")
        Q = FunctionSpace(mesh, "DG", dim-1, variant="alfeld")
    else:
        raise ValueError("Unexpected family")
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


# Test that DirichletBC does not set derivative nodes of supersmooth H1 functions
def test_supersmooth_bcs(mesh):
    tdim = mesh.topological_dimension()
    if tdim == 3:
        V = FunctionSpace(mesh, "GNH1div", 3)
    else:
        V = FunctionSpace(mesh, "Alfeld-Sorokina", 2)

    assert V.finat_element.complex.is_macrocell()

    # check that V in H1
    assert V.ufl_element().sobolev_space == H1

    # check that V is supersmooth
    nodes = V.finat_element.fiat_equivalent.dual.nodes
    deriv_nodes = [i for i, node in enumerate(nodes) if len(node.deriv_dict)]
    assert len(deriv_nodes) == tdim + 1

    deriv_ids = V.cell_node_list[:, deriv_nodes]
    u = Function(V)

    CG = FunctionSpace(mesh, "Lagrange", 2)
    RT = FunctionSpace(mesh, "RT", 1)
    for sub in [1, (1, 2), "on_boundary"]:
        bc = DirichletBC(V, 0, sub)

        # check that we have the expected number of bc nodes
        nnodes = len(bc.nodes)
        expected = tdim * len(DirichletBC(CG, 0, sub).nodes)
        if tdim == 3:
            expected += len(DirichletBC(RT, 0, sub).nodes)
        assert nnodes == expected

        # check that the bc does not set the derivative nodes
        u.assign(111)
        u.dat.data_wo[deriv_ids] = 42
        bc.zero(u)
        assert numpy.allclose(u.dat.data_ro[deriv_ids], 42)
