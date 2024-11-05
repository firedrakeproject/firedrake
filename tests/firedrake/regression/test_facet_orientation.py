"""Tests if nodes shared between cells (e.g. nodes on vertices, facets etc.)
are interpreted consistently on CG spaces by the cells that share them.
"""

from os.path import abspath, dirname, join
import pytest
import numpy as np
from firedrake import *

cwd = abspath(dirname(__file__))


@pytest.mark.parametrize('mesh_thunk',
                         [lambda: UnitSquareMesh(5, 5, quadrilateral=False),
                          lambda: UnitSquareMesh(5, 5, quadrilateral=True),
                          lambda: UnitIcosahedralSphereMesh(2),
                          lambda: UnitCubedSphereMesh(3),
                          lambda: Mesh(join(cwd, "..", "meshes",
                                            "unitsquare_unstructured_quadrilaterals.msh"))])
def test_consistent_facet_orientation(mesh_thunk):
    mesh = mesh_thunk()
    x = SpatialCoordinate(mesh)
    degree = 3
    fe_cg = FiniteElement("CG", mesh.ufl_cell(), degree, variant="equispaced")
    V = FunctionSpace(mesh, fe_cg)  # continuous space
    fe_dg = FiniteElement("DG", mesh.ufl_cell(), degree, variant="equispaced")
    W = FunctionSpace(mesh, fe_dg)  # discontinuous space

    Q = FunctionSpace(mesh, "DG", 0)  # result space

    expression = x[0]*(x[0] + sqrt(2.0)) + x[1]
    f = Function(V).interpolate(expression)
    g = Function(W).interpolate(expression)

    q = Function(Q).interpolate(Constant(0.0))

    domain = "{[i]: 0 <= i < C.dofs}"
    instructions = """
    for i
        R[0, 0] = fmax(real(R[0, 0]), abs(C[i, 0] - D[i, 0]))
    end
    """
    par_loop((domain, instructions), dx, {'C': (f, READ), 'D': (g, READ), 'R': (q, RW)})

    assert np.allclose(q.dat.data, 0.0)
