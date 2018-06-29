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

    degree = 3
    V = FunctionSpace(mesh, "CG", degree)  # continuous space
    W = FunctionSpace(mesh, "DG", degree)  # discontinuous space

    Q = FunctionSpace(mesh, "DG", 0)  # result space

    expression = Expression("x[0]*(x[0] + sqrt(2.0)) + x[1]")
    f = Function(V).interpolate(expression)
    g = Function(W).interpolate(expression)

    q = Function(Q).interpolate(Expression("0.0"))

    domain = "{[i]: 0 <= i < C.dofs}"
    instructions = """
    for i
        R[0, 0] = fmax(R[0, 0], fabs(C[i, 0] - D[i, 0]))
    end
    """
    par_loop(domain, instructions, dx, {'C': (f, READ), 'D': (g, READ), 'R': (q, RW)})

    assert np.allclose(q.dat.data, 0.0)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
