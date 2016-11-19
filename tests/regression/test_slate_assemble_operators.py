import pytest
import numpy as np
from firedrake import *


def gen_mesh(cell):
    """Generate a mesh according to the cell provided."""
    if cell == interval:
        return UnitIntervalMesh(1)
    elif cell == triangle:
        return UnitSquareMesh(1, 1)
    elif cell == tetrahedron:
        return UnitCubeMesh(1, 1, 1)
    elif cell == quadrilateral:
        return UnitSquareMesh(1, 1, quadrilateral=True)
    else:
        raise ValueError("%s cell  not recognized" % cell)


@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_poisson_operator(degree, cell):
    """Assemble the Poisson operator in SLATE and
    compare with Firedrake."""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = inner(grad(u), grad(v))*dx

    P = assemble(Tensor(form))
    ref = assemble(form)

    assert np.allclose(P.M.values, ref.M.values)


@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("cell", (interval,
                                  triangle,
                                  tetrahedron,
                                  quadrilateral))
def test_helmholtz_operator(degree, cell):
    """Assemble the (nice) Helmholtz operator in SLATE and
    compare with Firedrake."""
    mesh = gen_mesh(cell)
    V = FunctionSpace(mesh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = (inner(grad(u), grad(v)) + u*v)*dx

    H = assemble(Tensor(form))
    ref = assemble(form)

    assert np.allclose(H.M.values, ref.M.values)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
