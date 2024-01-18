from firedrake import *
from firedrake.__future__ import *
import numpy as np
import pytest


def at(function, point):
    return function.at(point)


def vertex_only_mesh(function, point):
    vom = VertexOnlyMesh(function.function_space().mesh(), point)
    vom_fs = VectorFunctionSpace(vom, "DG", 0)
    return assemble(interpolate(function, vom_fs))


@pytest.mark.parametrize("point_eval", [
    at,
    vertex_only_mesh,
])
def test_convergence_rate(point_eval):
    """Check points on immersed manifold projects to the correct point
    on the mesh."""
    res = [2**i for i in range(4, 10)]
    error = []

    for r in res:
        m = CircleManifoldMesh(ncells=r, radius=1e6)
        test_coords = 1e6 * np.column_stack(
            (np.cos(np.linspace(0, 2*np.pi, 3**6, endpoint=False)),
             np.sin(np.linspace(0, 2*np.pi, 3**6, endpoint=False)))
        )
        f = assemble(interpolate(SpatialCoordinate(m),
                                 VectorFunctionSpace(m, "Lagrange", 1)))
        if point_eval is at:
            sol = np.array(point_eval(f, test_coords))
            error += [np.linalg.norm(test_coords - sol)]
        elif point_eval is vertex_only_mesh:
            func = point_eval(f, test_coords)
            vom = func.function_space().ufl_domain()
            sol = np.array(func.dat.data_ro)
            error += [np.linalg.norm(test_coords[vom.topology._dm_renumbering] - sol)]

    convergence_rate = np.array(
        [np.log(error[i]/error[i+1])/np.log(res[i+1]/res[i])
         for i in range(len(res)-1)]
    )
    assert (convergence_rate > 0.9 * 2).all()
