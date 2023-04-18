from firedrake import *
import numpy as np

import pytest


def test_convergence_rate():
    res = [2**i for i in range(4, 10)]
    error = []

    for r in res:
        m = CircleManifoldMesh(ncells=r)
        x = SpatialCoordinate(m)
        test_coords = interpolate(
            x,
            functionspace.VectorFunctionSpace(m, "CG", 9)
        ).dat.data[:]
        V = FunctionSpace(m, 'Lagrange', 4)
        f = interpolate(sin(2*pi*x[0])*sin(2*pi*x[1]), V)
        sol1 = np.array(f.at(test_coords))
        sol2 = (
            np.sin(2*np.pi*test_coords[:, 0])
            * np.sin(2*np.pi*test_coords[:, 1])
        )
        error += [np.sqrt(np.linalg.norm(sol1 - sol2)/(9*r))]

    convergence_rate = np.array(
        [np.log(error[i]/error[i+1])/np.log(res[i+1]/res[i])
         for i in range(len(res)-1)]
    )

    assert (convergence_rate > 0.9 * 3).all()
