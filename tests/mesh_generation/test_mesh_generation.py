from firedrake import *
from math import pi


def run_test():
    volume = []

    def integrate_one(m):
        V = FunctionSpace(m, 'CG', 1)
        u = Function(V)
        u.interpolate(Expression("1"))
        return assemble(u * dx)
    volume.append(integrate_one(UnitIntervalMesh(3)))
    volume.append(integrate_one(UnitSquareMesh(3, 3)))
    volume.append(integrate_one(UnitCubeMesh(3, 3, 3)))
    volume.append(integrate_one(UnitCircleMesh(15)))
    volume.append(integrate_one(UnitTriangleMesh()))
    volume.append(integrate_one(UnitTetrahedronMesh()))

    expected = [1, 1, 1, pi * 0.5 ** 2, 0.5, 0.5 / 3]
    return np.array(volume) - np.array(expected)
