import firedrake
import numpy as np
mesh = firedrake.UnitCubeMesh(1, 1, 1)
from mpi4py import MPI


def identity(family, degree):
    fs = firedrake.FunctionSpace(mesh, family, degree)

    f = firedrake.Function(fs)
    out = firedrake.Function(fs)

    u = firedrake.TrialFunction(fs)
    v = firedrake.TestFunction(fs)

    firedrake.assemble(u * v * firedrake.dx)

    f.interpolate(firedrake.Expression("x[0]"))
    if degree == 1:
        file = firedrake.io.File("cube.pvd")
        file << f
    firedrake.assemble(f * v * firedrake.dx)
    firedrake.solve(u * v * firedrake.dx == f * v * firedrake.dx, out)

    err = np.max(np.abs(out.dat.data - f.dat.data))
    MPI.COMM_WORLD.allreduce(MPI.IN_PLACE, err, MPI.MAX)
    return err


def run_test():
    family = "Lagrange"
    degree = range(1, 5)
    error = []
    for d in degree:
        error.append(identity(family, d))
    return error


if __name__ == "__main__":

    error = run_test()
    import pickle
    if MPI.COMM_WORLD.rank == 0:
        with open("test-output.dat", "w") as f:
            pickle.dump(error, f)
