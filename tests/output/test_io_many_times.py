import pytest
from firedrake import *
import os


def test_io_many_times(tmpdir):
    filename = os.path.join(str(tmpdir), "test_io_many_times_dump.h5")
    mesh = UnitSquareMesh(nx=15, ny=15,
                          quadrilateral=True, name="UnitSquareMesh")
    Q = FunctionSpace(mesh, "CG", 1)
    T = Function(Q)
    with CheckpointFile(filename, "a") as outfile:
        for timestep in range(0, 1000):
            outfile.save_function(T, name=f"AAAAAAAAAAA_{timestep}")
