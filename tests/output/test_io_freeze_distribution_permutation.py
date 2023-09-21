import pytest
from firedrake import *
from pyop2.mpi import COMM_WORLD
import numpy as np
import os

cwd = os.path.abspath(os.path.dirname(__file__))
mesh_name = "m"
func_name = "f"


@pytest.mark.parallel(nprocs=7)
@pytest.mark.parametrize('case', ["interval",
                                  "interval_small",
                                  "interval_periodic",
                                  "triangle",
                                  "triangle_small",
                                  "quadrilateral",
                                  "triangle_periodic",
                                  "quadrilateral_periodic",
                                  "triangle_extrusion"])
def test_io_freeze_dist_perm_base(case, tmpdir):
    filename = os.path.join(str(tmpdir), "test_io_freeze_dist_perm_base_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    if case == "interval":
        mesh = UnitIntervalMesh(17, name=mesh_name)
        V = FunctionSpace(mesh, "P", 5)
        x, = SpatialCoordinate(mesh)
        f = Function(V, name=func_name).interpolate(cos(x))
    elif case == "interval_small":
        mesh = UnitIntervalMesh(1, name=mesh_name)
        V = FunctionSpace(mesh, "P", 3)
        x, = SpatialCoordinate(mesh)
        f = Function(V, name=func_name).interpolate(cos(x))
    elif case == "interval_periodic":
        mesh = PeriodicUnitIntervalMesh(7, name=mesh_name)
        V = FunctionSpace(mesh, "P", 5)
        x, = SpatialCoordinate(mesh)
        f = Function(V, name=func_name).interpolate(cos(2 * pi * x))
    elif case == "triangle":
        mesh = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name)
        V = FunctionSpace(mesh, "BDMF", 3)
        x, y = SpatialCoordinate(mesh)
        f = Function(V, name=func_name).interpolate(as_vector([cos(x), 2 * sin(y)]))
    elif case == "triangle_small":
        mesh = UnitSquareMesh(2, 2, name=mesh_name)
        V = FunctionSpace(mesh, "BDME", 2)
        x, y = SpatialCoordinate(mesh)
        f = Function(V, name=func_name).interpolate(as_vector([cos(x), 2 * sin(y)]))
    elif case == "quadrilateral":
        mesh = Mesh(os.path.join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"), name=mesh_name)
        V = FunctionSpace(mesh, "RTCE", 3)
        x, y = SpatialCoordinate(mesh)
        f = Function(V, name=func_name).project(as_vector([cos(x), 2 * sin(y)]))
    elif case == "triangle_periodic":
        mesh = PeriodicUnitSquareMesh(20, 20, direction="both", name=mesh_name)
        x, y = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, "P", 4)
        f = Function(V, name=func_name).interpolate(cos(2 * pi * x) + sin(4 * pi * y))
    elif case == "quadrilateral_periodic":
        mesh = PeriodicUnitSquareMesh(20, 20, quadrilateral=True, direction="both", name=mesh_name)
        x, y = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, "CG", 4)
        f = Function(V, name=func_name).interpolate(cos(2 * pi * x) + cos(8 * pi * y))
    elif case == "triangle_extrusion":
        base_mesh = UnitSquareMesh(10, 10)
        mesh = ExtrudedMesh(base_mesh, layers=4, name=mesh_name)
        V = VectorFunctionSpace(mesh, "CG", 2)
        x, y, z = SpatialCoordinate(mesh)
        f = Function(V, name=func_name).interpolate(as_vector([cos(x), cos(y), cos(z)]))
    else:
        raise NotImplementedError("no such test")
    ref = f.copy(deepcopy=True)
    for i in range(4):
        with CheckpointFile(filename, "w") as cf:
            cf.save_mesh(mesh)
            cf.save_function(f)
        with CheckpointFile(filename, "r") as cf:
            mesh = cf.load_mesh(mesh_name)
            f = cf.load_function(mesh, func_name)
        assert np.allclose(f.dat.data_ro, ref.dat.data_ro)
