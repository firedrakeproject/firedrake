from firedrake import *
from os.path import abspath, dirname, join
from pyop2.mpi import COMM_WORLD
from backward_compat_check_generate_functions import _get_function


cwd = abspath(dirname(__file__))
mesh_name = "m"
extruded_mesh_name = "m_extruded"
func_name = "f"


assert COMM_WORLD.size == 7, "Use 7 processes to generate backward compat check files"
filename = join(cwd, "backward_compat_check_files", f"backward_compat_check_file_{CheckpointFile._storage_version}.h5")
mesh = Mesh("./docs/notebooks/stokes-control.msh", name=mesh_name)
extm = ExtrudedMesh(mesh, 4, name=extruded_mesh_name)
V0 = VectorFunctionSpace(extm, "P", 1, vfamily="Real", vdegree=0, dim=2)
helem1 = FiniteElement("BDMF", "triangle", 2)
velem1 = FiniteElement("DG", "interval", 2)
elem1 = HDiv(TensorProductElement(helem1, velem1))
V1 = VectorFunctionSpace(extm, elem1, dim=2)
V = V0 * V1
f = _get_function(V, name=func_name)
with CheckpointFile(filename, 'w') as afile:
    afile.save_function(f)
