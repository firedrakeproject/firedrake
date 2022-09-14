from firedrake import *
import pytest
from os.path import abspath, dirname, join
from backward_compat_check_generate_functions import _get_function


# Run "mpiexec -n 7 python tests/output/backward_compat_check_generate_file.py" to generate new backward compat test files


cwd = abspath(dirname(__file__))
mesh_name = "m"
extruded_mesh_name = "m_extruded"
func_name = "f"


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize('version', ["2.0.0"])
def test_io_backward_compat_(version):
    filename = join(cwd, "backward_compat_check_files", f"backward_compat_check_file_{version}.h5")
    with CheckpointFile(filename, 'r') as afile:
        mesh = afile.load_mesh(extruded_mesh_name)
        f = afile.load_function(mesh, func_name)
    V = f.function_space()
    fe = _get_function(V)
    assert assemble(inner(f - fe, f - fe) * dx) < 2.e-11
