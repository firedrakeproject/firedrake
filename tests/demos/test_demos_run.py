import pytest
from os.path import abspath, basename, dirname, join, splitext
import os
import subprocess
import glob
import sys
from firedrake.petsc import get_external_packages


cwd = abspath(dirname(__file__))
demo_dir = join(cwd, "..", "..", "demos")
VTK_DEMOS = [
    "benney_luke.py",
    "burgers.py",
    "camassaholm.py",
    "geometric_multigrid.py",
    "helmholtz.py",
    "higher_order_mass_lumping.py",
    "linear_fluid_structure_interaction.py",
    "linear_wave_equation.py",
    "ma-demo.py",
    "navier_stokes.py",
    "netgen_mesh.py",
    "poisson_mixed.py",
    "qg_1layer_wave.py",
    "qgbasinmodes.py",
    "qg_winddrivengyre.py",
    "rayleigh-benard.py",
    "stokes.py",
    "test_extrusion_lsw.py",
]


# Discover the demo files by globbing the demo directory
@pytest.fixture(params=glob.glob("%s/*/*.py.rst" % demo_dir),
                ids=lambda x: basename(x))
def rst_file(request):
    return abspath(request.param)


@pytest.fixture
def env():
    env = os.environ.copy()
    env["MPLBACKEND"] = "pdf"
    return env


@pytest.fixture
def py_file(rst_file, tmpdir, monkeypatch):
    # Change to the temporary directory (monkeypatch ensures that this
    # is undone when the fixture usage disappears)
    monkeypatch.chdir(tmpdir)

    # Check if we need to generate any meshes
    geos = glob.glob("%s/*.geo" % dirname(rst_file))
    for geo in geos:
        name = "%s.msh" % splitext(basename(geo))[0]
        if os.path.exists(name):
            # No need to generate if it's already there
            continue
        try:
            subprocess.check_call(["gmsh", geo, "-format", "msh2", "-3", "-o", str(tmpdir.join(name))])
        except (subprocess.CalledProcessError, OSError):
            # Skip if unable to make mesh
            pytest.skip("Unable to generate mesh file, skipping test")

    # Get the name of the python file that pylit will make
    name = splitext(basename(rst_file))[0]
    output = str(tmpdir.join(name))
    # Convert rst demo to runnable python file
    subprocess.check_call(["pylit", rst_file, output])
    return output


@pytest.mark.skipcomplex  # Will need to add a seperate case for a complex demo.
def test_demo_runs(py_file, env):
    # Add pytest skips for missing imports or packages
    if basename(py_file) in ("stokes.py", "rayleigh-benard.py", "saddle_point_systems.py", "navier_stokes.py", "netgen_mesh.py"):
        if "mumps" not in get_external_packages():
            pytest.skip("MUMPS not installed with PETSc")

    if basename(py_file) in ("stokes.py", "rayleigh-benard.py", "saddle_point_systems.py", "qg_1layer_wave.py"):
        if "hypre" not in get_external_packages():
            pytest.skip("hypre not installed with PETSc")

    if basename(py_file) == "qgbasinmodes.py":
        try:
            # Do not use `pytest.importorskip` to check for slepc4py:
            # It isn't sufficient to actually detect whether slepc4py
            # is installed. Both petsc4py and slepc4py require
            # `from xy4py import Xy`
            # to actually load the library.
            from slepc4py import SLEPc  # noqa: F401
        except ImportError:
            pytest.skip(reason="SLEPc unavailable, skipping qgbasinmodes.py")

    if basename(py_file) in ("DG_advection.py", "qgbasinmodes.py"):
        pytest.importorskip(
            "matplotlib",
            reason=f"Matplotlib unavailable, skipping {basename(py_file)}"
        )

    if basename(py_file) == "netgen_mesh.py":
        pytest.importorskip(
            "netgen",
            reason="Netgen unavailable, skipping Netgen test."
        )
        pytest.importorskip(
            "ngsPETSc",
            reason="ngsPETSc unavailable, skipping Netgen test."
        )
        try:
            from slepc4py import SLEPc  # noqa: F401, F811
        except ImportError:
            pytest.skip(reason="SLEPc unavailable, skipping netgen_mesh.py")

    if basename(py_file) in VTK_DEMOS:
        try:
            import vtkmodules.vtkCommonDataModel  # noqa: F401
        except ImportError:
            pytest.skip(reason=f"VTK unavailable, skipping {basename(py_file)}")

    subprocess.check_call([sys.executable, py_file], env=env)
