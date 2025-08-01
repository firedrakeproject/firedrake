import glob
import importlib
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from os.path import abspath, basename, dirname, join, splitext

import pyadjoint
import pytest


Demo = namedtuple("Demo", ["loc", "requirements"])


CWD = abspath(dirname(__file__))
DEMO_DIR = join(CWD, "..", "..", "..", "demos")

SERIAL_DEMOS = [
    Demo(("benney_luke", "benney_luke"), ["vtk"]),
    Demo(("boussinesq", "boussinesq"), []),
    Demo(("burgers", "burgers"), ["vtk"]),
    Demo(("camassa-holm", "camassaholm"), ["vtk"]),
    Demo(("DG_advection", "DG_advection"), ["matplotlib"]),
    Demo(("eigenvalues_QG_basinmodes", "qgbasinmodes"), ["matplotlib", "slepc", "vtk"]),
    Demo(("extruded_continuity", "extruded_continuity"), []),
    Demo(("helmholtz", "helmholtz"), ["vtk"]),
    Demo(("higher_order_mass_lumping", "higher_order_mass_lumping"), ["vtk"]),
    Demo(("immersed_fem", "immersed_fem"), []),
    Demo(("linear_fluid_structure_interaction", "linear_fluid_structure_interaction"), ["vtk"]),
    Demo(("linear-wave-equation", "linear_wave_equation"), ["vtk"]),
    Demo(("ma-demo", "ma-demo"), ["vtk"]),
    Demo(("matrix_free", "navier_stokes"), ["mumps", "vtk"]),
    Demo(("matrix_free", "poisson"), []),
    Demo(("matrix_free", "rayleigh-benard"), ["hypre", "mumps", "vtk"]),
    Demo(("matrix_free", "stokes"), ["hypre", "mumps", "vtk"]),
    Demo(("multicomponent", "multicomponent"), ["vtk, netgen"]),
    Demo(("multigrid", "geometric_multigrid"), ["vtk"]),
    Demo(("netgen", "netgen_mesh"), ["mumps", "netgen", "slepc", "vtk"]),
    Demo(("nonlinear_QG_winddrivengyre", "qg_winddrivengyre"), ["vtk"]),
    Demo(("parallel-printing", "parprint"), []),
    Demo(("poisson", "poisson_mixed"), ["vtk"]),
    Demo(("patch", "poisson_mg_patches"), []),
    Demo(("patch", "stokes_vanka_patches"), []),
    Demo(("patch", "hcurl_riesz_star"), []),
    Demo(("patch", "hdiv_riesz_star"), []),
    Demo(("quasigeostrophy_1layer", "qg_1layer_wave"), ["hypre", "vtk"]),
    Demo(("saddle_point_pc", "saddle_point_systems"), ["hypre", "mumps"]),
    Demo(("fast_diagonalisation", "fast_diagonalisation_poisson"), ["mumps"]),
    Demo(('vlasov_poisson_1d', 'vp1d'), []),
    Demo(('shape_optimization', 'shape_optimization'), ["adjoint", "vtk"])
]
PARALLEL_DEMOS = [
    Demo(("full_waveform_inversion", "full_waveform_inversion"), ["adjoint"]),
]


@pytest.fixture
def env():
    env = os.environ.copy()
    env["MPLBACKEND"] = "pdf"
    return env


def test_no_missing_demos():
    all_demo_locs = {
        demo.loc
        for demos in [SERIAL_DEMOS, PARALLEL_DEMOS]
        for demo in demos
    }
    for rst_file in glob.glob(f"{DEMO_DIR}/*/*.py.rst"):
        rst_path = Path(rst_file)
        demo_dir = rst_path.parent.name
        demo_name, _, _ = rst_path.name.split(".")
        demo_loc = (demo_dir, demo_name)
        assert demo_loc in all_demo_locs
        all_demo_locs.remove(demo_loc)
    assert not all_demo_locs, "Unrecognised demos listed"


def _maybe_skip_demo(demo, skip_dependency):
    skip_dep, dependency_skip_markers_and_reasons = skip_dependency
    # Add pytest skips for missing imports or packages

    for dep, _, reason in dependency_skip_markers_and_reasons:
        if dep in demo.requirements and skip_dep(dep):
            pytest.skip(reason)


def _prepare_demo(demo, monkeypatch, tmpdir):
    # Change to the temporary directory (monkeypatch ensures that this
    # is undone when the fixture usage disappears)
    monkeypatch.chdir(tmpdir)

    demo_dir, demo_name = demo.loc
    rst_file = f"{DEMO_DIR}/{demo_dir}/{demo_name}.py.rst"

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
    py_file = str(tmpdir.join(name))
    # Convert rst demo to runnable python file
    subprocess.check_call(["pylit", rst_file, py_file])
    return Path(py_file)


def _exec_file(py_file):
    # To execute a file we import it. We therefore need to modify sys.path so the
    # tempdir can be found.
    sys.path.insert(0, str(py_file.parent))
    importlib.import_module(py_file.with_suffix("").name)
    sys.path.pop(0)  # cleanup


@pytest.mark.skipcomplex
@pytest.mark.parametrize("demo", SERIAL_DEMOS, ids=["/".join(d.loc) for d in SERIAL_DEMOS])
def test_serial_demo(demo, env, monkeypatch, tmpdir, skip_dependency):
    _maybe_skip_demo(demo, skip_dependency)
    py_file = _prepare_demo(demo, monkeypatch, tmpdir)
    _exec_file(py_file)

    if "adjoint" in demo.requirements:
        pyadjoint.pause_annotation()
        pyadjoint.get_working_tape().clear_tape()


@pytest.mark.parallel(2)
@pytest.mark.skipcomplex
@pytest.mark.parametrize("demo", PARALLEL_DEMOS, ids=["/".join(d.loc) for d in PARALLEL_DEMOS])
def test_parallel_demo(demo, env, monkeypatch, tmpdir, skip_dependency):
    _maybe_skip_demo(demo, skip_dependency)
    py_file = _prepare_demo(demo, monkeypatch, tmpdir)
    _exec_file(py_file)

    if "adjoint" in demo.requirements:
        pyadjoint.pause_annotation()
        pyadjoint.get_working_tape().clear_tape()
