import pytest
from os.path import abspath, basename, dirname, join, splitext
import os
import subprocess
import glob
import sys


cwd = abspath(dirname(__file__))
demo_dir = join(cwd, "..", "..", "demos")
pylit = join(cwd, "..", "..", "pylit", "pylit.py")


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
        try:
            subprocess.check_call(["gmsh", geo, "-format", "msh2", "-3", "-o", str(tmpdir.join(name))])
        except (subprocess.CalledProcessError, OSError):
            # Skip if unable to make mesh
            pytest.skip("Unable to generate mesh file, skipping test")

    # Get the name of the python file that pylit will make
    name = splitext(basename(rst_file))[0]
    output = str(tmpdir.join(name))
    # Convert rst demo to runnable python file
    subprocess.check_call([pylit, rst_file, output])
    return output


def test_demo_runs(py_file, env):
    subprocess.check_call([sys.executable, py_file], env=env)
