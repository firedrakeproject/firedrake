import glob
import os
import shutil
import subprocess
import sys

import pytest


cwd = os.path.abspath(os.path.dirname(__file__))
nb_dir = os.path.join(cwd, "..", "..", "..", "docs", "notebooks")


# Discover the notebook files by globbing the notebook directory. The notebooks
# are stored as jupytext py:percent scripts and converted to .ipynb on the fly.
@pytest.fixture(params=glob.glob(os.path.join(nb_dir, "*.py")),
                ids=lambda x: os.path.basename(x))
def py_file(request):
    # Notebook 08-composable-solvers.py still has an issue, the cell is commented out
    return os.path.abspath(request.param)


@pytest.mark.skipcomplex  # Will need to add a seperate case for a complex tutorial.
@pytest.mark.skipplot
def test_notebook_runs(py_file, tmpdir, monkeypatch, skip_dependency):
    skip_dep, dependency_skip_markers_and_reasons = skip_dependency
    basename = os.path.basename(py_file)
    if basename in ("08-composable-solvers.py", "12-HPC_demo.py"):
        if skip_dep("mumps"):
            pytest.skip("MUMPS not installed with PETSc")

    # Copy across the data files the notebooks reference at runtime (relative to
    # the working directory) and convert the py:percent script to a notebook.
    for data in glob.glob(os.path.join(nb_dir, "*.geo")) + glob.glob(os.path.join(nb_dir, "*.msh")):
        shutil.copy(data, str(tmpdir))
    image_dir = os.path.join(nb_dir, "image")
    if os.path.isdir(image_dir):
        shutil.copytree(image_dir, str(tmpdir.join("image")))

    monkeypatch.chdir(tmpdir)
    ipynb_file = str(tmpdir.join(os.path.splitext(basename)[0] + ".ipynb"))
    subprocess.check_call(["jupytext", "--to", "ipynb", "--output", ipynb_file, py_file])
    subprocess.check_call([sys.executable, "-m", "pytest", "--nbval-lax", ipynb_file])
