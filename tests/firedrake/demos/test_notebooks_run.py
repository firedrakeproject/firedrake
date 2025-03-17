import glob
import os
import subprocess
import sys

import pytest


cwd = os.path.abspath(os.path.dirname(__file__))
nb_dir = os.path.join(cwd, "..", "..", "..", "docs", "notebooks")


# Discover the notebook files by globbing the notebook directory
@pytest.fixture(params=glob.glob(os.path.join(nb_dir, "*.ipynb")),
                ids=lambda x: os.path.basename(x))
def ipynb_file(request):
    # Notebook 08-composable-solvers.ipynb still has an issue, the cell is commented out
    return os.path.abspath(request.param)


@pytest.mark.skipcomplex  # Will need to add a seperate case for a complex tutorial.
@pytest.mark.skipplot
def test_notebook_runs(ipynb_file, tmpdir, monkeypatch, skip_dependency):
    if os.path.basename(ipynb_file) in ("08-composable-solvers.ipynb", "12-HPC_demo.ipynb"):
        if skip_dependency("mumps"):
            pytest.skip("MUMPS not installed with PETSc")

    monkeypatch.chdir(tmpdir)
    subprocess.check_call([sys.executable, "-m", "pytest", "--nbval-lax", ipynb_file])
