import pytest
import os
import subprocess
import glob
from firedrake.petsc import get_external_packages


try:
    import matplotlib.pyplot as plt  # noqa: 401
except ImportError:
    pytest.skip("Matplotlib not installed", allow_module_level=True)


cwd = os.path.abspath(os.path.dirname(__file__))
nb_dir = os.path.join(cwd, "..", "..", "docs", "notebooks")


# Discover the notebook files by globbing the notebook directory
@pytest.fixture(params=glob.glob(os.path.join(nb_dir, "*.ipynb")),
                ids=lambda x: os.path.basename(x))
def ipynb_file(request):
    # Notebook 08-composable-solvers.ipynb still has an issue, the cell is commented out
    return os.path.abspath(request.param)


@pytest.mark.skipcomplex  # Will need to add a seperate case for a complex tutorial.
def test_notebook_runs(ipynb_file, tmpdir, monkeypatch):
    import pytest  # Unclear why this needs to be imported again

    if os.path.basename(ipynb_file) in ("08-composable-solvers.ipynb", "12-HPC_demo.ipynb"):
        if "mumps" not in get_external_packages():
            pytest.skip("MUMPS not installed with PETSc")
    monkeypatch.chdir(tmpdir)
    pytest = os.path.join(os.environ.get("VIRTUAL_ENV"), "bin", "pytest")
    subprocess.check_call([pytest, "--nbval-lax", ipynb_file])
