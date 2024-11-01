"""Replaces functionality from the removed `firedrake_configuration` module."""

import os

from pathlib import Path


# Make a configuration that mimics the old one:
_config = {
    "options": {
        "package_manager": False,
        "minimal_petsc": False,
        "mpicc": "/opt/mpich/bin/mpicc",
        "mpicxx": "/opt/mpich/bin/mpicxx",
        "mpif90": "/opt/mpich/bin/mpif90",
        "mpiexec": "/opt/mpich/bin/mpiexec",
        "disable_ssh": False,
        "honour_petsc_dir": True,
        "with_parmetis": False,
        "slepc": True,
        "packages": [],
        "honour_pythonpath": False,
        "opencascade": False,
        "tinyasm": True,
        "petsc_int_type": "int32",
        "cache_dir": "/scratch/jbetteri/firedrake_py311_opt/.cache",
        "complex": False,
        "remove_build_files": False,
        "with_blas": None,
        "torch": "cpu",
        "netgen": True,
        "jax": False,
    },
    "environment": {},
    "additions": [],
}


def get_config():
    """Return the current configuration dictionary"""
    return None


def setup_cache_dirs():
    root = Path(os.environ.get("VIRTUAL_ENV", "~")).joinpath(".cache")
    if "PYOP2_CACHE_DIR" not in os.environ:
        os.environ["PYOP2_CACHE_DIR"] = str(root.joinpath("pyop2"))
    if 'FIREDRAKE_TSFC_KERNEL_CACHE_DIR' not in os.environ:
        os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] = str(root.joinpath("tsfc"))
