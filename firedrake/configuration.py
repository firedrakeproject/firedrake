"""Replaces functionality from the removed `firedrake_configuration` module."""

import os
from pathlib import Path


def setup_cache_dirs():
    root = Path(os.environ.get("VIRTUAL_ENV", "~")).joinpath(".cache")
    if "PYOP2_CACHE_DIR" not in os.environ:
        os.environ["PYOP2_CACHE_DIR"] = str(root.joinpath("pyop2"))
    if 'FIREDRAKE_TSFC_KERNEL_CACHE_DIR' not in os.environ:
        os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] = str(root.joinpath("tsfc"))
