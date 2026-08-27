"""Replaces functionality from the removed `firedrake_configuration` module."""

import os
import sys
from pathlib import Path


def setup_cache_dirs():
    root = os.environ.get("FIREDRAKE_CACHE_DIR")
    if root is None:
        prefix = Path(sys.prefix)
        base = prefix if os.access(prefix, os.W_OK) else Path.home()
        root = str(base.joinpath(".cache"))
    os.environ.setdefault("FIREDRAKE_CACHE_DIR", root)

    root = Path(root)
    if "PYOP2_CACHE_DIR" not in os.environ:
        os.environ["PYOP2_CACHE_DIR"] = str(root.joinpath("pyop2"))
    if "FIREDRAKE_TSFC_KERNEL_CACHE_DIR" not in os.environ:
        os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] = str(root.joinpath("tsfc"))
    # loopy's persistent caches go through pytools, which only listens to
    # XDG_CACHE_HOME (or platformdirs' default) for its cache location.
    if "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = str(root)
