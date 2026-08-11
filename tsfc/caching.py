"""A fingerprint of the toolchain that generates code, for callers that cache on a form.

Firedrake indexes its kernel caches by the form being compiled. It does not index
them by the compiler that lowered the form. An edit to this toolchain leaves every
kernel already on disk in place. :func:`codegen_key` closes that gap. A caller folds
it into its own cache key. An edit anywhere in the toolchain then changes the key
for every kernel that it could have changed.

The key is fixed once, when this module is imported. A process does not recompute it
per compile. Import is the moment a process starts running a particular toolchain, so
fixing the key there keeps the cost off the compile path entirely.
"""
from __future__ import annotations

import hashlib
import os
from importlib import import_module, metadata
from pathlib import Path
from typing import Hashable

#: Import names of the packages that TSFC's output depends on.
#: TSFC lowers through FInAT, FIAT and GEM, and generates code through UFL and loopy.
_TOOLCHAIN = ("tsfc", "finat", "FIAT", "gem", "ufl", "loopy")


def stamp_source_tree(root: os.PathLike) -> Hashable:
    """Fingerprint every ``.py`` file under `root`.

    Stats each file rather than reading it, so the cost is cheap enough to pay at
    import time.

    Parameters
    ----------
    root : os.PathLike
        Directory to scan, recursively.

    Returns
    -------
    Hashable
        A digest that changes when a file under `root` is added, removed, or has its
        size or modification time change.
    """
    root = Path(root)
    entries = tuple(sorted(
        (str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns)
        for path in root.rglob("*.py")
        for stat in (path.stat(),)
    ))
    return hashlib.sha1(repr(entries).encode()).hexdigest()


def _is_editable(dist_name: str) -> bool:
    dir_info = getattr(metadata.distribution(dist_name).origin, "dir_info", None)
    return bool(getattr(dir_info, "editable", False))


def _package_stamp(name: str, distributions: dict[str, list[str]]) -> Hashable:
    module = import_module(name)
    dist_names = distributions.get(name)
    if dist_names and not _is_editable(dist_names[0]):
        return metadata.version(dist_names[0])
    return stamp_source_tree(Path(module.__file__).resolve().parent)


# `packages_distributions()` scans every installed distribution's metadata, so it is
# called once here and shared, rather than once per package in `_TOOLCHAIN`.
_DISTRIBUTIONS = metadata.packages_distributions()
_CODEGEN_KEY: Hashable = tuple(_package_stamp(name, _DISTRIBUTIONS) for name in _TOOLCHAIN)


def codegen_key() -> Hashable:
    """Fingerprint the toolchain that TSFC compiles through.

    Two calls compare equal exactly when TSFC, FInAT, FIAT, GEM, UFL and loopy were all
    unchanged at the time this module was imported.

    Returns
    -------
    Hashable
        A value suitable for folding into a `cachetools` hash key.
    """
    return _CODEGEN_KEY
