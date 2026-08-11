import importlib
import os
import time

import tsfc
import tsfc.caching
from tsfc.caching import codegen_key, stamp_source_tree


def test_stamp_source_tree_moves_when_a_file_moves(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    stamp1 = stamp_source_tree(tmp_path)

    # A file system timestamp can be coarser than a single Python statement, so
    # force the mtime forward instead of relying on wall-clock time to pass.
    a = tmp_path / "a.py"
    os.utime(a, (a.stat().st_atime, a.stat().st_mtime + 1))
    stamp2 = stamp_source_tree(tmp_path)

    assert stamp1 != stamp2


def test_stamp_source_tree_is_stable(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")

    assert stamp_source_tree(tmp_path) == stamp_source_tree(tmp_path)


def test_codegen_key_is_stable_between_calls():
    assert codegen_key() == codegen_key()


def test_codegen_key_reflects_toolchain_source_at_import_time():
    """This is the end-to-end property that the fix exists for.

    A process that imports `tsfc.caching` after a file under `tsfc/` is edited
    must get a different `codegen_key()`. A process that imported it before the
    edit must not. `codegen_key()` is fixed at import time. It does not recompute
    per compile. So this test simulates a fresh process with a reload, rather
    than calling `codegen_key()` again in place.
    """
    edited = tsfc.__file__
    original_mtime = os.stat(edited).st_mtime

    key_before = codegen_key()
    try:
        os.utime(edited, (os.stat(edited).st_atime, original_mtime + 1))
        importlib.reload(tsfc.caching)
        key_after = codegen_key()
        assert key_before != key_after
    finally:
        os.utime(edited, (os.stat(edited).st_atime, original_mtime))
        importlib.reload(tsfc.caching)


def test_importing_tsfc_caching_is_cheap():
    """Guards against the mtime/size approach regressing to a content hash: the
    one-time cost, paid when this module is imported, must stay small."""
    t0 = time.perf_counter()
    importlib.reload(tsfc.caching)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0
