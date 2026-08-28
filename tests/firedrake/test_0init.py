import os
import pytest
from firedrake import *
from firedrake.configuration import setup_cache_dirs


def test_pyop3_cache_dir_set_correctly():
    assert "PYOP3_CACHE_DIR" in os.environ
    assert op2.configuration["cache_dir"] == os.environ["PYOP3_CACHE_DIR"]


CACHE_ENV_VARS = (
    "FIREDRAKE_CACHE_DIR",
    "PYOP3_CACHE_DIR",
    "FIREDRAKE_TSFC_KERNEL_CACHE_DIR",
    "XDG_CACHE_HOME",
)


@pytest.fixture
def clean_cache_env(monkeypatch):
    # monkeypatch.delenv() snapshots each variable's prior value (set or
    # unset) and restores it automatically at teardown, even though
    # setup_cache_dirs() itself writes to os.environ directly.
    for var in CACHE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_setup_cache_dirs_uses_writable_sys_prefix(clean_cache_env, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.prefix", str(tmp_path))

    setup_cache_dirs()

    root = tmp_path.joinpath(".cache")
    assert os.environ["PYOP3_CACHE_DIR"] == str(root.joinpath("pyop3"))
    assert os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] == str(root.joinpath("tsfc"))
    assert os.environ["XDG_CACHE_HOME"] == str(root)


def test_setup_cache_dirs_falls_back_when_sys_prefix_is_not_writable(clean_cache_env, monkeypatch, tmp_path):
    # os.access(..., os.W_OK) always reports True for root regardless of the
    # file mode, so a real chmod can't simulate "not writable" in CI; fake
    # os.access itself instead.
    unwritable = tmp_path.joinpath("unwritable")
    unwritable.mkdir()
    monkeypatch.setattr("sys.prefix", str(unwritable))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path.joinpath("home"))
    real_access = os.access
    monkeypatch.setattr(
        "os.access",
        lambda path, mode, *a, **kw: False if str(path) == str(unwritable) else real_access(path, mode, *a, **kw),
    )

    setup_cache_dirs()

    root = tmp_path.joinpath("home", ".cache")
    assert os.environ["PYOP3_CACHE_DIR"] == str(root.joinpath("pyop3"))


def test_setup_cache_dirs_honours_firedrake_cache_dir(clean_cache_env, monkeypatch, tmp_path):
    monkeypatch.setenv("FIREDRAKE_CACHE_DIR", str(tmp_path))

    setup_cache_dirs()

    assert os.environ["PYOP3_CACHE_DIR"] == str(tmp_path.joinpath("pyop3"))
    assert os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] == str(tmp_path.joinpath("tsfc"))
    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path)


def test_setup_cache_dirs_does_not_override_explicit_settings(clean_cache_env, monkeypatch, tmp_path):
    monkeypatch.setenv("FIREDRAKE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PYOP3_CACHE_DIR", str(tmp_path.joinpath("custom-pyop3")))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path.joinpath("custom-xdg")))

    setup_cache_dirs()

    assert os.environ["PYOP3_CACHE_DIR"] == str(tmp_path.joinpath("custom-pyop3"))
    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path.joinpath("custom-xdg"))
    assert os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] == str(tmp_path.joinpath("tsfc"))
