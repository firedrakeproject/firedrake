import os
import pytest
from firedrake import *
from firedrake.configuration import setup_cache_dirs


def test_pyop2_custom_init():
    """PyOP2 init parameters set by the user should be retained."""
    op2.init(debug=True, log_level='CRITICAL')
    UnitIntervalMesh(2)
    import logging
    logger = logging.getLogger('pyop2')
    assert logger.getEffectiveLevel() == CRITICAL
    assert op2.configuration['debug'] is True
    op2.configuration.reset()


def test_pyop2_cache_dir_set_correctly():
    assert "PYOP2_CACHE_DIR" in os.environ
    assert op2.configuration["cache_dir"] == os.environ["PYOP2_CACHE_DIR"]


CACHE_ENV_VARS = (
    "FIREDRAKE_CACHE_DIR",
    "PYOP2_CACHE_DIR",
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
    assert os.environ["PYOP2_CACHE_DIR"] == str(root.joinpath("pyop2"))
    assert os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] == str(root.joinpath("tsfc"))
    assert os.environ["XDG_CACHE_HOME"] == str(root)


def test_setup_cache_dirs_falls_back_when_sys_prefix_is_not_writable(clean_cache_env, monkeypatch, tmp_path):
    unwritable = tmp_path.joinpath("unwritable")
    unwritable.mkdir(mode=0o555)
    monkeypatch.setattr("sys.prefix", str(unwritable))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path.joinpath("home"))

    try:
        setup_cache_dirs()
        root = tmp_path.joinpath("home", ".cache")
        assert os.environ["PYOP2_CACHE_DIR"] == str(root.joinpath("pyop2"))
    finally:
        unwritable.chmod(0o755)


def test_setup_cache_dirs_honours_firedrake_cache_dir(clean_cache_env, monkeypatch, tmp_path):
    monkeypatch.setenv("FIREDRAKE_CACHE_DIR", str(tmp_path))

    setup_cache_dirs()

    assert os.environ["PYOP2_CACHE_DIR"] == str(tmp_path.joinpath("pyop2"))
    assert os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] == str(tmp_path.joinpath("tsfc"))
    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path)


def test_setup_cache_dirs_does_not_override_explicit_settings(clean_cache_env, monkeypatch, tmp_path):
    monkeypatch.setenv("FIREDRAKE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PYOP2_CACHE_DIR", str(tmp_path.joinpath("custom-pyop2")))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path.joinpath("custom-xdg")))

    setup_cache_dirs()

    assert os.environ["PYOP2_CACHE_DIR"] == str(tmp_path.joinpath("custom-pyop2"))
    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path.joinpath("custom-xdg"))
    assert os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] == str(tmp_path.joinpath("tsfc"))
