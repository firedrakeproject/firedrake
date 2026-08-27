import os

from firedrake.configuration import setup_cache_dirs

CACHE_ENV_VARS = (
    "FIREDRAKE_CACHE_DIR",
    "PYOP2_CACHE_DIR",
    "FIREDRAKE_TSFC_KERNEL_CACHE_DIR",
    "XDG_CACHE_HOME",
)


def _clear_cache_env(monkeypatch):
    for var in CACHE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_setup_cache_dirs_uses_writable_sys_prefix(monkeypatch, tmp_path):
    _clear_cache_env(monkeypatch)
    monkeypatch.setattr("sys.prefix", str(tmp_path))

    setup_cache_dirs()

    root = tmp_path.joinpath(".cache")
    assert os.environ["FIREDRAKE_CACHE_DIR"] == str(root)
    assert os.environ["PYOP2_CACHE_DIR"] == str(root.joinpath("pyop2"))
    assert os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] == str(root.joinpath("tsfc"))
    assert os.environ["XDG_CACHE_HOME"] == str(root)


def test_setup_cache_dirs_falls_back_when_sys_prefix_is_not_writable(monkeypatch, tmp_path):
    _clear_cache_env(monkeypatch)
    unwritable = tmp_path.joinpath("unwritable")
    unwritable.mkdir(mode=0o555)
    monkeypatch.setattr("sys.prefix", str(unwritable))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path.joinpath("home"))

    try:
        setup_cache_dirs()
        root = tmp_path.joinpath("home", ".cache")
        assert os.environ["FIREDRAKE_CACHE_DIR"] == str(root)
    finally:
        unwritable.chmod(0o755)


def test_setup_cache_dirs_honours_firedrake_cache_dir(monkeypatch, tmp_path):
    _clear_cache_env(monkeypatch)
    monkeypatch.setenv("FIREDRAKE_CACHE_DIR", str(tmp_path))

    setup_cache_dirs()

    assert os.environ["PYOP2_CACHE_DIR"] == str(tmp_path.joinpath("pyop2"))
    assert os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] == str(tmp_path.joinpath("tsfc"))
    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path)


def test_setup_cache_dirs_does_not_override_explicit_settings(monkeypatch, tmp_path):
    _clear_cache_env(monkeypatch)
    monkeypatch.setenv("FIREDRAKE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PYOP2_CACHE_DIR", str(tmp_path.joinpath("custom-pyop2")))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path.joinpath("custom-xdg")))

    setup_cache_dirs()

    assert os.environ["PYOP2_CACHE_DIR"] == str(tmp_path.joinpath("custom-pyop2"))
    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path.joinpath("custom-xdg"))
    assert os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] == str(tmp_path.joinpath("tsfc"))
