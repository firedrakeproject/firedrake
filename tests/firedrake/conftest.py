"""Global test configuration."""

import os
import sys

# Disable warnings for missing options when running with pytest as PETSc does
# not know what to do with the pytest arguments.
os.environ["FIREDRAKE_DISABLE_OPTIONS_LEFT"] = "1"

import pytest
from firedrake.petsc import PETSc, get_external_packages


def _skip_test_dependency(dependency):
    """
    Returns whether to skip tests with a certain dependency.

    Usually, this will return True if the dependency is not available.
    However, on CI we never want to skip tests because we should
    test all functionality there, so if the environment variable
    FIREDRAKE_CI=1 then this will always return False.
    """
    skip = True

    if os.getenv("FIREDRAKE_CI") == "1":
        return not skip

    if dependency == "slepc":
        try:
            from slepc4py import SLEPc  # noqa: F401
            del SLEPc
            return not skip
        except ImportError:
            return skip

    elif dependency == 'matplotlib':
        try:
            import matplotlib  # noqa: F401
            del matplotlib
            return not skip
        except ImportError:
            return skip

    elif dependency == "pytorch":
        try:
            import firedrake.ml.pytorch as fd_torch  # noqa: F401
            del fd_torch
            return not skip
        except ImportError:
            return skip

    elif dependency == "jax":
        try:
            import firedrake.ml.jax as fd_jax  # noqa: F401
            del fd_jax
            return not skip
        except ImportError:
            return skip

    elif dependency == "netgen":
        try:
            import netgen  # noqa: F401
            del netgen
            import ngsPETSc  # noqa: F401
            del ngsPETSc
            return not skip
        except ImportError:
            return skip

    elif dependency in ("mumps", "hypre"):
        return dependency not in get_external_packages()

    else:
        raise ValueError("Unrecognised dependency to check: {dependency = }")


dependency_skip_markers_and_reasons = (
    ("mumps", "skipmumps", "MUMPS not installed with PETSc"),
    ("hypre", "skiphypre", "hypre not installed with PETSc"),
    ("slepc", "skipslepc", "SLEPc is not installed"),
    ("pytorch", "skiptorch", "PyTorch is not installed"),
    ("jax", "skipjax", "JAX is not installed"),
    ("matplotlib", "skipplot", "Matplotlib is not installed"),
    ("netgen", "skipnetgen", "Netgen and ngsPETSc are not installed"),
)


# This allows us to check test dependencies within tests e.g. the demo tests
@pytest.fixture
def skip_dependency():
    return _skip_test_dependency, dependency_skip_markers_and_reasons


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "skipmumps: mark as skipped unless MUMPS is installed"
    )
    config.addinivalue_line(
        "markers",
        "skiphypre: mark as skipped unless hypre is installed"
    )
    config.addinivalue_line(
        "markers",
        "skipcomplex: mark as skipped in complex mode"
    )
    config.addinivalue_line(
        "markers",
        "skipreal: mark as skipped unless in complex mode"
    )
    config.addinivalue_line(
        "markers",
        "skipcomplexnoslate: mark as skipped in complex mode due to lack of Slate"
    )
    config.addinivalue_line(
        "markers",
        "skipslepc: mark as skipped if slepc4py is not installed"
    )
    config.addinivalue_line(
        "markers",
        "skiptorch: mark as skipped if PyTorch is not installed"
    )
    config.addinivalue_line(
        "markers",
        "skipjax: mark as skipped if JAX is not installed"
    )
    config.addinivalue_line(
        "markers",
        "skipplot: mark as skipped if matplotlib is not installed"
    )
    config.addinivalue_line(
        "markers",
        "skipnetgen: mark as skipped if netgen and ngsPETSc is not installed"
    )


def pytest_collection_modifyitems(session, config, items):
    from firedrake.utils import complex_mode, SLATE_SUPPORTS_COMPLEX

    for item in items:
        if complex_mode:
            if item.get_closest_marker("skipcomplex") is not None:
                item.add_marker(pytest.mark.skip(reason="Test makes no sense in complex mode"))
            if item.get_closest_marker("skipcomplexnoslate") and not SLATE_SUPPORTS_COMPLEX:
                item.add_marker(pytest.mark.skip(reason="Test skipped due to lack of Slate complex support"))
        else:
            if item.get_closest_marker("skipreal") is not None:
                item.add_marker(pytest.mark.skip(reason="Test makes no sense unless in complex mode"))

        for dep, marker, reason in dependency_skip_markers_and_reasons:
            if _skip_test_dependency(dep) and item.get_closest_marker(marker) is not None:
                item.add_marker(pytest.mark.skip(reason))


@pytest.fixture(scope="module", autouse=True)
def check_empty_tape(request):
    """Check that the tape is empty at the end of each module"""
    from pyadjoint.tape import annotate_tape, get_working_tape

    def fin():
        # make sure taping is switched off
        assert not annotate_tape()

        # make sure the tape is empty
        tape = get_working_tape()
        if tape is not None:
            assert len(tape.get_blocks()) == 0

    request.addfinalizer(fin)


class _petsc_raises:
    """Context manager for catching PETSc-raised exceptions.

    The usual `pytest.raises` exception handler is not suitable for errors
    raised inside a callback to PETSc because the error is wrapped inside a
    `PETSc.Error` object and so this context manager unpacks this to access
    the actual internal error.

    Parameters
    ----------
    exc_type :
        The exception type that is expected to be raised inside a PETSc callback.

    """
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, traceback):
        # There is a bug where 'exc_val' is occasionally the wrong thing,
        # either 'None' or some unrelated garbage collection error. In my
        # tests this error only exists for Python < 3.12.11.
        if exc_type is PETSc.Error:
            if sys.version_info < (3, 12, 11):
                return True
            else:
                if isinstance(exc_val.__cause__, self.exc_type):
                    return True


@pytest.fixture
def petsc_raises():
    # This function is needed because pytest does not support classes as fixtures.
    return _petsc_raises
