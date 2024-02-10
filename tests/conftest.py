"""Global test configuration."""

import pytest


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "skipcomplex: mark as skipped in complex mode")
    config.addinivalue_line(
        "markers",
        "skipreal: mark as skipped unless in complex mode")
    config.addinivalue_line(
        "markers",
        "skipcomplexnoslate: mark as skipped in complex mode due to lack of Slate")
    config.addinivalue_line(
        "markers",
        "skiptorch: mark as skipped if PyTorch is not installed")
    config.addinivalue_line(
        "markers",
        "skipplot: mark as skipped if matplotlib is not installed")
    config.addinivalue_line(
        "markers",
        "skipnetgen: mark as skipped if netgen and ngsPETSc is not installed")


def pytest_collection_modifyitems(session, config, items):
    from firedrake.utils import complex_mode, SLATE_SUPPORTS_COMPLEX

    try:
        import matplotlib
        del matplotlib
        matplotlib_installed = True
    except ImportError:
        matplotlib_installed = False

    try:
        import firedrake.ml.pytorch as fd_ml
        del fd_ml
        ml_backend = True
    except ImportError:
        ml_backend = False

    try:
        import netgen
        del netgen
        import ngsPETSc
        del ngsPETSc
        netgen_installed = True
    except ImportError:
        netgen_installed = False

    for item in items:
        if complex_mode:
            if item.get_closest_marker("skipcomplex") is not None:
                item.add_marker(pytest.mark.skip(reason="Test makes no sense in complex mode"))
            if item.get_closest_marker("skipcomplexnoslate") and not SLATE_SUPPORTS_COMPLEX:
                item.add_marker(pytest.mark.skip(reason="Test skipped due to lack of Slate complex support"))
        else:
            if item.get_closest_marker("skipreal") is not None:
                item.add_marker(pytest.mark.skip(reason="Test makes no sense unless in complex mode"))

        if not ml_backend:
            if item.get_closest_marker("skiptorch") is not None:
                item.add_marker(pytest.mark.skip(reason="Test makes no sense if PyTorch is not installed"))

        if not matplotlib_installed:
            if item.get_closest_marker("skipplot") is not None:
                item.add_marker(pytest.mark.skip(reason="Test cannot be run unless Matplotlib is installed"))

        if not netgen_installed:
            if item.get_closest_marker("skipnetgen") is not None:
                item.add_marker(pytest.mark.skip(reason="Test cannot be run unless Netgen and ngsPETSc are installed"))


@pytest.fixture(scope="module", autouse=True)
def check_empty_tape(request):
    """Check that the tape is empty at the end of each module"""
    from pyadjoint.tape import get_working_tape

    def fin():
        tape = get_working_tape()
        if tape is not None:
            assert len(tape.get_blocks()) == 0

    request.addfinalizer(fin)
