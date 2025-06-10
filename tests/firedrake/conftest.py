"""Global test configuration."""

import pytest
from firedrake.petsc import PETSc, get_external_packages


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "skipmumps: mark as skipped unless MUMPS is installed"
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
        "skipvtk: mark as skipped if vtk is not installed"
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

    try:
        import matplotlib
        del matplotlib
        matplotlib_installed = True
    except ImportError:
        matplotlib_installed = False

    try:
        import firedrake.ml.pytorch as fd_torch
        del fd_torch
        torch_backend = True
    except ImportError:
        torch_backend = False

    try:
        import firedrake.ml.jax as fd_jax
        del fd_jax
        jax_backend = True
    except ImportError:
        jax_backend = False

    try:
        import netgen
        del netgen
        import ngsPETSc
        del ngsPETSc
        netgen_installed = True
    except ImportError:
        netgen_installed = False

    try:
        from firedrake.output import VTKFile
        del VTKFile
        vtk_installed = True
    except ImportError:
        vtk_installed = False

    for item in items:
        if complex_mode:
            if item.get_closest_marker("skipcomplex") is not None:
                item.add_marker(pytest.mark.skip(reason="Test makes no sense in complex mode"))
            if item.get_closest_marker("skipcomplexnoslate") and not SLATE_SUPPORTS_COMPLEX:
                item.add_marker(pytest.mark.skip(reason="Test skipped due to lack of Slate complex support"))
        else:
            if item.get_closest_marker("skipreal") is not None:
                item.add_marker(pytest.mark.skip(reason="Test makes no sense unless in complex mode"))

        if "mumps" not in get_external_packages():
            if item.get_closest_marker("skipmumps") is not None:
                item.add_marker(pytest.mark.skip("MUMPS not installed with PETSc"))

        if not torch_backend:
            if item.get_closest_marker("skiptorch") is not None:
                item.add_marker(pytest.mark.skip(reason="Test makes no sense if PyTorch is not installed"))

        if not jax_backend:
            if item.get_closest_marker("skipjax") is not None:
                item.add_marker(pytest.mark.skip(reason="Test makes no sense if JAX is not installed"))

        if not matplotlib_installed:
            if item.get_closest_marker("skipplot") is not None:
                item.add_marker(pytest.mark.skip(reason="Test cannot be run unless Matplotlib is installed"))

        if not netgen_installed:
            if item.get_closest_marker("skipnetgen") is not None:
                item.add_marker(pytest.mark.skip(reason="Test cannot be run unless Netgen and ngsPETSc are installed"))

        if not vtk_installed:
            if item.get_closest_marker("skipvtk") is not None:
                item.add_marker(pytest.mark.skip(reason="Test cannot be run unless VTK is installed"))


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
        if exc_type is PETSc.Error and isinstance(exc_val.__cause__, self.exc_type):
            return True


@pytest.fixture
def petsc_raises():
    # This function is needed because pytest does not support classes as fixtures.
    return _petsc_raises
