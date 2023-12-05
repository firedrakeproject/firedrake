"""Global test configuration."""

import pytest
import pytest_mpi


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


def pytest_collection_modifyitems(session, config, items):
    from firedrake.utils import complex_mode, SLATE_SUPPORTS_COMPLEX

    try:
        import firedrake.ml.pytorch as fd_ml
        del fd_ml
        ml_backend = True
    except ImportError:
        ml_backend = False

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


@pytest.fixture(scope="module", autouse=True)
def check_empty_tape(request):
    """Check that the tape is empty at the end of each module"""
    from pyadjoint.tape import get_working_tape

    def fin():
        tape = get_working_tape()
        if tape is not None:
            assert len(tape.get_blocks()) == 0

    request.addfinalizer(fin)


# @pytest.hookimpl(tryfirst=True)
# def pytest_sessionfinish(session, exitstatus):
#     """Eagerly call PETSc finalize.
#
#     This is required because of a diabolical ordering issue between petsc4py and
#     pytest-xdist setup and teardown operations (see
#     https://github.com/firedrakeproject/firedrake/issues/3247). Without this
#     modification the ordering is:
#
#         pytest init -> PETSc init -> pytest finalize -> PETSc finalize
#
#     This is problematic because pytest finalize cleans up some state that causes
#     PETSc finalize to crash. To get around this we call PETSc finalize earlier
#     in the process resulting in:
#
#         pytest init -> PETSc init -> PETSc finalize -> pytest finalize
#
#     """
#     # import must be inside the function to avoid calling petsc4py initialize here
#     from petsc4py import PETSc
#     import pyop2
#
#     pyop2.mpi._free_comms()
#
#     # skip for parallel runs where finalize must be collective and xdist is not
#     # a concern (since it is not running on this "inner" process)
#     if not pytest_mpi._is_parallel_child_process():
#         PETSc._finalize()
