"""Global test configuration."""

import pytest
from firedrake.petsc import get_external_packages


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "markif_fixture: conditional marker"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark a test that takes a while to run"
    )
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors (default: 3)"
    )
    config.addinivalue_line(
        "markers",
        "skipmumps: mark as skipped unless MUMPS is installed"
    )
    config.addinivalue_line(
        "markers",
        "skipcomplexnoslate: mark as skipped in complex mode due to lack of Slate"
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
        "skipnetgen: mark as skipped if netgen and ngsPETSc is not installed"
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
        markif_fixtures = [m for m in item.own_markers if m.name == "markif_fixture"]
        for mark in markif_fixtures:
            '''@pytest.mark.markif_fixture(*marks, **conditions)
            marks: str | pytest.mark.structures.Mark
                marks to apply if conditions are met
            conditions: dict
                dictionary of conditions; consisting of function argument keys
                and fixture values or ids
            '''
            # (function argument names, fixture ids) in a list
            fixtures = [(name, id_) for name, id_ in zip(item.callspec.params.keys(), item.callspec._idlist)]
            # If all the fixtures are in the dictionary of conditions apply all of the marks
            if all((k, str(v)) in fixtures for k, v in mark.kwargs.items()):
                for label in mark.args:
                    if isinstance(label, str):
                        item.add_marker(getattr(pytest.mark, label)())
                    else:
                        item.add_marker(label())

        if mark := item.get_closest_marker("parallel"):
            nprocs = mark.kwargs.get("nprocs", 3)
            marker = f"parallel[{nprocs}]"
            if marker not in pytest.mark._markers:
                config.addinivalue_line(
                    "markers",
                    f"{marker}: internal marker"
                )
            item.add_marker(getattr(pytest.mark, marker))

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
    from pyadjoint.tape import get_working_tape

    def fin():
        tape = get_working_tape()
        if tape is not None:
            assert len(tape.get_blocks()) == 0

    request.addfinalizer(fin)
