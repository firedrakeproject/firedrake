import pytest


def pytest_collection_modifyitems(session, config, items):
    from firedrake.slate.slac import SUPPORTS_COMPLEX
    from firedrake.utils import complex_mode
    for item in items:
        test_file, *_ = item.location
        if not test_file.startswith("tests/slate/"):
            continue
        if not SUPPORTS_COMPLEX and complex_mode:
            item.add_marker(pytest.mark.skip(reason="Slate support for complex mode is missing"))
