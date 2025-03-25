import pytest


def pytest_collection_modifyitems(session, config, items):
    from firedrake.utils import complex_mode, SLATE_SUPPORTS_COMPLEX
    for item in items:
        test_file, *_ = item.location
        if not test_file.startswith("tests/firedrake/slate/"):
            continue
        if not SLATE_SUPPORTS_COMPLEX and complex_mode:
            item.add_marker(pytest.mark.skip(reason="Slate support for complex mode is missing"))
