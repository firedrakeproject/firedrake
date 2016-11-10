import pytest
import os


@pytest.mark.skipif("FIREDRAKE_CI_TESTS" not in os.environ, reason="Not running in CI")
def test_import_gusto():
    import gusto  # NOQA


@pytest.mark.skipif("FIREDRAKE_CI_TESTS" not in os.environ, reason="Not running in CI")
def test_import_thetis():
    import thetis # NOQA
