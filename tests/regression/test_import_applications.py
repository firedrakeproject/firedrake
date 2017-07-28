import pytest
import os
import subprocess
import sys


@pytest.fixture(params=["gusto", "thetis"])
def app(request):
    return request.param


@pytest.mark.skipif("FIREDRAKE_CI_TESTS" not in os.environ, reason="Not running in CI")
@pytest.mark.xfail(reason="Gusto/Thetis not yet python 3 compatible")
def test_import_app(app):
    # Have to run this in a subprocess in case the import pollutes the
    # test environment, e.g. by modifying firedrake parameters.
    subprocess.check_call([sys.executable, "-c", "import %s" % app])
