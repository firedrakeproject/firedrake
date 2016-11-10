import pytest


@pytest.fixture(scope="function")
def benchmark(pytestconfig, request):
    bench_plugin = pytestconfig.pluginmanager.getplugin("benchmark")
    if bench_plugin is not None:
        return bench_plugin.benchmark(request)
    return lambda request: pytest.skip("pytest-benchmark plugin not installed")


def pytest_runtest_setup(item):
    """Ensure that the assembly cache and lazy evaluation are off for tests."""
    from firedrake import parameters
    from functools import partial

    def _reset(lazy):
        parameters["pyop2_options"]["lazy_evaluation"] = lazy

    # Reset to default values after running the test item.
    item.addfinalizer(partial(_reset,
                              parameters["pyop2_options"]["lazy_evaluation"]))

    parameters["pyop2_options"]["lazy_evaluation"] = False
