import pytest


@pytest.fixture(scope="function")
def benchmark(pytestconfig, request):
    bench_plugin = pytestconfig.pluginmanager.getplugin("benchmark")
    if bench_plugin is not None:
        return bench_plugin.benchmark(request)
    return lambda request: pytest.skip("pytest-benchmark plugin not installed")
