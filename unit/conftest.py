from pyop2.backends import backends

def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default="sequential",
        help="Selection the backend: one of %s" % backends.keys())

def pytest_funcarg__backend(request):
    return request.config.option.backend
