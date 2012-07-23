"""
Auto-parametrization of test cases
==================================

Passing the parameter 'backend' to any test case will auto-parametrise
that test case for all selected backends. By default all backends from
the backends dict in the backends module are selected. Backends for
which the dependencies are not installed are thereby automatically
skipped. Tests execution is grouped per backend on a per-module basis
i.e. op2.init() and op2.exit() for a backend are only called once per
module.

Selecting for which backend to run
==================================

The default backends can be overridden by passing the
`--backend=<string>` parameter on test invocation. Passing it multiple
times runs the tests for all the given backends.

Backend-specific test cases
===========================

Not passing the parameter 'backend' to a test case will cause it to
only run once for the backend that is currently initialized. It's best
to group backend-specific test cases in a separate module and not use
the 'backend' parameter for any of them, but instead use module level
setup and teardown methods:

    def setup_module(module):
        op2.init(backend='sequential', diags=0)

    def teardown_module(module):
        op2.exit()
"""

from pyop2 import op2
from pyop2.backends import backends

def pytest_addoption(parser):
    parser.addoption("--backend", action="append",
        help="Selection the backend: one of %s" % backends.keys())

# Group test collection by backend instead of iterating through backends per
# test
def pytest_collection_modifyitems(items):
    def cmp(item1, item2):
        try:
            param1 = item1.callspec.getparam("backend")
            param2 = item2.callspec.getparam("backend")
            if param1 < param2:
                return -1
            elif param1 > param2:
                return 1
        except AttributeError:
            # Function has no callspec, ignore
            pass
        except ValueError:
            # Function has no callspec, ignore
            pass
        return 0
    items.sort(cmp=cmp)

# Parametrize tests to run on all backends
def pytest_generate_tests(metafunc):
    if 'backend' in metafunc.funcargnames:
        if metafunc.config.option.backend:
            backend = map(lambda x: x.lower(), metafunc.config.option.backend)
        else:
            backend = backends.keys()
        metafunc.parametrize("backend", backend, indirect=True)

def op2_init(backend):
    if op2.backends.get_backend() != 'pyop2.void':
        op2.exit()
    op2.init(backend)

def pytest_funcarg__backend(request):
    request.cached_setup(setup=lambda: op2_init(request.param),
                         teardown=lambda backend: op2.exit(),
                         extrakey=request.param)
    return request.param
