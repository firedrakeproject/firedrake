# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Global test configuration."""

import os
from itertools import product
import pytest

from pyop2 import op2
from pyop2.backends import backends


def pytest_cmdline_preparse(config, args):
    if 'PYTEST_VERBOSE' in os.environ and '-v' not in args:
        args.insert(0, '-v')
    if 'PYTEST_EXITFIRST' in os.environ and '-x' not in args:
        args.insert(0, '-x')
    if 'PYTEST_NOCAPTURE' in os.environ and '-s' not in args:
        args.insert(0, '-s')
    if 'PYTEST_TB' in os.environ and not any('--tb' in a for a in args):
        args.insert(0, '--tb=' + os.environ['PYTEST_TB'])
    else:
        # Default to short tracebacks
        args.insert(0, '--tb=short')
    if 'PYTEST_NPROCS' in os.environ and '-n' not in args:
        args.insert(0, '-n ' + os.environ['PYTEST_NPROCS'])
    if 'PYTEST_WATCH' in os.environ and '-f' not in args:
        args.insert(0, '-f')
    if 'PYTEST_LAZY' in os.environ:
        args.insert(0, '--lazy')
    if 'PYTEST_GREEDY' in os.environ:
        args.insert(0, '--greedy')


def pytest_addoption(parser):
    parser.addoption("--backend", action="append",
                     help="Selection the backend: one of %s" % backends.keys())
    parser.addoption("--lazy", action="store_true", help="Only run lazy mode")
    parser.addoption("--greedy", action="store_true", help="Only run greedy mode")


def pytest_collection_modifyitems(items):
    """Group test collection by backend instead of iterating through backends
    per test."""
    def cmp(item1, item2):
        def get_backend_param(item):
            try:
                return item.callspec.getparam("backend")
            # AttributeError if no callspec, ValueError if no backend parameter
            except:
                # If a test does not take the backend parameter, make sure it
                # is run before tests that take a backend
                return '_nobackend'

        param1 = get_backend_param(item1)
        param2 = get_backend_param(item2)

        # Group tests by backend
        if param1 < param2:
            return -1
        elif param1 > param2:
            return 1
        return 0
    items.sort(cmp=cmp)


@pytest.fixture
def skip_cuda():
    return None


@pytest.fixture
def skip_opencl():
    return None


@pytest.fixture
def skip_sequential():
    return None


@pytest.fixture
def skip_openmp():
    return None


@pytest.fixture
def skip_greedy():
    return None


@pytest.fixture
def skip_lazy():
    return None


def pytest_generate_tests(metafunc):
    """Parametrize tests to run on all backends."""

    if 'backend' in metafunc.fixturenames:

        skip_backends = set()
        # Skip backends specified on the module level
        if hasattr(metafunc.module, 'skip_backends'):
            skip_backends = skip_backends.union(
                set(metafunc.module.skip_backends))
        # Skip backends specified on the class level
        if hasattr(metafunc.cls, 'skip_backends'):
            skip_backends = skip_backends.union(
                set(metafunc.cls.skip_backends))

        # Use only backends specified on the command line if any
        if metafunc.config.option.backend:
            backend = set([x.lower() for x in metafunc.config.option.backend])
        # Otherwise use all available backends
        # FIXME: This doesn't really work since the list of backends is
        # dynamically populated as backends are imported
        else:
            backend = set(backends.keys())
        # Restrict to set of backends specified on the module level
        if hasattr(metafunc.module, 'backends'):
            backend = backend.intersection(set(metafunc.module.backends))
        # Restrict to set of backends specified on the class level
        if hasattr(metafunc.cls, 'backends'):
            backend = backend.intersection(set(metafunc.cls.backends))
        # It is preferable to run in greedy mode first, in
        # case some test create leftover computations
        lazy = []
        # Skip greedy execution by passing skip_greedy as a parameter
        if not ('skip_greedy' in metafunc.fixturenames or
                metafunc.config.option.lazy):
            lazy.append('greedy')
        # Skip lazy execution by passing skip_greedy as a parameter
        if not ('skip_lazy' in metafunc.fixturenames or
                metafunc.config.option.greedy):
            lazy.append('lazy')
        # Allow skipping individual backends by passing skip_<backend> as a
        # parameter
        backend = [b for b in backend.difference(skip_backends)
                   if not 'skip_' + b in metafunc.fixturenames]
        params = list(product(backend, lazy))
        metafunc.parametrize('backend', params or [(None, None)], indirect=True,
                             ids=['-'.join(p) for p in params])


@pytest.fixture(scope='session')
def backend(request):
    backend, lazy = request.param
    # Initialise the backend
    try:
        op2.init(backend=backend, lazy_evaluation=(lazy == 'lazy'))
    # Skip test if initialisation failed
    except:
        pytest.skip('Backend %s is not available' % backend)
    return backend
