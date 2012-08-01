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

"""
Auto-parametrization of test cases
==================================

Passing the parameter 'backend' to any test case will auto-parametrise
that test case for all selected backends. By default all backends from
the backends dict in the backends module are selected. Backends for
which the dependencies are not installed are thereby automatically
skipped. Tests execution is grouped per backend and op2.init() and
op2.exit() for a backend are only called once per test session.

Selecting for which backend to run
==================================

The default backends can be overridden by passing the
`--backend=<string>` parameter on test invocation. Passing it multiple
times runs the tests for all the given backends.

Skipping backends on a per-test basis
=====================================

To skip a particular backend in a test case, pass the 'skip_<backend>'
parameter to the test function, where '<backend>' is any valid backend
string.

Skipping backends on a module or class basis
============================================

You can supply a list of backends to skip for all tests in a given
module or class with the ``skip_backends`` attribute in the module or
class scope:

    # module test_foo.py

    # All tests in this module will not run for the CUDA backend
    skip_backends = ['cuda']

    class TestFoo:
        # All tests in this class will not run for the CUDA and OpenCL
        # backends
        skip_backends = ['opencl']

Selecting backends on a module or class basis
=============================================

Not passing the parameter 'backend' to a test case will cause it to
only run once for the backend that is currently initialized, which is
not always safe.

You can supply a list of backends for which to run all tests in a given
module or class with the ``backends`` attribute in the module or class
scope:

    # module test_foo.py

    # All tests in this module will only run for the CUDA and OpenCL
    # backens
    backends = ['cuda', 'opencl']

    class TestFoo:
        # All tests in this class will only run for the CUDA backend
        backends = ['sequential', 'cuda']

This set of backends to run for will be further restricted by the
backends selected via command line parameters if applicable.
"""

import pytest

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

def pytest_funcarg__skip_cuda(request):
    return None

def pytest_funcarg__skip_opencl(request):
    return None

def pytest_funcarg__skip_sequential(request):
    return None

# Parametrize tests to run on all backends
def pytest_generate_tests(metafunc):

    if 'backend' in metafunc.funcargnames:

        # Allow skipping individual backends by passing skip_<backend> as a parameter
        skip_backends = set()
        for b in backends.keys():
            if 'skip_'+b in metafunc.funcargnames:
                skip_backends.add(b)
        # Skip backends specified on the module level
        if hasattr(metafunc.module, 'skip_backends'):
            skip_backends = skip_backends.union(set(metafunc.module.skip_backends))
        # Skip backends specified on the class level
        if hasattr(metafunc.cls, 'skip_backends'):
            skip_backends = skip_backends.union(set(metafunc.cls.skip_backends))

        # Use only backends specified on the command line if any
        if metafunc.config.option.backend:
            backend = set(map(lambda x: x.lower(), metafunc.config.option.backend))
        # Otherwise use all available backends
        else:
            backend = set(backends.keys())
        # Restrict to set of backends specified on the module level
        if hasattr(metafunc.module, 'backends'):
            backend = backend.intersection(set(metafunc.module.backends))
        # Restrict to set of backends specified on the class level
        if hasattr(metafunc.cls, 'backends'):
            backend = backend.intersection(set(metafunc.cls.backends))
        # If there are no selected backends left, skip the test
        if not backend.difference(skip_backends):
            pytest.skip()
        metafunc.parametrize("backend", (b for b in backend if not b in skip_backends), indirect=True)

def op2_init(backend):
    # We need to clean up the previous backend first, because the teardown
    # hook is only run at the end of the session
    op2.exit()
    op2.init(backend)

def pytest_funcarg__backend(request):
    # If a testcase has the backend parameter but the parametrization leaves
    # i with no backends the request won't have a param, so return None
    if not hasattr(request, 'param'):
        return None
    # Call init/exit only once per session
    request.cached_setup(scope='session', setup=lambda: op2_init(request.param),
                         teardown=lambda backend: op2.exit(),
                         extrakey=request.param)
    return request.param
