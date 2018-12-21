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
import pytest
from pyop2 import op2


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
    parser.addoption("--lazy", action="store_true", help="Only run lazy mode")
    parser.addoption("--greedy", action="store_true", help="Only run greedy mode")


@pytest.fixture(autouse=True)
def initializer(request):
    lazy = request.param
    op2.configuration["lazy_evaluation"] = (lazy == "lazy")
    return lazy


@pytest.fixture
def skip_greedy():
    pass


@pytest.fixture
def skip_lazy():
    pass


def pytest_generate_tests(metafunc):
    """Parametrize tests to run on all backends."""

    lazy = []
    # Skip greedy execution by passing skip_greedy as a parameter
    if not ('skip_greedy' in metafunc.fixturenames
            or metafunc.config.option.lazy):
        lazy.append("greedy")
    # Skip lazy execution by passing skip_greedy as a parameter
    if not ('skip_lazy' in metafunc.fixturenames
            or metafunc.config.option.greedy):
        lazy.append("lazy")
    metafunc.parametrize('initializer', lazy, indirect=True)


def pytest_collection_modifyitems(items):
    """Group test collection by greedy/lazy."""
    def get_lazy(item):
        return item.callspec.getparam("initializer")
    items.sort(key=get_lazy)
