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

"""Compilation flag unit tests."""

import pytest
from packaging.version import Version

from pyop2.compilation import (
    LinuxClangCompiler,
    LinuxCrayCompiler,
    LinuxGnuCompiler,
    LinuxIntelCompiler,
    MacClangARMCompiler,
    MacClangCompiler,
)
from pyop2.configuration import configuration


# Every compiler that enables relaxed floating point semantics by default, with
# the flag it uses to do so.
RELAXED_MATH_COMPILERS = [
    (MacClangCompiler, "-ffast-math"),
    (MacClangARMCompiler, "-ffast-math"),
    (LinuxGnuCompiler, "-ffast-math"),
    (LinuxClangCompiler, "-ffast-math"),
    (LinuxCrayCompiler, "-ffast-math"),
    # -Ofast implies -fp-model=fast on the Intel compilers.
    (LinuxIntelCompiler, "-Ofast"),
]


@pytest.fixture
def safe_math(request):
    """Set ``safe_math`` for the duration of a test, then restore it."""
    original = configuration["safe_math"]
    configuration["safe_math"] = request.param
    yield request.param
    configuration["safe_math"] = original


@pytest.mark.parametrize("compiler, relaxed_flag", RELAXED_MATH_COMPILERS)
@pytest.mark.parametrize("safe_math", [False], indirect=True)
def test_relaxed_math_is_the_default(compiler, relaxed_flag, safe_math):
    """Without ``safe_math`` the relaxed floating point flag is passed.

    This is the historical behaviour and the switch must not change it.
    """
    flags = compiler(version=Version("1.0")).cflags
    assert relaxed_flag in flags


@pytest.mark.parametrize("compiler, relaxed_flag", RELAXED_MATH_COMPILERS)
@pytest.mark.parametrize("safe_math", [True], indirect=True)
def test_safe_math_drops_the_relaxed_flag(compiler, relaxed_flag, safe_math):
    """With ``safe_math`` set, no flag relaxing IEEE semantics is passed."""
    flags = compiler(version=Version("1.0")).cflags
    assert relaxed_flag not in flags
    # Guard against a variant creeping in by another spelling.
    for flag in flags:
        assert "fast-math" not in flag
        assert flag != "-Ofast"
        assert flag != "-fp-model=fast"


@pytest.mark.parametrize("compiler, relaxed_flag", RELAXED_MATH_COMPILERS)
@pytest.mark.parametrize("safe_math", [True], indirect=True)
def test_safe_math_keeps_optimising(compiler, relaxed_flag, safe_math):
    """Opting in to IEEE semantics should not also give up optimisation.

    Otherwise users face a false choice between correct arithmetic and
    reasonable performance.
    """
    flags = compiler(version=Version("1.0")).cflags
    assert any(flag in ("-O2", "-O3") for flag in flags)


@pytest.mark.parametrize("safe_math", [True], indirect=True)
def test_intel_asks_for_precise_semantics(safe_math):
    """Intel needs the model stated explicitly, dropping -Ofast is not enough."""
    flags = LinuxIntelCompiler(version=Version("1.0")).cflags
    assert "-fp-model=precise" in flags


@pytest.mark.parametrize("compiler, relaxed_flag", RELAXED_MATH_COMPILERS)
def test_safe_optflags_differ_only_in_the_math_flag(compiler, relaxed_flag):
    """``_safe_optflags`` should be ``_optflags`` minus the relaxed maths.

    Kept as a static check on the class attributes so that a future edit to one
    tuple and not the other is caught here rather than by a user wondering why
    ``safe_math`` made their kernels slow.
    """
    dropped = set(compiler._optflags) - set(compiler._safe_optflags)
    added = set(compiler._safe_optflags) - set(compiler._optflags)

    assert relaxed_flag in dropped
    # Intel swaps -Ofast for -O3 plus an explicit model, everything else only
    # removes a flag.
    if compiler is LinuxIntelCompiler:
        assert added == {"-O3", "-fp-model=precise"}
    else:
        assert not added
        assert dropped == {relaxed_flag}
