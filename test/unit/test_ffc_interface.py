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

import pytest
ffc_interface = pytest.importorskip('pyop2.ffc_interface')
import os
from ufl import *


@pytest.mark.xfail("not hasattr(ffc_interface.constants, 'PYOP2_VERSION')")
class TestFFCCache:

    """FFC code generation cache tests."""

    @pytest.fixture
    def mass(cls):
        e = FiniteElement('CG', triangle, 1)
        u = TestFunction(e)
        v = TrialFunction(e)
        return u * v * dx

    @pytest.fixture
    def mass2(cls):
        e = FiniteElement('CG', triangle, 2)
        u = TestFunction(e)
        v = TrialFunction(e)
        return u * v * dx

    @pytest.fixture
    def rhs(cls):
        e = FiniteElement('CG', triangle, 1)
        v = TrialFunction(e)
        g = Coefficient(e)
        return g * v * ds

    @pytest.fixture
    def rhs2(cls):
        e = FiniteElement('CG', triangle, 1)
        v = TrialFunction(e)
        f = Coefficient(e)
        g = Coefficient(e)
        return f * v * dx + g * v * ds

    @pytest.fixture
    def cache_key(cls, mass):
        return ffc_interface.FFCKernel(mass, 'mass').cache_key

    def test_ffc_cache_dir_exists(self, backend):
        """Importing ffc_interface should create FFC Kernel cache dir."""
        assert os.path.exists(ffc_interface.FFCKernel._cachedir)

    def test_ffc_cache_persist_on_disk(self, backend, cache_key):
        """FFCKernel should be persisted on disk."""
        assert os.path.exists(
            os.path.join(ffc_interface.FFCKernel._cachedir, cache_key))

    def test_ffc_cache_read_from_disk(self, backend, cache_key):
        """Loading an FFCKernel from disk should yield the right object."""
        assert ffc_interface.FFCKernel._read_from_disk(
            cache_key).cache_key == cache_key

    def test_ffc_compute_form_data(self, backend, mass):
        """Compiling a form attaches form data."""
        ffc_interface.compile_form(mass, 'mass')

        assert mass.form_data()

    def test_ffc_same_form(self, backend, mass):
        """Compiling the same form twice should load kernels from cache."""
        k1 = ffc_interface.compile_form(mass, 'mass')
        k2 = ffc_interface.compile_form(mass, 'mass')

        assert k1 is k2

    def test_ffc_different_forms(self, backend, mass, mass2):
        """Compiling different forms should not load kernels from cache."""
        k1 = ffc_interface.compile_form(mass, 'mass')
        k2 = ffc_interface.compile_form(mass2, 'mass')

        assert k1 is not k2

    def test_ffc_different_names(self, backend, mass):
        """Compiling different forms should not load kernels from cache."""
        k1 = ffc_interface.compile_form(mass, 'mass')
        k2 = ffc_interface.compile_form(mass, 'mass2')

        assert k1 is not k2

    def test_ffc_cell_kernel(self, backend, mass):
        k = ffc_interface.compile_form(mass, 'mass')
        assert 'cell_integral' in k[0].code and len(k) == 1

    def test_ffc_exterior_facet_kernel(self, backend, rhs):
        k = ffc_interface.compile_form(rhs, 'rhs')
        assert 'exterior_facet_integral' in k[0].code and len(k) == 1

    def test_ffc_cell_exterior_facet_kernel(self, backend, rhs2):
        k = ffc_interface.compile_form(rhs2, 'rhs2')
        assert 'cell_integral' in k[
            0].code and 'exterior_facet_integral' in k[1].code and len(k) == 2

if __name__ == '__main__':
    pytest.main(os.path.abspath(__file__))
