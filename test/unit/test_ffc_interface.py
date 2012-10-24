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
from pyop2 import op2, ffc_interface
from ufl import *

backends = ['opencl', 'sequential', 'cuda']

@pytest.mark.xfail("not hasattr(ffc_interface.constants, 'PYOP2_VERSION')")
class TestFFCCache:
    """FFC code generation cache tests."""

    def pytest_funcarg__mass(cls, request):
        e = FiniteElement('CG', triangle, 1)
        u = TestFunction(e)
        v = TrialFunction(e)
        return u*v*dx

    def pytest_funcarg__mass2(cls, request):
        e = FiniteElement('CG', triangle, 2)
        u = TestFunction(e)
        v = TrialFunction(e)
        return u*v*dx

    def pytest_funcarg__rhs(cls, request):
        e = FiniteElement('CG', triangle, 1)
        v = TrialFunction(e)
        g = Coefficient(e)
        return g*v*ds

    def pytest_funcarg__rhs2(cls, request):
        e = FiniteElement('CG', triangle, 1)
        v = TrialFunction(e)
        f = Coefficient(e)
        g = Coefficient(e)
        return f*v*dx + g*v*ds

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

    def test_ffc_cell_kernel(self, backend, mass):
        k = ffc_interface.compile_form(mass, 'mass')
        assert 'cell_integral' in k[0].code and k[1] is None and k[2] is None

    def test_ffc_exterior_facet_kernel(self, backend, rhs):
        k = ffc_interface.compile_form(rhs, 'rhs')
        assert 'exterior_facet_integral' in k[2].code and k[0] is None and k[1] is None

    def test_ffc_cell_exterior_facet_kernel(self, backend, rhs2):
        k = ffc_interface.compile_form(rhs2, 'rhs2')
        assert 'cell_integral' in k[0].code and 'exterior_facet_integral' in k[2].code and k[1] is None

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
