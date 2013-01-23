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

"""Provides the interface to FFC for compiling a form, and transforms the FFC-
generated code in order to make it suitable for passing to the backends."""

from ufl import Form
from ufl.algorithms import as_form
from ffc import default_parameters, compile_form as ffc_compile_form
from ffc import constants
from ffc.log import set_level, ERROR
from ffc.jitobject import JITObject
import re

from op2 import Kernel

_form_cache = {}

def compile_form(form, name):
    """Compile a form using FFC and return an OP2 kernel"""

    # Check that we get a Form
    if not isinstance(form, Form):
        form = as_form(form)

    ffc_parameters = default_parameters()
    ffc_parameters['write_file'] = False
    ffc_parameters['format'] = 'pyop2'

    # Silence FFC
    set_level(ERROR)

    # As of UFL 1.0.0-2 a form signature is stable w.r.t. to Coefficient/Index
    # counts
    key = form.signature()
    # Check the cache first: this saves recompiling the form for every time
    # step in time-varying problems
    kernels, form_data = _form_cache.get(key, (None, None))
    if form_data is None:
        code = ffc_compile_form(form, prefix=name, parameters=ffc_parameters)
        form_data = form.form_data()

        kernels = [ Kernel(code, '%s_%s_integral_0_%s' % (name, m.domain_type(), \
                                                          m.domain_id().subdomain_ids()[0])) \
                    for m in map(lambda x: x.measure(), form.integrals()) ]
        kernels = tuple(kernels)
        _form_cache[key] = kernels, form_data

    # Attach the form data FFC has computed for our form (saves preprocessing
    # the form later on)
    form._form_data = form_data
    return kernels
