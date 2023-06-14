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

from pyop2 import op2
from pyop2.mpi import COMM_WORLD


def test_global_operations():
    g1 = op2.Global(1, data=2., comm=COMM_WORLD)
    g2 = op2.Global(1, data=5., comm=COMM_WORLD)

    assert (g1 + g2).data == 7.
    assert (g2 - g1).data == 3.
    assert (-g2).data == -5.
    assert (g1 * g2).data == 10.
    g1 *= g2
    assert g1.data == 10.


def test_global_dat_version():
    g1 = op2.Global(1, data=1., comm=COMM_WORLD)
    g2 = op2.Global(1, data=2., comm=COMM_WORLD)

    assert g1.dat_version == 0
    assert g2.dat_version == 0

    # Access data property
    d1 = g1.data

    assert g1.dat_version == 1
    assert g2.dat_version == 0

    # Access data property
    g2.data[:] += 1

    assert g1.dat_version == 1
    assert g2.dat_version == 1

    # Access zero property
    g1.zero()

    assert g1.dat_version == 2
    assert g2.dat_version == 1

    # Access data setter
    g2.data = d1

    assert g1.dat_version == 2
    assert g2.dat_version == 2
