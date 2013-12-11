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

from pyop2.ir.ast_base import *


class LoopVectoriser(object):

    """ Loop vectorizer """

    def __init__(self, loop_optimiser, isa, compiler):
        self.lo = loop_optimiser
        self.intr = self._set_isa(isa)
        self.comp = self._set_compiler(compiler)

    def _set_isa(self, isa):
        """Set the instruction set. """

        if isa == 'avx':
            return {
                'inst_set': 'AVX',
                'avail_reg': 16,
                'alignment': 32,
                'dp_reg': 4,  # Number of double values per register
                'reg': lambda n: 'ymm%s' % n,
                'zeroall': '_mm256_zeroall ()',
                'setzero': AVXSetZero(),
                'decl_var': '__m256d',
                'align_array': lambda p: '__attribute__((aligned(%s)))' % p,
                'symbol': lambda s, r: AVXLoad(s, r),
                'store': lambda m, r: AVXStore(m, r),
                'mul': lambda r1, r2: AVXProd(r1, r2),
                'div': lambda r1, r2: AVXDiv(r1, r2),
                'add': lambda r1, r2: AVXSum(r1, r2),
                'sub': lambda r1, r2: AVXSub(r1, r2),
                'l_perm': lambda r, f: AVXLocalPermute(r, f),
                'g_perm': lambda r1, r2, f: AVXGlobalPermute(r1, r2, f),
                'unpck_hi': lambda r1, r2: AVXUnpackHi(r1, r2),
                'unpck_lo': lambda r1, r2: AVXUnpackLo(r1, r2)
            }

    def _set_compiler(self, compiler):
        """Set compiler-specific keywords. """

        if compiler == 'intel':
            return {
                'align': lambda o: '__attribute__((aligned(%s)))' % o,
                'decl_aligned_for': '#pragma vector aligned'
            }
