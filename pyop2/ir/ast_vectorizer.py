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

from math import ceil

from ast_base import *
import ast_plan


class LoopVectoriser(object):

    """ Loop vectorizer """

    def __init__(self, loop_optimiser):
        if not vectorizer_init:
            raise RuntimeError("Vectorizer must be initialized first.")
        self.lo = loop_optimiser
        self.intr = intrinsics
        self.comp = compiler
        self.iloops = self._inner_loops(loop_optimiser.loop_nest)
        self.padded = []

    def align_and_pad(self, decl_scope, autovect=False, only_align=False):
        """Pad all data structures accessed in the loop nest to the nearest
        multiple of the vector length. Also align them to the size of the
        vector length in order to issue aligned loads and stores. Tell about
        the alignment to the back-end compiler by adding suitable pragmas to
        loops. Finally, adjust trip count and bound of each innermost loop
        in which padded and aligned arrays are written to."""

        used_syms = [s.symbol for s in self.lo.sym]
        acc_decls = [d for s, d in decl_scope.items() if s in used_syms]

        # Padding
        if not only_align:
            for ad in acc_decls:
                d = ad[0]
                if d.sym.rank:
                    rounded = vect_roundup(d.sym.rank[-1])
                    d.sym.rank = d.sym.rank[:-1] + (rounded,)
                    self.padded.append(d.sym)

        # Alignment
        for ds in decl_scope.values():
            d, s = ds
            if d.sym.rank and s is not ast_plan.PARAM_VAR:
                d.attr.append(self.comp["align"](self.intr["alignment"]))
        if autovect:
            for l in self.iloops:
                l.pragma = self.comp["decl_aligned_for"]

        # Loop adjustment
        for l in self.iloops:
            for stm in l.children[0].children:
                sym = stm.children[0]
                if sym.rank and sym.rank[-1] == l.it_var():
                    bound = l.cond.children[1]
                    l.cond.children[1] = c_sym(vect_roundup(bound.symbol))

    def _inner_loops(self, node):
        """Find inner loops in the subtree rooted in node."""

        def find_iloops(node, loops):
            if perf_stmt(node):
                return False
            elif isinstance(node, Block):
                return any([find_iloops(s, loops) for s in node.children])
            elif isinstance(node, For):
                found = find_iloops(node.children[0], loops)
                if not found:
                    loops.append(node)
                return True

        loops = []
        find_iloops(node, loops)
        return loops


intrinsics = {}
compiler = {}
vectorizer_init = False


def init_vectorizer(isa, comp):
    global intrinsics, compiler, vectorizer_init
    intrinsics = _init_isa(isa)
    compiler = _init_compiler(comp)
    vectorizer_init = True


def _init_isa(isa):
    """Set the intrinsics instruction set. """

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


def _init_compiler(compiler):
    """Set compiler-specific keywords. """

    if compiler == 'intel':
        return {
            'align': lambda o: '__attribute__((aligned(%s)))' % o,
            'decl_aligned_for': '#pragma vector aligned'
        }


def vect_roundup(x):
    """Return x rounded up to the vector length. """
    word_len = intrinsics.get("dp_reg") or 1
    return int(ceil(x / float(word_len))) * word_len
