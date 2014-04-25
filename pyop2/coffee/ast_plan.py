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

"""Transform the kernel's AST according to the backend we are running over."""

from ast_base import *
from ast_optimizer import AssemblyOptimizer
from ast_vectorizer import AssemblyVectorizer

# Possibile optimizations
AUTOVECT = 1        # Auto-vectorization
V_OP_PADONLY = 2    # Outer-product vectorization + extra operations
V_OP_PEEL = 3       # Outer-product vectorization + peeling
V_OP_UAJ = 4        # Outer-product vectorization + unroll-and-jam
V_OP_UAJ_EXTRA = 5  # Outer-product vectorization + unroll-and-jam + extra iters

# Track the scope of a variable in the kernel
LOCAL_VAR = 0  # Variable declared and used within the kernel
PARAM_VAR = 1  # Variable is a kernel parameter (ie declared in the signature)


class ASTKernel(object):

    """Manipulate the kernel's Abstract Syntax Tree.

    The single functionality present at the moment is provided by the
    :meth:`plan_gpu` method, which transforms the AST for GPU execution.
    """

    def __init__(self, ast):
        self.ast = ast
        self.decls, self.fors = self._visit_ast(ast, fors=[], decls={})

    def _visit_ast(self, node, parent=None, fors=None, decls=None):
        """Return lists of:

        * Declarations within the kernel
        * Loop nests
        * Dense Linear Algebra Blocks

        that will be exploited at plan creation time."""

        if isinstance(node, Decl):
            decls[node.sym.symbol] = (node, LOCAL_VAR)
            return (decls, fors)
        elif isinstance(node, For):
            fors.append((node, parent))
            return (decls, fors)
        elif isinstance(node, FunDecl):
            self.fundecl = node
            for d in node.args:
                decls[d.sym.symbol] = (d, PARAM_VAR)
        elif isinstance(node, (FlatBlock, PreprocessNode, Symbol)):
            return (decls, fors)

        for c in node.children:
            self._visit_ast(c, node, fors, decls)

        return (decls, fors)

    def plan_gpu(self):
        """Transform the kernel suitably for GPU execution.

        Loops decorated with a ``pragma pyop2 itspace`` are hoisted out of
        the kernel. The list of arguments in the function signature is
        enriched by adding iteration variables of hoisted loops. Size of
        kernel's non-constant tensors modified in hoisted loops are modified
        accordingly.

        For example, consider the following function: ::

            void foo (int A[3]) {
              int B[3] = {...};
              #pragma pyop2 itspace
              for (int i = 0; i < 3; i++)
                A[i] = B[i];
            }

        plan_gpu modifies its AST such that the resulting output code is ::

            void foo(int A[1], int i) {
              A[0] = B[i];
            }
        """

        asm = [AssemblyOptimizer(l, pre_l, self.decls) for l, pre_l in self.fors]
        for ao in asm:
            itspace_vrs, accessed_vrs = ao.extract_itspace()

            for v in accessed_vrs:
                # Change declaration of non-constant iteration space-dependent
                # parameters by shrinking the size of the iteration space
                # dimension to 1
                decl = set(
                    [d for d in self.fundecl.args if d.sym.symbol == v.symbol])
                dsym = decl.pop().sym if len(decl) > 0 else None
                if dsym and dsym.rank:
                    dsym.rank = tuple([1 if i in itspace_vrs else j
                                       for i, j in zip(v.rank, dsym.rank)])

                # Remove indices of all iteration space-dependent and
                # kernel-dependent variables that are accessed in an itspace
                v.rank = tuple([0 if i in itspace_vrs and dsym else i
                                for i in v.rank])

            # Add iteration space arguments
            self.fundecl.args.extend([Decl("int", c_sym("%s" % i))
                                     for i in itspace_vrs])

        # Clean up the kernel removing variable qualifiers like 'static'
        for decl in self.decls.values():
            d, place = decl
            d.qual = [q for q in d.qual if q not in ['static', 'const']]

        if hasattr(self, 'fundecl'):
            self.fundecl.pred = [q for q in self.fundecl.pred
                                 if q not in ['static', 'inline']]

    def plan_cpu(self, opts):
        """Transform and optimize the kernel suitably for CPU execution."""

        # Fetch user-provided options/hints on how to transform the kernel
        licm = opts.get('licm')
        slice_factor = opts.get('slice')
        vect = opts.get('vect')
        ap = opts.get('ap')
        split = opts.get('split')

        v_type, v_param = vect if vect else (None, None)

        asm = [AssemblyOptimizer(l, pre_l, self.decls) for l, pre_l in self.fors]
        for ao in asm:
            # 1) Loop-invariant code motion
            inv_outer_loops = []
            if licm:
                inv_outer_loops = ao.generalized_licm()  # noqa
                self.decls.update(ao.decls)

            # 2) Splitting
            if split:
                ao.split(split[0], split[1])

            # 3) Register tiling
            if slice_factor and v_type == AUTOVECT:
                ao.slice_loop(slice_factor)

            # 4) Vectorization
            if initialized:
                vect = AssemblyVectorizer(ao, intrinsics, compiler)
                if ap:
                    vect.align_and_pad(self.decls)
                if v_type and v_type != AUTOVECT:
                    vect.outer_product(v_type, v_param)


# These global variables capture the internal state of COFFEE
intrinsics = {}
compiler = {}
initialized = False


def init_coffee(isa, comp):
    """Initialize COFFEE."""

    global intrinsics, compiler, initialized
    intrinsics = _init_isa(isa)
    compiler = _init_compiler(comp)
    if intrinsics and compiler:
        initialized = True


def _init_isa(isa):
    """Set the intrinsics instruction set. """

    if isa == 'sse':
        return {
            'inst_set': 'SSE',
            'avail_reg': 16,
            'alignment': 16,
            'dp_reg': 2,  # Number of double values per register
            'reg': lambda n: 'xmm%s' % n
        }

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
            'symbol_load': lambda s, r, o=None: AVXLoad(s, r, o),
            'symbol_set': lambda s, r, o=None: AVXSet(s, r, o),
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
            'decl_aligned_for': '#pragma vector aligned',
            'AVX': ['-xAVX'],
            'SSE': ['-xSSE'],
            'vect_header': '#include <immintrin.h>'
        }

    if compiler == 'gnu':
        return {
            'align': lambda o: '__attribute__((aligned(%s)))' % o,
            'decl_aligned_for': '#pragma vector aligned',
            'AVX': ['-mavx'],
            'SSE': ['-msse'],
            'vect_header': '#include <immintrin.h>'
        }
