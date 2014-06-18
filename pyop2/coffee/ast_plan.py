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
from ast_linearalgebra import AssemblyLinearAlgebra
from ast_autotuner import Autotuner

from copy import deepcopy as dcopy
import itertools
import operator


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

    def __init__(self, ast, include_dirs=[]):
        self.ast = ast
        # Used in case of autotuning
        self.include_dirs = include_dirs
        # Track applied optimizations
        self.blas = False
        self.ap = False

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

        decls, fors = self._visit_ast(self.ast, fors=[], decls={})
        asm = [AssemblyOptimizer(l, pre_l, decls) for l, pre_l in fors]
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
        for decl in decls.values():
            d, place = decl
            d.qual = [q for q in d.qual if q not in ['static', 'const']]

        if hasattr(self, 'fundecl'):
            self.fundecl.pred = [q for q in self.fundecl.pred
                                 if q not in ['static', 'inline']]

    def plan_cpu(self, opts):
        """Transform and optimize the kernel suitably for CPU execution."""

        # Unrolling threshold when autotuning
        autotune_unroll_ths = 10
        # The higher, the more precise and costly is autotuning
        autotune_resolution = 100000000
        # Kernel variants tested when autotuning is enabled
        autotune_minimal = [('licm', 1, False, (None, None), True, None, False, None, False),
                            ('split', 3, False, (None, None), True, (1, 0), False, None, False),
                            ('vect', 2, False, (V_OP_UAJ, 1), True, None, False, None, False)]
        autotune_all = [('licm', 1, False, (None, None), True, None, False, None, False),
                        ('licm', 2, False, (None, None), True, None, False, None, False),
                        ('licm', 3, False, (None, None), True, None, False, None, False),
                        ('licm', 3, False, (None, None), True, None, False, None, True),
                        ('split', 3, False, (None, None), True, (1, 0), False, None, False),
                        ('split', 3, False, (None, None), True, (2, 0), False, None, False),
                        ('split', 3, False, (None, None), True, (4, 0), False, None, False),
                        ('vect', 2, False, (V_OP_UAJ, 1), True, None, False, None, False),
                        ('vect', 2, False, (V_OP_UAJ, 2), True, None, False, None, False),
                        ('vect', 2, False, (V_OP_UAJ, 3), True, None, False, None, False)]

        def _generate_cpu_code(self, licm, slice_factor, vect, ap, split, blas, unroll, permute):
            """Generate kernel code according to the various optimization options."""

            v_type, v_param = vect

            # Combining certain optimizations is meaningless/forbidden.
            if unroll and blas:
                raise RuntimeError("COFFEE Error: cannot unroll and then convert to BLAS")
            if permute and blas:
                raise RuntimeError("COFFEE Error: cannot permute and then convert to BLAS")
            if permute and licm != 3:
                raise RuntimeError("COFFEE Error: cannot permute without full expression rewriter")
            if unroll and v_type and v_type != AUTOVECT:
                raise RuntimeError("COFFEE Error: outer-product vectorization needs no unroll")
            if permute and v_type and v_type != AUTOVECT:
                raise RuntimeError("COFFEE Error: outer-product vectorization needs no permute")

            decls, fors = self._visit_ast(self.ast, fors=[], decls={})
            asm = [AssemblyOptimizer(l, pre_l, decls) for l, pre_l in fors]
            for ao in asm:
                # 1) Loop-invariant code motion
                if licm:
                    ao.generalized_licm(licm)
                    decls.update(ao.decls)

                # 2) Splitting
                if split:
                    ao.split(split[0], split[1])

                # 3) Permute integration loop
                if permute:
                    ao.permute_int_loop()

                # 3) Unroll/Unroll-and-jam
                if unroll:
                    ao.unroll({0: unroll[0], 1: unroll[1], 2: unroll[2]})

                # 4) Register tiling
                if slice_factor and v_type == AUTOVECT:
                    ao.slice_loop(slice_factor)

                # 5) Vectorization
                if initialized:
                    vect = AssemblyVectorizer(ao, intrinsics, compiler)
                    if ap:
                        vect.alignment(decls)
                        if not blas:
                            vect.padding(decls)
                            self.ap = True
                    if v_type and v_type != AUTOVECT:
                        if intrinsics['inst_set'] == 'SSE':
                            raise RuntimeError("COFFEE Error: SSE vectorization not supported")
                        vect.outer_product(v_type, v_param)

                # 6) Conversion into blas calls
                if blas:
                    ala = AssemblyLinearAlgebra(ao, decls)
                    self.blas = ala.transform(blas)

            # Ensure kernel is always marked static inline
            if hasattr(self, 'fundecl'):
                # Remove either or both of static and inline (so that we get the order right)
                self.fundecl.pred = [q for q in self.fundecl.pred
                                     if q not in ['static', 'inline']]
                self.fundecl.pred.insert(0, 'inline')
                self.fundecl.pred.insert(0, 'static')

            return asm

        def _heuristic_unroll_factors(sizes, ths):
            """Return a list of unroll factors to try given the sizes in ``sizes``.
            The return value is a list of tuples, where each element in a tuple
            represents the unroll factor for the corresponding loop in the nest.

            :arg ths: unrolling threshold
            """
            i_loop, j_loop, k_loop = sizes
            # Determine individual unroll factors
            i_factors = [i+1 for i in range(i_loop) if i_loop % (i+1) == 0] or [0]
            j_factors = [i+1 for i in range(j_loop) if j_loop % (i+1) == 0] or [0]
            k_factors = [1]
            # Return the cartesian product of all possible unroll factors not exceeding the threshold
            unroll_factors = list(itertools.product(i_factors, j_factors, k_factors))
            return [x for x in unroll_factors if reduce(operator.mul, x) <= ths]

        if opts.get('autotune'):
            if not (compiler and intrinsics):
                raise RuntimeError("COFFEE Error: must properly initialize COFFEE for autotuning")
            # Set granularity of autotuning
            resolution = autotune_resolution
            unroll_ths = autotune_unroll_ths
            autotune_configs = autotune_all
            if opts['autotune'] == 'minimal':
                resolution = 1
                autotune_configs = autotune_minimal
                unroll_ths = 4
            elif blas_interface:
                autotune_configs.append(('blas', 3, 0, (None, None), True, (1, 0),
                                         blas_interface['name'], None, False))
            variants = []
            autotune_configs_unroll = []
            tunable = True
            original_ast = dcopy(self.ast)
            # Generate basic kernel variants
            for params in autotune_configs:
                opt, _params = params[0], params[1:]
                asm = _generate_cpu_code(self, *_params)
                if not asm:
                    # Not a local assembly kernel, nothing to tune
                    tunable = False
                    break
                if opt in ['licm', 'split']:
                    # Heuristically apply a set of unroll factors on top of the transformation
                    ao = asm[0]
                    int_loop_sz = ao.int_loop.size() if ao.int_loop else 0
                    asm_outer_sz = ao.asm_itspace[0][0].size() if len(ao.asm_itspace) >= 1 else 0
                    asm_inner_sz = ao.asm_itspace[1][0].size() if len(ao.asm_itspace) >= 2 else 0
                    loop_sizes = [int_loop_sz, asm_outer_sz, asm_inner_sz]
                    for factor in _heuristic_unroll_factors(loop_sizes, unroll_ths):
                        autotune_configs_unroll.append(params[:7] + (factor,) + params[8:])
                # Add the variant to the test cases the autotuner will have to run
                variants.append(self.ast)
                self.ast = dcopy(original_ast)
            # On top of some of the basic kernel variants, apply unroll/unroll-and-jam
            for params in autotune_configs_unroll:
                asm = _generate_cpu_code(self, *params[1:])
                variants.append(self.ast)
                self.ast = dcopy(original_ast)
            if tunable:
                # Determine the fastest kernel implementation
                autotuner = Autotuner(variants, asm[0].asm_itspace, self.include_dirs,
                                      compiler, intrinsics, blas_interface)
                fastest = autotuner.tune(resolution)
                variants = autotune_configs + autotune_configs_unroll
                name, params = variants[fastest][0], variants[fastest][1:]
                # Discard values set while autotuning
                if name != 'blas':
                    self.blas = False
            else:
                # The kernel is not transformed since it was not a local assembly kernel
                params = (0, False, (None, None), True, None, False, None, False)
        elif opts.get('blas'):
            # Conversion into blas requires a specific set of transformations
            # in order to identify and extract matrix multiplies.
            if not blas_interface:
                raise RuntimeError("COFFEE Error: must set PYOP2_BLAS to convert into BLAS calls")
            params = (3, 0, (None, None), True, (1, 0), opts['blas'], None, False)
        else:
            # Fetch user-provided options/hints on how to transform the kernel
            params = (opts.get('licm'), opts.get('slice'), opts.get('vect') or (None, None),
                      opts.get('ap'), opts.get('split'), opts.get('blas'), opts.get('unroll'),
                      opts.get('permute'))

        # Generate a specific code version
        _generate_cpu_code(self, *params)

    def gencode(self):
        """Generate a string representation of the AST."""
        return self.ast.gencode()

# These global variables capture the internal state of COFFEE
intrinsics = {}
compiler = {}
blas_interface = {}
initialized = False


def init_coffee(isa, comp, blas):
    """Initialize COFFEE."""

    global intrinsics, compiler, blas_interface, initialized
    intrinsics = _init_isa(isa)
    compiler = _init_compiler(comp)
    blas_interface = _init_blas(blas)
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

    return {}


def _init_compiler(compiler):
    """Set compiler-specific keywords. """

    if compiler == 'intel':
        return {
            'name': 'intel',
            'align': lambda o: '__attribute__((aligned(%s)))' % o,
            'decl_aligned_for': '#pragma vector aligned',
            'force_simdization': '#pragma simd',
            'AVX': '-xAVX',
            'SSE': '-xSSE',
            'ipo': '-ip',
            'vect_header': '#include <immintrin.h>'
        }

    if compiler == 'gnu':
        return {
            'name': 'gnu',
            'align': lambda o: '__attribute__((aligned(%s)))' % o,
            'decl_aligned_for': '#pragma vector aligned',
            'AVX': '-mavx',
            'SSE': '-msse',
            'ipo': '',
            'vect_header': '#include <immintrin.h>'
        }

    return {}


def _init_blas(blas):
    """Initialize a dictionary of blas-specific keywords for code generation."""

    import os

    blas_dict = {
        'dir': os.environ.get("PYOP2_BLAS_DIR", ""),
        'namespace': ''
    }

    if blas == 'mkl':
        blas_dict.update({
            'name': 'mkl',
            'header': '#include <mkl.h>',
            'link': ['-lmkl_rt']
        })
    elif blas == 'atlas':
        blas_dict.update({
            'name': 'atlas',
            'header': '#include "cblas.h"',
            'link': ['-lsatlas']
        })
    elif blas == 'eigen':
        blas_dict.update({
            'name': 'eigen',
            'header': '#include <Eigen/Dense>',
            'namespace': 'using namespace Eigen;',
            'link': []
        })
    else:
        return {}
    return blas_dict
