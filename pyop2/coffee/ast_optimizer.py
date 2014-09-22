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

try:
    from collections import OrderedDict
# OrderedDict was added in Python 2.7. Earlier versions can use ordereddict
# from PyPI
except ImportError:
    from ordereddict import OrderedDict
from collections import defaultdict
import itertools
from copy import deepcopy as dcopy

import networkx as nx

from ast_base import *
from ast_utils import ast_update_ofs, itspace_size_ofs, itspace_merge
import ast_plan


class AssemblyOptimizer(object):

    """Assembly optimiser interface class"""

    def __init__(self, loop_nest, pre_header, kernel_decls, is_mixed):
        """Initialize the AssemblyOptimizer.

        :arg loop_nest:    root node of the local assembly code.
        :arg pre_header:   parent of the root node
        :arg kernel_decls: list of declarations of variables which are visible
                           within the local assembly code block.
        :arg is_mixed:     true if the assembly operation uses mixed (vector)
                           function spaces."""
        self.pre_header = pre_header
        self.kernel_decls = kernel_decls
        # Properties of the assembly operation
        self._is_mixed = is_mixed
        # Track applied optimizations
        self._is_precomputed = False
        self._has_zeros = False
        # Expressions evaluating the element matrix
        self.asm_expr = {}
        # Track nonzero regions accessed in the various loops
        self.nz_in_fors = {}
        # Integration loop (if any)
        self.int_loop = None
        # Fully parallel iteration space in the assembly loop nest
        self.asm_itspace = []
        # Expression graph tracking data dependencies
        self.expr_graph = ExpressionGraph()
        # Dictionary contaning various information about hoisted expressions
        self.hoisted = OrderedDict()
        # Inspect the assembly loop nest and collect info
        self.fors, self.decls, self.sym = self._visit_nest(loop_nest)
        self.fors = zip(*self.fors)[0]

    def _visit_nest(self, node):
        """Explore the loop nest and collect various info like:

        * Loops
        * Declarations and Symbols
        * Optimisations requested by the higher layers via pragmas"""

        def check_opts(node, parent, fors):
            """Check if node is associated some pragma. If that is the case,
            it saves this info so as to enable pyop2 optimising such node. """
            if node.pragma:
                opts = node.pragma[0].split(" ", 2)
                if len(opts) < 3:
                    return
                if opts[1] == "pyop2":
                    if opts[2] == "integration":
                        # Found integration loop
                        self.int_loop = node
                        return
                    if opts[2] == "itspace":
                        # Found high-level optimisation
                        self.asm_itspace.append((node, parent))
                        return
                    delim = opts[2].find('(')
                    opt_name = opts[2][:delim].replace(" ", "")
                    opt_par = opts[2][delim:].replace(" ", "")
                    if opt_name == "assembly":
                        # Found high-level optimisation
                        # Store outer product iteration variables, parent, loops
                        it_vars = [opt_par[1], opt_par[3]]
                        fors, fors_parents = zip(*fors)
                        loops = [l for l in fors if l.it_var() in it_vars]
                        self.asm_expr[node] = (it_vars, parent, loops)
                    else:
                        raise RuntimeError("Unrecognised opt %s - skipping it", opt_name)
                else:
                    raise RuntimeError("Unrecognised pragma found '%s'", node.pragma[0])

        def inspect(node, parent, fors, decls, symbols):
            if isinstance(node, Block):
                self.block = node
                for n in node.children:
                    inspect(n, node, fors, decls, symbols)
                return (fors, decls, symbols)
            elif isinstance(node, For):
                check_opts(node, parent, fors)
                fors.append((node, parent))
                return inspect(node.children[0], node, fors, decls, symbols)
            elif isinstance(node, Par):
                return inspect(node.children[0], node, fors, decls, symbols)
            elif isinstance(node, Decl):
                decls[node.sym.symbol] = (node, ast_plan.LOCAL_VAR)
                return (fors, decls, symbols)
            elif isinstance(node, Symbol):
                symbols.add(node)
                return (fors, decls, symbols)
            elif isinstance(node, Expr):
                for child in node.children:
                    inspect(child, node, fors, decls, symbols)
                return (fors, decls, symbols)
            elif isinstance(node, Perfect):
                check_opts(node, parent, fors)
                for child in node.children:
                    inspect(child, node, fors, decls, symbols)
                return (fors, decls, symbols)
            else:
                return (fors, decls, symbols)

        return inspect(node, self.pre_header, [], {}, set())

    def _get_root(self):
        """Return the root node of the assembly loop nest. It can be either the
        loop over quadrature points or, if absent, a generic point in the
        assembly routine."""
        return self.int_loop.children[0] if self.int_loop else self.pre_header

    def extract_itspace(self):
        """Remove fully-parallel loop from the iteration space. These are
        the loops that were marked by the user/higher layer with a ``pragma
        pyop2 itspace``."""

        itspace_vrs = []
        for node, parent in reversed(self.asm_itspace):
            parent.children.extend(node.children[0].children)
            parent.children.remove(node)
            itspace_vrs.append(node.it_var())

        any_in = lambda a, b: any(i in b for i in a)
        accessed_vrs = [s for s in self.sym if any_in(s.rank, itspace_vrs)]

        return (itspace_vrs, accessed_vrs)

    def rewrite(self, level):
        """Rewrite an assembly expression to minimize floating point operations
        and relieve register pressure. This involves several possible transformations:

        1. Generalized loop-invariant code motion
        2. Factorization of common loop-dependent terms
        3. Expansion of constants over loop-dependent terms
        4. Zero-valued columns avoidance
        5. Precomputation of integration-dependent terms

        :arg level: The optimization level (0, 1, 2, 3, 4). The higher, the more
                    invasive is the re-writing of the assembly expressions,
                    trying to eliminate unnecessary floating point operations.

                    * level == 1: performs "basic" generalized loop-invariant \
                                  code motion
                    * level == 2: level 1 + expansion of terms, factorization of \
                                  basis functions appearing multiple times in the \
                                  same expression, and finally another run of \
                                  loop-invariant code motion to move invariant \
                                  sub-expressions exposed by factorization
                    * level == 3: level 2 + avoid computing zero-columns
                    * level == 4: level 3 + precomputation of read-only expressions \
                                  out of the assembly loop nest
        """

        if not self.asm_expr:
            return

        parent = (self.pre_header, self.kernel_decls)
        for expr in self.asm_expr.items():
            ew = AssemblyRewriter(expr, self.int_loop, self.sym, self.decls,
                                  parent, self.hoisted, self.expr_graph)
            # Perform expression rewriting
            if level > 0:
                ew.licm()
            if level > 1:
                ew.expand()
                ew.distribute()
                ew.licm()
                # Fuse loops iterating along the same iteration space
                lm = PerfectSSALoopMerger(self.expr_graph, self._get_root())
                lm.merge()
                ew.simplify()
            # Precompute expressions
            if level == 4:
                self._precompute(expr)
                self._is_precomputed = True

        # Eliminate zero-valued columns if the kernel operation uses mixed (vector)
        # function spaces, leading to zero-valued columns in basis function arrays
        if level == 3 and self._is_mixed:
            # Split the assembly expression into separate loop nests,
            # based on sum's associativity. This exposes more opportunities
            # for restructuring loops, since different summands may have
            # contiguous regions of zero-valued columns in different
            # positions. The ZeroLoopScheduler, indeed, analyzes statements
            # "one by one", and changes the iteration spaces of the enclosing
            # loops accordingly.
            elf = ExprLoopFissioner(self.expr_graph, self._get_root(), 1)
            new_asm_expr = {}
            for expr in self.asm_expr.items():
                new_asm_expr.update(elf.expr_fission(expr, False))
            # Search for zero-valued columns and restructure the iteration spaces
            zls = ZeroLoopScheduler(self.expr_graph, self._get_root(),
                                    (self.kernel_decls, self.decls))
            self.asm_expr = zls.reschedule()[-1]
            self.nz_in_fors = zls.nz_in_fors
            self._has_zeros = True

    def slice(self, slice_factor=None):
        """Perform slicing of the innermost loop to enhance register reuse.
        For example, given a loop:

        .. code-block:: none

            for i = 0 to N
              f()

        the following sequence of loops is generated:

        .. code-block:: none

            for i = 0 to k
              f()
            for i = k to 2k
              f()
            # ...
            for i = (N-1)k to N
              f()

        The goal is to improve register re-use by relying on the backend
        compiler unrolling and vector-promoting the sliced loops."""

        if slice_factor == -1:
            slice_factor = 20  # Defaut value

        for stmt, stmt_info in self.asm_expr.items():
            # First, find outer product loops in the nest
            it_vars, parent, loops = stmt_info

            # Build sliced loops
            sliced_loops = []
            n_loops = loops[1].cond.children[1].symbol / slice_factor
            rem_loop_sz = loops[1].cond.children[1].symbol
            init = 0
            for i in range(n_loops):
                loop = dcopy(loops[1])
                loop.init.init = Symbol(init, ())
                loop.cond.children[1] = Symbol(slice_factor * (i + 1), ())
                init += slice_factor
                sliced_loops.append(loop)

            # Build remainder loop
            if rem_loop_sz > 0:
                init = slice_factor * n_loops
                loop = dcopy(loops[1])
                loop.init.init = Symbol(init, ())
                loop.cond.children[1] = Symbol(rem_loop_sz, ())
                sliced_loops.append(loop)

            # Append sliced loops at the right point in the nest
            par_block = loops[0].children[0]
            pb = par_block.children
            idx = pb.index(loops[1])
            par_block.children = pb[:idx] + sliced_loops + pb[idx + 1:]

    def unroll(self, loops_factor):
        """Unroll loops in the assembly nest.

        :arg loops_factor: dictionary from loops to unroll (factor, increment).
            Loops are specified as integers:

            * 0 = integration loop,
            * 1 = test functions loop,
            * 2 = trial functions loop.

            A factor of 0 denotes that the corresponding loop is not present.
        """

        def update_stmt(node, var, factor):
            """Add an offset ``factor`` to every iteration variable ``var`` in
            ``node``."""
            if isinstance(node, Symbol):
                new_ofs = []
                node.offset = node.offset or ((1, 0) for i in range(len(node.rank)))
                for r, ofs in zip(node.rank, node.offset):
                    new_ofs.append((ofs[0], ofs[1] + factor) if r == var else ofs)
                node.offset = tuple(new_ofs)
            else:
                for n in node.children:
                    update_stmt(n, var, factor)

        def unroll_loop(asm_expr, it_var, factor):
            """Unroll assembly expressions in ``asm_expr`` along iteration variable
            ``it_var`` a total of ``factor`` times."""
            new_asm_expr = {}
            unroll_loops = set()
            for stmt, stmt_info in asm_expr.items():
                it_vars, parent, loops = stmt_info
                new_stmts = []
                # Determine the loop along which to unroll
                if self.int_loop and self.int_loop.it_var() == it_var:
                    loop = self.int_loop
                elif loops[0].it_var() == it_var:
                    loop = loops[0]
                else:
                    loop = loops[1]
                unroll_loops.add(loop)
                # Unroll individual statements
                for i in range(factor):
                    new_stmt = dcopy(stmt)
                    update_stmt(new_stmt, loop.it_var(), (i+1))
                    parent.children.append(new_stmt)
                    new_stmts.append(new_stmt)
                new_asm_expr.update(dict(zip(new_stmts,
                                             [stmt_info for i in range(len(new_stmts))])))
            # Update the increment of each unrolled loop
            for l in unroll_loops:
                l.incr.children[1].symbol += factor
            return new_asm_expr

        int_factor = loops_factor[0]
        asm_outer_factor = loops_factor[1]
        asm_inner_factor = loops_factor[2]

        # Unroll-and-jam integration loop
        if int_factor > 1 and self._is_precomputed:
            self.asm_expr.update(unroll_loop(self.asm_expr, self.int_loop.it_var(),
                                             int_factor-1))
        # Unroll-and-jam test functions loop
        if asm_outer_factor > 1:
            self.asm_expr.update(unroll_loop(self.asm_expr, self.asm_itspace[0][0].it_var(),
                                             asm_outer_factor-1))
        # Unroll trial functions loop
        if asm_inner_factor > 1:
            self.asm_expr.update(unroll_loop(self.asm_expr, self.asm_itspace[1][0].it_var(),
                                             asm_inner_factor-1))

    def permute(self):
        """Permute the integration loop with the innermost loop in the assembly nest.
        This transformation is legal if ``_precompute`` was invoked. Storage layout of
        all 2-dimensional arrays involved in the element matrix computation is
        transposed."""

        def transpose_layout(node, transposed, to_transpose):
            """Transpose the storage layout of symbols in ``node``. If the symbol is
            in a declaration, then its statically-known size is transposed (e.g.
            double A[3][4] -> double A[4][3]). Otherwise, its iteration variables
            are swapped (e.g. A[i][j] -> A[j][i]).

            If ``to_transpose`` is empty, then all symbols encountered in the traversal of
            ``node`` are transposed. Otherwise, only symbols in ``to_transpose`` are
            transposed."""
            if isinstance(node, Symbol):
                if not to_transpose:
                    transposed.add(node.symbol)
                elif node.symbol in to_transpose:
                    node.rank = (node.rank[1], node.rank[0])
            elif isinstance(node, Decl):
                transpose_layout(node.sym, transposed, to_transpose)
            elif isinstance(node, FlatBlock):
                return
            else:
                for n in node.children:
                    transpose_layout(n, transposed, to_transpose)

        if not self.int_loop or not self._is_precomputed:
            return

        new_asm_expr = {}
        new_outer_loop = None
        new_inner_loops = []
        permuted = set()
        transposed = set()
        for stmt, stmt_info in self.asm_expr.items():
            it_vars, parent, loops = stmt_info
            inner_loop = loops[-1]
            # Permute loops
            if inner_loop in permuted:
                continue
            else:
                permuted.add(inner_loop)
            new_outer_loop = new_outer_loop or dcopy(inner_loop)
            inner_loop.init = dcopy(self.int_loop.init)
            inner_loop.cond = dcopy(self.int_loop.cond)
            inner_loop.incr = dcopy(self.int_loop.incr)
            inner_loop.pragma = dcopy(self.int_loop.pragma)
            new_asm_loops = (new_outer_loop,) if len(loops) == 1 else (new_outer_loop, loops[0])
            new_asm_expr[stmt] = (it_vars, parent, new_asm_loops)
            new_inner_loops.append(new_asm_loops[-1])
            new_outer_loop.children[0].children = new_inner_loops
            # Track symbols whose storage layout should be transposed for unit-stridness
            transpose_layout(stmt.children[1], transposed, set())
        blk = self.pre_header.children
        blk.insert(blk.index(self.int_loop), new_outer_loop)
        blk.remove(self.int_loop)
        # Update assembly expressions and integration loop
        self.asm_expr = new_asm_expr
        self.int_loop = inner_loop
        # Transpose storage layout of all symbols involved in assembly
        transpose_layout(self.pre_header, set(), transposed)

    def split(self, cut=1):
        """Split assembly expressions into multiple chunks exploiting sum's
        associativity. Each chunk will have ``cut`` summands.

        For example, consider the following piece of code: ::

            for i
              for j
                A[i][j] += X[i]*Y[j] + Z[i]*K[j] + B[i]*X[j]

        If ``cut=1`` the expression is cut into chunks of length 1: ::

            for i
              for j
                A[i][j] += X[i]*Y[j]
            for i
              for j
                A[i][j] += Z[i]*K[j]
            for i
              for j
                A[i][j] += B[i]*X[j]

        If ``cut=2`` the expression is cut into chunks of length 2, plus a
        remainder chunk of size 1: ::

            for i
              for j
                A[i][j] += X[i]*Y[j] + Z[i]*K[j]
            # Remainder:
            for i
              for j
                A[i][j] += B[i]*X[j]
        """

        if not self.asm_expr:
            return

        new_asm_expr = {}
        elf = ExprLoopFissioner(self.expr_graph, self._get_root(), cut)
        for splittable in self.asm_expr.items():
            # Split the expression
            new_asm_expr.update(elf.expr_fission(splittable, True))
        self.asm_expr = new_asm_expr

    def _precompute(self, expr):
        """Precompute all expressions contributing to the evaluation of the local
        assembly tensor. Precomputation implies vector expansion and hoisting
        outside of the loop nest. This renders the assembly loop nest perfect.

        For example:
        for i
          for r
            A[r] += f(i, ...)
          for j
            for k
              LT[j][k] += g(A[r], ...)

        becomes
        for i
          for r
            A[i][r] += f(...)
        for i
          for j
            for k
              LT[j][k] += g(A[i][r], ...)
        """

        def update_syms(node, precomputed):
            if isinstance(node, Symbol):
                if node.symbol in precomputed:
                    node.rank = precomputed[node.symbol] + node.rank
            else:
                for n in node.children:
                    update_syms(n, precomputed)

        def precompute_stmt(node, precomputed, new_outer_block):
            """Recursively precompute, and vector-expand if already precomputed,
            all terms rooted in node."""

            if isinstance(node, Symbol):
                # Vector-expand the symbol if already pre-computed
                if node.symbol in precomputed:
                    node.rank = precomputed[node.symbol] + node.rank
            elif isinstance(node, Expr):
                for n in node.children:
                    precompute_stmt(n, precomputed, new_outer_block)
            elif isinstance(node, (Assign, Incr)):
                # Precompute the LHS of the assignment
                symbol = node.children[0]
                precomputed[symbol.symbol] = (self.int_loop.it_var(),)
                new_rank = (self.int_loop.it_var(),) + symbol.rank
                symbol.rank = new_rank
                # Vector-expand the RHS
                precompute_stmt(node.children[1], precomputed, new_outer_block)
                # Finally, append the new node
                new_outer_block.append(node)
            elif isinstance(node, Decl):
                new_outer_block.append(node)
                if isinstance(node.init, Symbol):
                    node.init.symbol = "{%s}" % node.init.symbol
                elif isinstance(node.init, Expr):
                    new_assign = Assign(dcopy(node.sym), node.init)
                    precompute_stmt(new_assign, precomputed, new_outer_block)
                    node.init = EmptyStatement()
                # Vector-expand the declaration of the precomputed symbol
                node.sym.rank = (self.int_loop.size(),) + node.sym.rank
            elif isinstance(node, For):
                # Precompute and/or Vector-expand inner statements
                new_children = []
                for n in node.children[0].children:
                    precompute_stmt(n, precomputed, new_children)
                node.children[0].children = new_children
                new_outer_block.append(node)
            else:
                raise RuntimeError("Precompute error: found unexpteced node: %s" % str(node))

        # The integration loop must be present for precomputation to be meaningful
        if not self.int_loop:
            return

        expr, expr_info = expr
        asm_outer_loop = expr_info[2][0]

        # Precomputation
        precomputed_block = []
        precomputed_syms = {}
        for i in self.int_loop.children[0].children:
            if i == asm_outer_loop:
                break
            elif isinstance(i, FlatBlock):
                continue
            else:
                precompute_stmt(i, precomputed_syms, precomputed_block)

        # Wrap hoisted for/assignments/increments within a loop
        new_outer_block = []
        searching_stmt = []
        for i in precomputed_block:
            if searching_stmt and not isinstance(i, (Assign, Incr)):
                wrap = Block(searching_stmt, open_scope=True)
                precompute_for = For(dcopy(self.int_loop.init), dcopy(self.int_loop.cond),
                                     dcopy(self.int_loop.incr), wrap, dcopy(self.int_loop.pragma))
                new_outer_block.append(precompute_for)
                searching_stmt = []
            if isinstance(i, For):
                wrap = Block([i], open_scope=True)
                precompute_for = For(dcopy(self.int_loop.init), dcopy(self.int_loop.cond),
                                     dcopy(self.int_loop.incr), wrap, dcopy(self.int_loop.pragma))
                new_outer_block.append(precompute_for)
            elif isinstance(i, (Assign, Incr)):
                searching_stmt.append(i)
            else:
                new_outer_block.append(i)

        # Delete precomputed stmts from original loop nest
        self.int_loop.children[0].children = [asm_outer_loop]

        # Update the AST adding the newly precomputed blocks
        root = self.pre_header.children
        ofs = root.index(self.int_loop)
        self.pre_header.children = root[:ofs] + new_outer_block + root[ofs:]

        # Update the AST by vector-expanding the pre-computed accessed variables
        update_syms(expr.children[1], precomputed_syms)


class AssemblyRewriter(object):
    """Provide operations to re-write an assembly expression:

    * Loop-invariant code motion: find and hoist sub-expressions which are
      invariant with respect to an assembly loop
    * Expansion: transform an expression ``(a + b)*c`` into ``(a*c + b*c)``
    * Distribute: transform an expression ``a*b + a*c`` into ``a*(b+c)``"""

    def __init__(self, expr, int_loop, syms, decls, parent, hoisted, expr_graph):
        """Initialize the AssemblyRewriter.

        :arg expr:       provide generic information related to an assembly
                         expression, including the depending for loops.
        :arg int_loop:   the loop along which integration is performed.
        :arg syms:       list of AST symbols used to evaluate the local element
                         matrix.
        :arg decls:      list of AST declarations of the various symbols in ``syms``.
        :arg parent:     the parent AST node of the assembly loop nest.
        :arg hoisted:    dictionary that tracks hoisted expressions
        :arg expr_graph: expression graph that tracks symbol dependencies
        """
        self.expr, self.expr_info = expr
        self.int_loop = int_loop
        self.syms = syms
        self.decls = decls
        self.parent, self.parent_decls = parent
        self.hoisted = hoisted
        self.expr_graph = expr_graph
        # Properties of the assembly expression
        self._licm = 0
        self._expanded = False

    def licm(self):
        """Perform loop-invariant code motion.

        Invariant expressions found in the loop nest are moved "after" the
        outermost independent loop and "after" the fastest varying dimension
        loop. Here, "after" means that if the loop nest has two loops ``i``
        and ``j``, and ``j`` is in the body of ``i``, then ``i`` comes after
        ``j`` (i.e. the loop nest has to be read from right to left).

        For example, if a sub-expression ``E`` depends on ``[i, j]`` and the
        loop nest has three loops ``[i, j, k]``, then ``E`` is hoisted out from
        the body of ``k`` to the body of ``i``). All hoisted expressions are
        then wrapped within a suitable loop in order to exploit compiler
        autovectorization. Note that this applies to constant sub-expressions
        as well, in which case hoisting after the outermost loop takes place."""

        def extract(node, expr_dep, length=0):
            """Extract invariant sub-expressions from the original assembly
            expression. Hoistable sub-expressions are stored in expr_dep."""

            def hoist(node, dep, expr_dep, _extract=True):
                if _extract:
                    node = Par(node) if isinstance(node, Symbol) else node
                    expr_dep[dep].append(node)
                extract.has_extracted = extract.has_extracted or _extract

            if isinstance(node, Symbol):
                return (node.loop_dep, extract.INV, 1)
            if isinstance(node, Par):
                return (extract(node.children[0], expr_dep, length))

            # Traverse the expression tree
            left, right = node.children
            dep_l, info_l, len_l = extract(left, expr_dep, length)
            dep_r, info_r, len_r = extract(right, expr_dep, length)
            node_len = len_l + len_r

            if info_l == extract.KSE and info_r == extract.KSE:
                if dep_l != dep_r:
                    # E.g. (A[i]*alpha + D[i])*(B[j]*beta + C[j])
                    hoist(left, dep_l, expr_dep)
                    hoist(right, dep_r, expr_dep)
                    return ((), extract.HOI, node_len)
                else:
                    # E.g. (A[i]*alpha)+(B[i]*beta)
                    return (dep_l, extract.KSE, node_len)
            elif info_l == extract.KSE and info_r == extract.INV:
                hoist(left, dep_l, expr_dep)
                hoist(right, dep_r, expr_dep, (dep_r and len_r == 1) or len_r > 1)
                return ((), extract.HOI, node_len)
            elif info_l == extract.INV and info_r == extract.KSE:
                hoist(right, dep_r, expr_dep)
                hoist(left, dep_l, expr_dep, (dep_l and len_l == 1) or len_l > 1)
                return ((), extract.HOI, node_len)
            elif info_l == extract.INV and info_r == extract.INV:
                if not dep_l and not dep_r:
                    # E.g. alpha*beta
                    return ((), extract.INV, node_len)
                elif dep_l and dep_r and dep_l != dep_r:
                    # E.g. A[i]*B[j]
                    hoist(left, dep_l, expr_dep, not self._licm or len_l > 1)
                    hoist(right, dep_r, expr_dep, not self._licm or len_r > 1)
                    return ((), extract.HOI, node_len)
                elif dep_l and dep_r and dep_l == dep_r:
                    # E.g. A[i] + B[i]
                    return (dep_l, extract.INV, node_len)
                elif dep_l and not dep_r:
                    # E.g. A[i]*alpha
                    hoist(right, dep_r, expr_dep, len_r > 1)
                    return (dep_l, extract.KSE, node_len)
                elif dep_r and not dep_l:
                    # E.g. alpha*A[i]
                    hoist(left, dep_l, expr_dep, len_l > 1)
                    return (dep_r, extract.KSE, node_len)
                else:
                    raise RuntimeError("Error while hoisting invariant terms")
            elif info_l == extract.HOI and info_r == extract.KSE:
                hoist(right, dep_r, expr_dep, len_r > 2)
                return ((), extract.HOI, node_len)
            elif info_l == extract.KSE and info_r == extract.HOI:
                hoist(left, dep_l, expr_dep, len_l > 2)
                return ((), extract.HOI, node_len)
            elif info_l == extract.HOI or info_r == extract.HOI:
                return ((), extract.HOI, node_len)
            else:
                raise RuntimeError("Fatal error while finding hoistable terms")

        extract.INV = 0  # Invariant term(s)
        extract.KSE = 1  # Keep searching invariant sub-expressions
        extract.HOI = 2  # Stop searching, done hoisting
        extract.has_extracted = False

        def replace(node, syms_dict, n_replaced):
            if isinstance(node, Symbol):
                if str(Par(node)) in syms_dict:
                    return True
                else:
                    return False
            if isinstance(node, Par):
                if str(node) in syms_dict:
                    return True
                else:
                    return replace(node.children[0], syms_dict, n_replaced)
            # Found invariant sub-expression
            if str(node) in syms_dict:
                return True

            # Traverse the expression tree and replace
            left = node.children[0]
            right = node.children[1]
            if replace(left, syms_dict, n_replaced):
                left = Par(left) if isinstance(left, Symbol) else left
                replacing = syms_dict[str(left)]
                node.children[0] = dcopy(replacing)
                n_replaced[str(replacing)] += 1
            if replace(right, syms_dict, n_replaced):
                right = Par(right) if isinstance(right, Symbol) else right
                replacing = syms_dict[str(right)]
                node.children[1] = dcopy(replacing)
                n_replaced[str(replacing)] += 1
            return False

        # Extract read-only sub-expressions that do not depend on at least
        # one loop in the loop nest
        inv_dep = {}
        typ = self.parent_decls[self.expr.children[0].symbol][0].typ
        while True:
            expr_dep = defaultdict(list)
            extract(self.expr.children[1], expr_dep)

            # While end condition
            if self._licm and not extract.has_extracted:
                break
            extract.has_extracted = False
            self._licm += 1

            for dep, expr in sorted(expr_dep.items()):
                # 0) Determine the loops that should wrap invariant statements
                # and where such loops should be placed in the loop nest
                place = self.int_loop.children[0] if self.int_loop else self.parent
                out_asm_loop, in_asm_loop = self.expr_info[2]
                ofs = lambda: place.children.index(out_asm_loop)
                if dep and out_asm_loop.it_var() == dep[-1]:
                    wl = out_asm_loop
                elif dep and in_asm_loop.it_var() == dep[-1]:
                    wl = in_asm_loop
                else:
                    wl = None

                # 1) Remove identical sub-expressions
                expr = dict([(str(e), e) for e in expr]).values()

                # 2) Create the new invariant sub-expressions and temporaries
                sym_rank, for_dep = (tuple([wl.size()]), tuple([wl.it_var()])) \
                    if wl else ((), ())
                syms = [Symbol("LI_%s_%d_%s" % ("".join(dep).upper() if dep else "C",
                        self._licm, i), sym_rank) for i in range(len(expr))]
                var_decl = [Decl(typ, _s) for _s in syms]
                for_sym = [Symbol(_s.sym.symbol, for_dep) for _s in var_decl]

                # 3) Create the new for loop containing invariant terms
                _expr = [Par(e) if not isinstance(e, Par) else e for e in expr]
                inv_for = [Assign(_s, e) for _s, e in zip(for_sym, _expr)]

                # 4) Update the lists of symbols accessed and of decls
                self.syms.update([d.sym for d in var_decl])
                lv = ast_plan.LOCAL_VAR
                self.decls.update(dict(zip([d.sym.symbol for d in var_decl],
                                           [(v, lv) for v in var_decl])))

                # 5) Replace invariant sub-trees with the proper tmp variable
                n_replaced = dict(zip([str(s) for s in for_sym], [0]*len(for_sym)))
                replace(self.expr.children[1], dict(zip([str(i) for i in expr], for_sym)),
                        n_replaced)

                # 6) Track hoisted symbols and symbols dependencies
                sym_info = [(i, j, inv_for) for i, j in zip(_expr, var_decl)]
                self.hoisted.update(zip([s.symbol for s in for_sym], sym_info))
                for s, e in zip(for_sym, expr):
                    self.expr_graph.add_dependency(s, e, n_replaced[str(s)] > 1)

                # 7a) Update expressions hoisted along a known dimension (same dep)
                if for_dep in inv_dep:
                    _var_decl, _inv_for = inv_dep[for_dep][0:2]
                    _var_decl.extend(var_decl)
                    _inv_for.extend(inv_for)
                    continue

                # 7b) Keep track of hoisted stuff
                inv_dep[for_dep] = (var_decl, inv_for, place, ofs, wl)

        for dep, dep_info in sorted(inv_dep.items()):
            var_decl, inv_for, place, ofs, wl = dep_info
            # Create the hoisted code
            if wl:
                new_for = [dcopy(wl)]
                new_for[0].children[0] = Block(inv_for, open_scope=True)
                inv_for = new_for
            # Append the new node at the right level in the loop nest
            new_block = var_decl + inv_for + [FlatBlock("\n")] + place.children[ofs():]
            place.children = place.children[:ofs()] + new_block
            # Update information about hoisted symbols
            for i in var_decl:
                old_sym_info = self.hoisted[i.sym.symbol]
                old_sym_info = old_sym_info[0:2] + (inv_for[0],) + (place.children,)
                self.hoisted[i.sym.symbol] = old_sym_info

    def count_occurrences(self, str_key=False):
        """For each variable in the assembly expression, count how many times
        it appears as involved in some operations. For example, for the
        expression ``a*(5+c) + b*(a+4)``, return ``{a: 2, b: 1, c: 1}``."""

        def count(node, counter):
            if isinstance(node, Symbol):
                node = str(node) if str_key else (node.symbol, node.rank)
                if node in counter:
                    counter[node] += 1
                else:
                    counter[node] = 1
            else:
                for c in node.children:
                    count(c, counter)

        counter = {}
        count(self.expr.children[1], counter)
        return counter

    def expand(self):
        """Expand assembly expressions such that: ::

            Y[j] = f(...)
            (X[i]*Y[j])*F + ...

        becomes: ::

            Y[j] = f(...)*F
            (X[i]*Y[j]) + ...

        This may be useful for several purposes:

        * Relieve register pressure; when, for example, ``(X[i]*Y[j])`` is
          computed in a loop L' different than the loop L'' in which ``Y[j]``
          is evaluated, and ``cost(L') > cost(L'')``
        * It is also a step towards exposing well-known linear algebra
          operations, like matrix-matrix multiplies."""

        # Select the assembly iteration variable along which the expansion should
        # be performed. The heuristics here is that the expansion occurs along the
        # iteration variable which appears in more unique arrays. This will allow
        # distribution to be more effective.
        asm_out, asm_in = (self.expr_info[0][0], self.expr_info[0][1])
        it_var_occs = {asm_out: 0, asm_in: 0}
        for s in self.count_occurrences().keys():
            if s[1] and s[1][0] in it_var_occs:
                it_var_occs[s[1][0]] += 1

        exp_var = asm_out if it_var_occs[asm_out] < it_var_occs[asm_in] else asm_in
        ee = ExpressionExpander(self.hoisted, self.expr_graph, self.parent)
        ee.expand(self.expr.children[1], self.expr, it_var_occs, exp_var)
        self.decls.update(ee.expanded_decls)
        self.syms.update(ee.expanded_syms)
        self._expanded = True

    def distribute(self):
        """Apply to the distributivity property to the assembly expression.
        E.g. ::

            A[i]*B[j] + A[i]*C[j]

        becomes ::

            A[i]*(B[j] + C[j])."""

        def find_prod(node, occs, to_distr):
            if isinstance(node, Par):
                find_prod(node.children[0], occs, to_distr)
            elif isinstance(node, Sum):
                find_prod(node.children[0], occs, to_distr)
                find_prod(node.children[1], occs, to_distr)
            elif isinstance(node, Prod):
                left, right = (node.children[0], node.children[1])
                l_str, r_str = (str(left), str(right))
                if occs[l_str] > 1 and occs[r_str] > 1:
                    if occs[l_str] > occs[r_str]:
                        dist = l_str
                        target = (left, right)
                        occs[r_str] -= 1
                    else:
                        dist = r_str
                        target = (right, left)
                        occs[l_str] -= 1
                elif occs[l_str] > 1 and occs[r_str] == 1:
                    dist = l_str
                    target = (left, right)
                elif occs[r_str] > 1 and occs[l_str] == 1:
                    dist = r_str
                    target = (right, left)
                elif occs[l_str] == 1 and occs[r_str] == 1:
                    dist = l_str
                    target = (left, right)
                else:
                    raise RuntimeError("Distribute error: symbol not found")
                to_distr[dist].append(target)

        def create_sum(symbols):
            if len(symbols) == 1:
                return symbols[0]
            else:
                return Sum(symbols[0], create_sum(symbols[1:]))

        # Expansion ensures the expression to be in a form like:
        # tensor[i][j] += A[i]*B[j] + C[i]*D[j] + A[i]*E[j] + ...
        if not self._expanded:
            raise RuntimeError("Distribute error: expansion required first.")

        to_distr = defaultdict(list)
        find_prod(self.expr.children[1], self.count_occurrences(True), to_distr)

        # Create the new assembly expression
        new_prods = []
        for d in to_distr.values():
            dist, target = zip(*d)
            target = Par(create_sum(target)) if len(target) > 1 else create_sum(target)
            new_prods.append(Par(Prod(dist[0], target)))
        self.expr.children[1] = Par(create_sum(new_prods))

    def simplify(self):
        """Scan the hoisted terms one by one and eliminate duplicate sub-expressions.
        Remove useless assignments (e.g. a = b, and b never used later)."""

        def replace_expr(node, parent, parent_idx, it_var, hoisted_expr):
            """Recursively search for any sub-expressions rooted in node that have
            been hoisted and therefore are already kept in a temporary. Replace them
            with such temporary."""
            if isinstance(node, Symbol):
                return
            else:
                tmp_sym = hoisted_expr.get(str(node)) or hoisted_expr.get(str(parent))
                if tmp_sym:
                    # Found a temporary value already hosting the value of node
                    parent.children[parent_idx] = Symbol(dcopy(tmp_sym), (it_var,))
                else:
                    # Go ahead recursively
                    for i, n in enumerate(node.children):
                        replace_expr(n, node, i, it_var, hoisted_expr)

        # Remove duplicates
        hoisted_expr = {}
        for sym, sym_info in self.hoisted.items():
            expr, var_decl, inv_for, place = sym_info
            if not isinstance(inv_for, For):
                continue
            # Check if any sub-expressions rooted in expr is alredy stored in a temporary
            replace_expr(expr.children[0], expr, 0, inv_for.it_var(), hoisted_expr)
            # Track the (potentially modified) hoisted expression
            hoisted_expr[str(expr)] = sym


class ExpressionExpander(object):
    """Expand assembly expressions such that: ::

        Y[j] = f(...)
        (X[i]*Y[j])*F + ...

    becomes: ::

        Y[j] = f(...)*F
        (X[i]*Y[j]) + ..."""

    CONST = -1
    ITVAR = -2

    def __init__(self, var_info, expr_graph, expr):
        self.var_info = var_info
        self.expr_graph = expr_graph
        self.parent = expr
        self.expanded_decls = {}
        self.found_consts = {}
        self.expanded_syms = []

    def _do_expand(self, sym, const):
        """Perform the actual expansion. If there are no dependencies, then
        the already hoisted expression is expanded. Otherwise, if the symbol to
        be expanded occurs multiple times in the expression, or it depends on
        other hoisted symbols that will also be expanded, create a new symbol."""

        old_expr, var_decl, inv_for, place = self.var_info[sym.symbol]

        # The expanding expression is first assigned to a temporary value in order
        # to minimize code size and, possibly, work around compiler's inefficiencies
        # when doing loop-invariant code motion
        const_str = str(const)
        if const_str in self.found_consts:
            const = dcopy(self.found_consts[const_str])
        elif not isinstance(const, Symbol):
            const_sym = Symbol("const%d" % len(self.found_consts), ())
            new_const_decl = Decl("double", dcopy(const_sym), const)
            # Keep track of the expansion
            self.expanded_decls[new_const_decl.sym.symbol] = (new_const_decl, ast_plan.LOCAL_VAR)
            self.expanded_syms.append(new_const_decl.sym)
            self.found_consts[const_str] = const_sym
            self.expr_graph.add_dependency(const_sym, const, False)
            # Update the AST
            place.insert(place.index(inv_for), new_const_decl)
            const = const_sym

        # No dependencies, just perform the expansion
        if not self.expr_graph.has_dep(sym):
            old_expr.children[0] = Prod(Par(old_expr.children[0]), dcopy(const))
            self.expr_graph.add_dependency(sym, const, False)
            return

        # Create a new symbol, expression, and declaration
        new_expr = Par(Prod(dcopy(sym), const))
        sym.symbol += "_EXP%d" % len(self.expanded_syms)
        new_node = Assign(dcopy(sym), new_expr)
        new_var_decl = dcopy(var_decl)
        new_var_decl.sym.symbol = sym.symbol
        # Append new expression and declaration
        inv_for.children[0].children.append(new_node)
        place.insert(place.index(var_decl), new_var_decl)
        self.expanded_decls[new_var_decl.sym.symbol] = (new_var_decl, ast_plan.LOCAL_VAR)
        self.expanded_syms.append(new_var_decl.sym)
        # Update tracked information
        self.var_info[sym.symbol] = (new_expr, new_var_decl, inv_for, place)
        self.expr_graph.add_dependency(sym, new_expr, 0)

    def expand(self, node, parent, it_vars, exp_var):
        """Perform the expansion of the expression rooted in ``node``. Terms are
        expanded along the iteration variable ``exp_var``."""

        if isinstance(node, Symbol):
            if not node.rank:
                return ([node], self.CONST)
            elif node.rank[-1] not in it_vars.keys():
                return ([node], self.CONST)
            else:
                return ([node], self.ITVAR)
        elif isinstance(node, Par):
            return self.expand(node.children[0], node, it_vars, exp_var)
        elif isinstance(node, Prod):
            l_node, l_type = self.expand(node.children[0], node, it_vars, exp_var)
            r_node, r_type = self.expand(node.children[1], node, it_vars, exp_var)
            if l_type == self.ITVAR and r_type == self.ITVAR:
                # Found an expandable product
                to_exp = l_node if l_node[0].rank[-1] == exp_var else r_node
                return (to_exp, self.ITVAR)
            elif l_type == self.CONST and r_type == self.CONST:
                # Product of constants; they are both used for expansion (if any)
                return ([node], self.CONST)
            else:
                # Do the expansion
                const = l_node[0] if l_type == self.CONST else r_node[0]
                expandable, exp_node = (l_node, node.children[0]) \
                    if l_type == self.ITVAR else (r_node, node.children[1])
                for sym in expandable:
                    # Perform the expansion
                    if sym.symbol not in self.var_info:
                        raise RuntimeError("Expansion error: no symbol: %s" % sym.symbol)
                    old_expr, var_decl, inv_for, place = self.var_info[sym.symbol]
                    self._do_expand(sym, const)
                # Update the parent node, since an expression has been expanded
                if parent.children[0] == node:
                    parent.children[0] = exp_node
                elif parent.children[1] == node:
                    parent.children[1] = exp_node
                else:
                    raise RuntimeError("Expansion error: wrong parent-child association")
                return (expandable, self.ITVAR)
        elif isinstance(node, Sum):
            l_node, l_type = self.expand(node.children[0], node, it_vars, exp_var)
            r_node, r_type = self.expand(node.children[1], node, it_vars, exp_var)
            if l_type == self.ITVAR and r_type == self.ITVAR:
                return (l_node + r_node, self.ITVAR)
            elif l_type == self.CONST and r_type == self.CONST:
                return ([node], self.CONST)
            else:
                return (None, self.CONST)
        else:
            raise RuntimeError("Expansion error: unknown node: %s" % str(node))


class LoopScheduler(object):

    """Base class for classes that handle loop scheduling; that is, loop fusion,
    loop distribution, etc."""

    def __init__(self, expr_graph, root):
        """Initialize the LoopScheduler.

        :arg expr_graph: the ExpressionGraph tracking all data dependencies involving
                         identifiers that appear in ``root``.
        :arg root:       the node where loop scheduling takes place."""
        self.expr_graph = expr_graph
        self.root = root


class PerfectSSALoopMerger(LoopScheduler):

    """Analyze data dependencies and iteration spaces, then merge fusable
    loops.
    Statements must be in "soft" SSA form: they can be declared and initialized
    at declaration time, then they can be assigned a value in only one place."""

    def __init__(self, expr_graph, root):
        super(PerfectSSALoopMerger, self).__init__(expr_graph, root)

    def _find_it_space(self, node):
        """Return the iteration space of the loop nest rooted in ``node``,
        as a tuple of 3-tuple, in which each 3-tuple is of the form
        (start, bound, increment)."""
        if isinstance(node, For):
            itspace = (node.start(), node.end(), node.increment())
            child_itspace = self._find_it_space(node.children[0].children[0])
            return (itspace, child_itspace) if child_itspace else (itspace,)

    def _accessed_syms(self, node, mode):
        """Return a list of symbols that are being accessed in the tree
        rooted in ``node``. If ``mode == 0``, looks for written to symbols;
        if ``mode==1`` looks for read symbols."""
        if isinstance(node, Symbol):
            return [node]
        elif isinstance(node, FlatBlock):
            return []
        elif isinstance(node, (Assign, Incr, Decr)):
            if mode == 0:
                return self._accessed_syms(node.children[0], mode)
            elif mode == 1:
                return self._accessed_syms(node.children[1], mode)
        elif isinstance(node, Decl):
            if mode == 0 and node.init and not isinstance(node.init, EmptyStatement):
                return self._accessed_syms(node.sym, mode)
            else:
                return []
        else:
            accessed_syms = []
            for n in node.children:
                accessed_syms.extend(self._accessed_syms(n, mode))
            return accessed_syms

    def _merge_loops(self, root, loop_a, loop_b):
        """Merge the body of ``loop_a`` in ``loop_b`` and eliminate ``loop_a``
        from the tree rooted in ``root``. Return a reference to the block
        containing the merged loop as well as the iteration variables used
        in the respective iteration spaces."""
        # Find the first statement in the perfect loop nest loop_b
        it_vars_a, it_vars_b = [], []
        while isinstance(loop_b.children[0], (Block, For)):
            if isinstance(loop_b, For):
                it_vars_b.append(loop_b.it_var())
            loop_b = loop_b.children[0]
        # Find the first statement in the perfect loop nest loop_a
        root_loop_a = loop_a
        while isinstance(loop_a.children[0], (Block, For)):
            if isinstance(loop_a, For):
                it_vars_a.append(loop_a.it_var())
            loop_a = loop_a.children[0]
        # Merge body of loop_a in loop_b
        loop_b.children[0:0] = loop_a.children
        # Remove loop_a from root
        root.children.remove(root_loop_a)
        return (loop_b, tuple(it_vars_a), tuple(it_vars_b))

    def _update_it_vars(self, node, it_vars):
        """Change the iteration variables in the nodes rooted in ``node``
        according to the map defined in ``it_vars``, which is a dictionary
        from old_iteration_variable to new_iteration_variable. For example,
        given it_vars = {'i': 'j'} and a node "A[i] = B[i]", change the node
        into "A[j] = B[j]"."""
        if isinstance(node, Symbol):
            new_rank = []
            for r in node.rank:
                new_rank.append(r if r not in it_vars else it_vars[r])
            node.rank = tuple(new_rank)
        elif not isinstance(node, FlatBlock):
            for n in node.children:
                self._update_it_vars(n, it_vars)

    def merge(self):
        """Merge perfect loop nests rooted in ``self.root``."""
        # {((start, bound, increment), ...) --> [outer_loop]}
        found_nests = defaultdict(list)
        written_syms = []
        # Collect some info visiting the tree rooted in node
        for n in self.root.children:
            if isinstance(n, For):
                # Track structure of iteration spaces
                found_nests[self._find_it_space(n)].append(n)
            else:
                # Track written variables
                written_syms.extend(self._accessed_syms(n, 0))

        # A perfect loop nest L1 is mergeable in a loop nest L2 if
        # 1 - their iteration space is identical; implicitly true because the keys,
        #     in the dictionary, are iteration spaces.
        # 2 - between the two nests, there are no statements that read from values
        #     computed in L1. This is checked next.
        # 3 - there are no read-after-write dependencies between variables written
        #     in L1 and read in L2. This is checked next.
        # Here, to simplify the data flow analysis, the last loop in the tree
        # rooted in node is selected as L2
        for itspace, loop_nests in found_nests.items():
            if len(loop_nests) == 1:
                # At least two loops are necessary for merging to be meaningful
                continue
            mergeable = []
            merging_in = loop_nests[-1]
            merging_in_read_syms = self._accessed_syms(merging_in, 1)
            for ln in loop_nests[:-1]:
                is_mergeable = True
                # Get the symbols written to in the loop nest ln
                ln_written_syms = self._accessed_syms(ln, 0)
                # Get the symbols written to between ln and merging_in (included)
                _written_syms = [self._accessed_syms(l, 0) for l in
                                 loop_nests[loop_nests.index(ln)+1:-1]]
                _written_syms = [i for l in _written_syms for i in l]  # list flattening
                _written_syms += written_syms
                # Check condition 2
                for ws, lws in itertools.product(_written_syms, ln_written_syms):
                    if self.expr_graph.has_dep(ws, lws):
                        is_mergeable = False
                        break
                # Check condition 3
                for lws, mirs in itertools.product(ln_written_syms,
                                                   merging_in_read_syms):
                    if lws.symbol == mirs.symbol and not lws.rank and not mirs.rank:
                        is_mergeable = False
                        break
                # Track mergeable loops
                if is_mergeable:
                    mergeable.append(ln)
            # If there is at least one mergeable loops, do the merging
            for l in reversed(mergeable):
                merged, l_itvars, m_itvars = self._merge_loops(self.root, l, merging_in)
                self._update_it_vars(merged, dict(zip(l_itvars, m_itvars)))


class ExprLoopFissioner(LoopScheduler):

    """Analyze data dependencies and iteration spaces, then fission associative
    operations in expressions.
    Fissioned expressions are placed in a separate loop nest."""

    def __init__(self, expr_graph, root, cut):
        """Initialize the ExprLoopFissioner.

        :arg cut: number of operands requested to fission expressions."""
        super(ExprLoopFissioner, self).__init__(expr_graph, root)
        self.cut = cut

    def _split_sum(self, node, parent, is_left, found, sum_count):
        """Exploit sum's associativity to cut node when a sum is found.
        Return ``True`` if a potentially splittable node is found, ``False``
        otherwise."""
        if isinstance(node, Symbol):
            return False
        elif isinstance(node, Par):
            return self._split_sum(node.children[0], (node, 0), is_left, found,
                                   sum_count)
        elif isinstance(node, Prod) and found:
            return False
        elif isinstance(node, Prod) and not found:
            if not self._split_sum(node.children[0], (node, 0), is_left, found,
                                   sum_count):
                return self._split_sum(node.children[1], (node, 1), is_left, found,
                                       sum_count)
            return True
        elif isinstance(node, Sum):
            sum_count += 1
            if not found:
                # Track the first Sum we found while cutting
                found = parent
            if sum_count == self.cut:
                # Perform the cut
                if is_left:
                    parent, parent_leaf = parent
                    parent.children[parent_leaf] = node.children[0]
                else:
                    found, found_leaf = found
                    found.children[found_leaf] = node.children[1]
                return True
            else:
                if not self._split_sum(node.children[0], (node, 0), is_left,
                                       found, sum_count):
                    return self._split_sum(node.children[1], (node, 1), is_left,
                                           found, sum_count)
                return True
        else:
            raise RuntimeError("Split error: found unknown node: %s" % str(node))

    def _sum_fission(self, expr, copy_loops):
        """Split an expression after ``cut`` operands. This results in two
        sub-expressions that are placed in different, although identical
        loop nests if ``copy_loops`` is true; they are placed in the same
        original loop nest otherwise. Return the two split expressions as a
        2-tuple, in which the second element is potentially further splittable."""
        expr_root, expr_info = expr
        it_vars, parent, loops = expr_info
        # Copy the original expression twice, and then split the two copies, that
        # we refer to as ``left`` and ``right``, meaning that the left copy will
        # be transformed in the sub-expression from the origin up to the cut point,
        # and analoguously for right.
        # For example, consider the expression a*b + c*d; the cut point is the sum
        # operator. Therefore, the left part is a*b, whereas the right part is c*d
        expr_root_left = dcopy(expr_root)
        expr_root_right = dcopy(expr_root)
        expr_left = Par(expr_root_left.children[1])
        expr_right = Par(expr_root_right.children[1])
        sleft = self._split_sum(expr_left.children[0], (expr_left, 0), True, None, 0)
        sright = self._split_sum(expr_right.children[0], (expr_right, 0), False, None, 0)

        if sleft and sright:
            index = parent.children.index(expr_root)
            # Append the left-split expression. Re-use a loop nest
            parent.children[index] = expr_root_left
            # Append the right-split (remainder) expression.
            if copy_loops:
                # Create a new loop nest
                new_loop = dcopy(loops[0])
                new_inner_loop = new_loop.children[0].children[0]
                new_inner_loop_block = new_inner_loop.children[0]
                new_inner_loop_block.children[0] = expr_root_right
                expr_right_loops = [new_loop, new_inner_loop]
                self.root.children.append(new_loop)
            else:
                parent.children.insert(index, expr_root_right)
                new_inner_loop_block, expr_right_loops = (parent, loops)
            # Attach info to the two split sub-expressions
            split = (expr_root_left, (it_vars, parent, loops))
            splittable = (expr_root_right, (it_vars, new_inner_loop_block,
                                            expr_right_loops))
            return (split, splittable)
        return ((expr_root, expr_info), ())

    def expr_fission(self, expr, copy_loops):
        """Split an expression containing ``x`` summands into ``x/cut`` chunks.
        Each chunk is placed in a separate loop nest if ``copy_loops`` is true,
        in the same loop nest otherwise. Return a dictionary of all of the split
        chunks, in which each entry has the same format of ``expr``.

        :arg expr:  the expression that needs to be split. This is given as
                    a tuple of two elements: the former is the expression
                    root node; the latter includes info about the expression,
                    particularly iteration variables of the enclosing loops,
                    the enclosing loops themselves, and the parent block.
        :arg copy_loops: true if the split expressions should be placed in two
                         separate, adjacent loop nests (iterating, of course,
                         along the same iteration space); false, otherwise."""

        split_exprs = {}
        splittable_expr = expr
        while splittable_expr:
            split_expr, splittable_expr = self._sum_fission(splittable_expr,
                                                            copy_loops)
            split_exprs[split_expr[0]] = split_expr[1]
        return split_exprs


class ZeroLoopScheduler(LoopScheduler):

    """Analyze data dependencies, iteration spaces, and domain-specific
    information to perform symbolic execution of the assembly code so as to
    determine how to restructure the loop nests to skip iteration over
    zero-valued columns.
    This implies that loops can be fissioned or merged. For example: ::

        for i = 0, N
          A[i] = C[i]*D[i]
          B[i] = E[i]*F[i]

    If the evaluation of A requires iterating over a region of contiguous
    zero-valued columns in C and D, then A is computed in a separate (smaller)
    loop nest: ::

        for i = 0 < (N-k)
          A[i+k] = C[i+k][i+k]
        for i = 0, N
          B[i] = E[i]*F[i]
    """

    def __init__(self, expr_graph, root, decls):
        """Initialize the ZeroLoopScheduler.

        :arg decls: lists of array declarations. A 2-tuple is expected: the first
                    element is the list of kernel declarations; the second element
                    is the list of hoisted temporaries declarations."""
        super(ZeroLoopScheduler, self).__init__(expr_graph, root)
        self.kernel_decls, self.hoisted_decls = decls
        # Track zero blocks in each symbol accessed in the computation rooted in root
        self.nz_in_syms = {}
        # Track blocks accessed for evaluating symbols in the various for loops
        # rooted in root
        self.nz_in_fors = OrderedDict()

    def _get_nz_bounds(self, node):
        if isinstance(node, Symbol):
            return (node.rank[-1], self.nz_in_syms[node.symbol])
        elif isinstance(node, Par):
            return self._get_nz_bounds(node.children[0])
        elif isinstance(node, Prod):
            return tuple([self._get_nz_bounds(n) for n in node.children])
        else:
            raise RuntimeError("Group iter space error: unknown node: %s" % str(node))

    def _merge_itvars_nz_bounds(self, itvar_nz_bounds_l, itvar_nz_bounds_r):
        """Given two dictionaries associating iteration variables to ranges
        of non-zero columns, merge the two dictionaries by combining ranges
        along the same iteration variables and return the merged dictionary.
        For example: ::

            dict1 = {'j': [(1,3), (5,6)], 'k': [(5,7)]}
            dict2 = {'j': [(3,4)], 'k': [(1,4)]}
            dict1 + dict2 -> {'j': [(1,6)], 'k': [(1,7)]}
        """
        new_itvar_nz_bounds = {}
        for itvar, nz_bounds in itvar_nz_bounds_l.items():
            if itvar.isdigit():
                # Skip constant dimensions
                continue
            # Compute the union of nonzero bounds along the same
            # iteration variable. Unify contiguous regions (for example,
            # [(1,3), (4,6)] -> [(1,6)]
            new_nz_bounds = nz_bounds + itvar_nz_bounds_r.get(itvar, ())
            merged_nz_bounds = itspace_merge(new_nz_bounds)
            new_itvar_nz_bounds[itvar] = merged_nz_bounds
        return new_itvar_nz_bounds

    def _set_var_to_zero(self, node, ofs, itspace):
        """Scan each variable ``v`` in ``node``: if non-initialized elements in ``v``
        are touched as iterating along ``itspace``, initialize ``v`` to 0.0."""

        def get_accessed_syms(node, nz_in_syms, found_syms):
            if isinstance(node, Symbol):
                nz_in_node = nz_in_syms.get(node.symbol)
                if nz_in_node:
                    nz_regions = dict(zip([r for r in node.rank], nz_in_node))
                    found_syms.append((node.symbol, nz_regions))
            else:
                for n in node.children:
                    get_accessed_syms(n, nz_in_syms, found_syms)

        # Determine the symbols accessed in node and their non-zero regions
        found_syms = []
        get_accessed_syms(node.children[1], self.nz_in_syms, found_syms)

        # If iteration space along which they are accessed is bigger than the
        # non-zero region, hoisted symbols must be initialized to zero
        for sym, nz_regions in found_syms:
            sym_decl = self.hoisted_decls.get(sym)
            if not sym_decl:
                continue
            for itvar, size in itspace:
                itvar_nz_regions = nz_regions.get(itvar)
                itvar_ofs = ofs.get(itvar)
                if not itvar_nz_regions or itvar_ofs is None:
                    # Sym does not iterate along this iteration variable, so skip
                    # the check
                    continue
                iteration_ok = False
                # Check that the iteration space actually corresponds to one of the
                # non-zero regions in the symbol currently analyzed
                for itvar_nz_region in itvar_nz_regions:
                    init_nz_reg, end_nz_reg = itvar_nz_region
                    if itvar_ofs == init_nz_reg and size == end_nz_reg + 1 - init_nz_reg:
                        iteration_ok = True
                        break
                if not iteration_ok:
                    # Iterating over a non-initialized region, need to zero it
                    sym_decl = sym_decl[0]
                    sym_decl.init = FlatBlock("{0.0}")

    def _track_expr_nz_columns(self, node):
        """Return the first and last indices assumed by the iteration variables
        appearing in ``node`` over regions of non-zero columns. For example,
        consider the following node, particularly its right-hand side: ::

        A[i][j] = B[i]*C[j]

        If B over i is non-zero in the ranges [0, k1] and [k2, k3], while C over
        j is non-zero in the range [N-k4, N], then return a dictionary: ::

        {i: ((0, k1), (k2, k3)), j: ((N-k4, N),)}

        If there are no zero-columns, return {}."""
        if isinstance(node, Symbol):
            if node.offset:
                raise RuntimeError("Zeros error: offsets not supported: %s" % str(node))
            nz_bounds = self.nz_in_syms.get(node.symbol)
            if nz_bounds:
                itvars = [r for r in node.rank]
                return dict(zip(itvars, nz_bounds))
            else:
                return {}
        elif isinstance(node, Par):
            return self._track_expr_nz_columns(node.children[0])
        else:
            itvar_nz_bounds_l = self._track_expr_nz_columns(node.children[0])
            itvar_nz_bounds_r = self._track_expr_nz_columns(node.children[1])
            if isinstance(node, (Prod, Div)):
                # Merge the nonzero bounds of different iteration variables
                # within the same dictionary
                return dict(itvar_nz_bounds_l.items() +
                            itvar_nz_bounds_r.items())
            elif isinstance(node, Sum):
                return self._merge_itvars_nz_bounds(itvar_nz_bounds_l,
                                                    itvar_nz_bounds_r)
            else:
                raise RuntimeError("Zeros error: unsupported operation: %s" % str(node))

    def _track_nz_blocks(self, node, parent=None, loop_nest=()):
        """Track the propagation of zero blocks along the computation which is
        rooted in ``self.root``.

        Before start tracking zero blocks in the nodes rooted in ``node``,
        ``self.nz_in_syms`` contains, for each known identifier, the ranges of
        its zero blocks. For example, assuming identifier A is an array and has
        zero-valued entries in positions from 0 to k and from N-k to N,
        ``self.nz_in_syms`` will contain an entry "A": ((0, k), (N-k, N)).
        If A is modified by some statements rooted in ``node``, then
        ``self.nz_in_syms["A"]`` will be modified accordingly.

        This method also updates ``self.nz_in_fors``, which maps loop nests to
        the enclosed symbols' non-zero blocks. For example, given the following
        code: ::

        { // root
          ...
          for i
            for j
              A = ...
              B = ...
        }

        Once traversed the AST, ``self.nz_in_fors`` will contain a (key, value)
        such that:
        ((<for i>, <for j>), root) -> {A: (i, (nz_along_i)), (j, (nz_along_j))}

        :arg node:      the node being currently inspected for tracking zero
                        blocks
        :arg parent:    the parent node of ``node``
        :arg loop_nest: tuple of for loops enclosing ``node``
        """
        if isinstance(node, (Assign, Incr, Decr)):
            symbol = node.children[0].symbol
            rank = node.children[0].rank
            itvar_nz_bounds = self._track_expr_nz_columns(node.children[1])
            if not itvar_nz_bounds:
                return
            # Reflect the propagation of non-zero blocks in the node's
            # target symbol. Note that by scanning loop_nest, the nonzero
            # bounds are stored in order. For example, if the symbol is
            # A[][], that is, it has two dimensions, then the first element
            # of the tuple stored in nz_in_syms[symbol] represents the nonzero
            # bounds for the first dimension, the second element the same for
            # the second dimension, and so on if it had had more dimensions.
            # Also, since nz_in_syms represents the propagation of non-zero
            # columns "up to this point of the computation", we have to merge
            # the non-zero columns produced by this node with those that we
            # had already found.
            nz_in_sym = tuple(itvar_nz_bounds[l.it_var()] for l in loop_nest
                              if l.it_var() in rank)
            if symbol in self.nz_in_syms:
                merged_nz_in_sym = []
                for i in zip(nz_in_sym, self.nz_in_syms[symbol]):
                    flat_nz_bounds = [nzb for nzb_sym in i for nzb in nzb_sym]
                    merged_nz_in_sym.append(itspace_merge(flat_nz_bounds))
                nz_in_sym = tuple(merged_nz_in_sym)
            self.nz_in_syms[symbol] = nz_in_sym
            if loop_nest:
                # Track the propagation of non-zero blocks in this specific
                # loop nest. Outer loops, i.e. loops that have non been
                # encountered as visiting from the root, are discarded.
                key = loop_nest[0]
                itvar_nz_bounds = dict([(k, v) for k, v in itvar_nz_bounds.items()
                                        if k in [l.it_var() for l in loop_nest]])
                if key not in self.nz_in_fors:
                    self.nz_in_fors[key] = []
                self.nz_in_fors[key].append((node, itvar_nz_bounds))
        if isinstance(node, For):
            self._track_nz_blocks(node.children[0], node, loop_nest + (node,))
        if isinstance(node, Block):
            for n in node.children:
                self._track_nz_blocks(n, node, loop_nest)

    def _track_nz_from_root(self):
        """Track the propagation of zero columns along the computation which is
        rooted in ``self.root``."""

        # Initialize a dict mapping symbols to their zero columns with the info
        # already available in the kernel's declarations
        for i, j in self.kernel_decls.items():
            nz_col_bounds = j[0].get_nonzero_columns()
            if nz_col_bounds:
                # Note that nz_bounds are stored as second element of a 2-tuple,
                # because the declared array is two-dimensional, in which the
                # second dimension represents the columns
                self.nz_in_syms[i] = (((0, j[0].sym.rank[0] - 1),),
                                      (nz_col_bounds,))
                if nz_col_bounds == (-1, -1):
                    # A fully zero-valued two dimensional array
                    self.nz_in_syms[i] = j[0].sym.rank

        # If zeros were not found, then just give up
        if not self.nz_in_syms:
            return {}

        # Track propagation of zero blocks by symbolically executing the code
        self._track_nz_blocks(self.root)

    def reschedule(self):
        """Restructure the loop nests rooted in ``self.root`` based on the
        propagation of zero-valued columns along the computation. This, therefore,
        involves fissing and fusing loops so as to remove iterations spent
        performing arithmetic operations over zero-valued entries.
        Return a list of dictionaries, a dictionary for each loop nest encountered.
        Each entry in a dictionary is of the form {stmt: (itvars, parent, loops)},
        in which ``stmt`` is a statement found in the loop nest from which the
        dictionary derives, ``itvars`` is the tuple of the iteration variables of
        the enclosing loops, ``parent`` is the AST node in which the loop nest is
        rooted, ``loops`` is the tuple of loops composing the loop nest."""

        # First, symbolically execute the code starting from self.root to track
        # the propagation of zeros
        self._track_nz_from_root()

        # Consider two statements A and B, and their iteration spaces.
        # If the two iteration spaces have:
        # - Same size and same bounds: then put A and B in the same loop nest
        #   for i, for j
        #     W1[i][j] = W2[i][j]
        #     Z1[i][j] = Z2[i][j]
        # - Same size but different bounds: then put A and B in the same loop
        #   nest, but add suitable offsets to all of the involved iteration
        #   variables
        #   for i, for j
        #     W1[i][j] = W2[i][j]
        #     Z1[i+k][j+k] = Z2[i+k][j+k]
        # - Different size: then put A and B in two different loop nests
        #   for i, for j
        #     W1[i][j] = W2[i][j]
        #   for i, for j  // Different loop bounds
        #     Z1[i][j] = Z2[i][j]
        all_moved_stmts = []
        new_nz_in_fors = {}
        for loop, stmt_itspaces in self.nz_in_fors.items():
            fissioned_loops = defaultdict(list)
            # Fission the loops on an intermediate representation
            for stmt, stmt_itspace in stmt_itspaces:
                nz_bounds_list = [i for i in itertools.product(*stmt_itspace.values())]
                for nz_bounds in nz_bounds_list:
                    itvar_nz_bounds = tuple(zip(stmt_itspace.keys(), nz_bounds))
                    itspace, stmt_ofs = itspace_size_ofs(itvar_nz_bounds)
                    fissioned_loops[itspace].append((dcopy(stmt), stmt_ofs))
            # Generate the actual code.
            # The dictionary is sorted because we must first execute smaller
            # loop nests, since larger ones may depend on them
            moved_stmts = {}
            for itspace, stmt_ofs in sorted(fissioned_loops.items()):
                new_loops, inner_block = c_from_itspace_to_fors(itspace)
                for stmt, ofs in stmt_ofs:
                    dict_ofs = dict(ofs)
                    ast_update_ofs(stmt, dict_ofs)
                    self._set_var_to_zero(stmt, dict_ofs, itspace)
                    inner_block.children.append(stmt)
                    moved_stmts[stmt] = (tuple(i[0] for i in ofs), inner_block, new_loops)
                new_nz_in_fors[new_loops[0]] = stmt_ofs
                # Append the created loops to the root
                index = self.root.children.index(loop)
                self.root.children.insert(index, new_loops[-1])
            self.root.children.remove(loop)
            all_moved_stmts.append(moved_stmts)

        self.nz_in_fors = new_nz_in_fors
        return all_moved_stmts


class ExpressionGraph(object):

    """Track read-after-write dependencies between symbols."""

    def __init__(self):
        self.deps = nx.DiGraph()

    def add_dependency(self, sym, expr, self_loop):
        """Extract symbols from ``expr`` and create a read-after-write dependency
        with ``sym``. If ``sym`` already has a dependency, then ``sym`` has a
        self dependency on itself."""

        def extract_syms(sym, node, deps):
            if isinstance(node, Symbol):
                deps.add_edge(sym, node.symbol)
            else:
                for n in node.children:
                    extract_syms(sym, n, deps)

        sym = sym.symbol
        # Add self-dependency
        if self_loop:
            self.deps.add_edge(sym, sym)
        extract_syms(sym, expr, self.deps)

    def has_dep(self, sym, target_sym=None):
        """If ``target_sym`` is not provided, return True if ``sym`` has a
        read-after-write dependency with some other symbols. This is the case if
        ``sym`` has either a self dependency or at least one input edge, meaning
        that other symbols depend on it.
        Otherwise, if ``target_sym`` is not None, return True if ``sym`` has a
        read-after-write dependency on it, i.e. if there is an edge from
        ``target_sym`` to ``sym``."""

        sym = sym.symbol
        if not target_sym:
            return sym in self.deps and zip(*self.deps.in_edges(sym))
        else:
            target_sym = target_sym.symbol
            return sym in self.deps and self.deps.has_edge(sym, target_sym)
