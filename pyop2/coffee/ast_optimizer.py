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

from collections import defaultdict, OrderedDict
import itertools
from copy import deepcopy as dcopy

import networkx as nx

from pyop2.coffee.ast_base import *
import ast_plan


class AssemblyOptimizer(object):

    """Loops optimiser:

    * Loop Invariant Code Motion (LICM)
      Backend compilers apply LICM to innermost loops only. In addition,
      hoisted expressions are usually not vectorized. Here, we apply LICM to
      terms which are known to be constant (i.e. they are declared const)
      and all of the loops in the nest are searched for sub-expressions
      involving such const terms only. Also, hoisted terms are wrapped
      within loops to exploit compiler autovectorization. This has proved to
      be beneficial for loop nests in which the bounds of all loops are
      relatively small (let's say less than 50-60).

    * register tiling:
      Given a rectangular iteration space, register tiling slices it into
      square tiles of user-provided size, with the aim of improving register
      pressure and register re-use."""

    def __init__(self, loop_nest, pre_header, kernel_decls):
        self.pre_header = pre_header
        self.kernel_decls = kernel_decls
        # Track applied optimizations
        self._is_precomputed = False
        self._has_zeros = False
        # Expressions evaluating the element matrix
        self.asm_expr = {}
        # Integration loop (if any)
        self.int_loop = None
        # Fully parallel iteration space in the assembly loop nest
        self.asm_itspace = []
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

    def rewrite_expression(self, level):
        """Generalized loop-invariant code motion.

        :arg level: The optimization level (0, 1, 2, 3, 4). The higher, the more
                    invasive is the re-writing of the assembly expressions,
                    trying to eliminate unnecessary floating point operations.
                    level == 1: performs "basic" generalized loop-invariant
                                code motion
                    level == 2: level 1 + expansion of terms, factorization of
                                basis functions appearing multiple times in the
                                same expression, and finally another run of
                                loop-invariant code motion to move invariant
                                sub-expressions exposed by factorization
                    level == 3: level 2 + avoid computing zero-columns
                    level == 4: level 3 + precomputation of read-only expressions
                                out of the assembly loop nest
        """

        parent = (self.pre_header, self.kernel_decls)
        for expr in self.asm_expr.items():
            ew = AssemblyRewriter(expr, self.int_loop, self.sym, self.decls, parent)
            # Perform expression rewriting
            if level > 0:
                ew.licm()
            if level > 1:
                ew.expand()
                ew.distribute()
                ew.licm()
                # Fuse loops iterating along the same iteration space
                ew_parent = self.int_loop.children[0] if self.int_loop else self.pre_header
                self._merge_perfect_loop_nests(ew_parent, ew.eg)
            # Eliminate zeros
            if level == 3:
                self._has_zeros = ew.zeros()
            # Precompute expressions
            if level == 4:
                self._precompute(expr)
                self._is_precomputed = True

    def slice_loop(self, slice_factor=None):
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

        :arg loops_factor:   dictionary from loops to unroll (factor, increment).
                             Loops are specified as integers: 0 = integration loop,
                             1 = test functions loop, 2 = trial functions loop.
                             A factor of 0 denotes that the corresponding loop
                             is not present.
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

    def permute_int_loop(self):
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

    def split(self, cut=1, length=0):
        """Split assembly expressions into multiple chunks exploiting sum's
        associativity.
        In "normal" circumstances, this transformation "splits" an expression into at most
        ``length`` chunks of ``cut`` operands. There are, however, special cases:
        If zeros were found while rewriting the assembly expression, ``length`` is ignored
        and the expression is split into X chunks, with X being the number of iteration
        spaces required to correctly perform the assembly.
        If ``length == 0``, the expression is completely split into chunks of one operand.

        For example, consider the following piece of code:

        .. code-block:: none

            for i
              for j
                A[i][j] += X[i]*Y[j] + Z[i]*K[j] + B[i]*X[j]

        If ``cut=1`` and ``length=1``, the cut is applied at most length=1 times, and this
        is transformed into:

        .. code-block:: none

            for i
              for j
                A[i][j] += X[i]*Y[j]
            // Remainder of the splitting:
            for i
              for j
                A[i][j] += Z[i]*K[j] + B[i]*X[j]

        If ``cut=1`` and ``length=0``, length is ignored and the expression is cut into chunks
        of size ``cut=1``:

        .. code-block:: none

            for i
              for j
                A[i][j] += X[i]*Y[j]
            for i
              for j
                A[i][j] += Z[i]*K[j]
            for i
              for j
                A[i][j] += B[i]*X[j]

        If ``cut=2`` and ``length=0``, length is ignored and the expression is cut into chunks
        of size ``cut=2``:

        .. code-block:: none

            for i
              for j
                A[i][j] += X[i]*Y[j] + Z[i]*K[j]
            // Remainder of the splitting:
            for i
              for j
                A[i][j] += B[i]*X[j]
        """

        def check_sum(par_node):
            """Return true if there are no sums in the sub-tree rooted in
            par_node, false otherwise."""
            if isinstance(par_node, Symbol):
                return False
            elif isinstance(par_node, Sum):
                return True
            elif isinstance(par_node, Par):
                return check_sum(par_node.children[0])
            elif isinstance(par_node, Prod):
                left = check_sum(par_node.children[0])
                right = check_sum(par_node.children[1])
                return left or right
            else:
                raise RuntimeError("Split error: found unknown node %s:" % str(par_node))

        def split_sum(node, parent, is_left, found, sum_count):
            """Exploit sum's associativity to cut node when a sum is found."""
            if isinstance(node, Symbol):
                return False
            elif isinstance(node, Par):
                return split_sum(node.children[0], (node, 0), is_left, found, sum_count)
            elif isinstance(node, Prod) and found:
                return False
            elif isinstance(node, Prod) and not found:
                if not split_sum(node.children[0], (node, 0), is_left, found, sum_count):
                    return split_sum(node.children[1], (node, 1), is_left, found, sum_count)
                return True
            elif isinstance(node, Sum):
                sum_count += 1
                if not found:
                    # Track the first Sum we found while cutting
                    found = parent
                if sum_count == cut:
                    # Perform the cut
                    if is_left:
                        parent, parent_leaf = parent
                        parent.children[parent_leaf] = node.children[0]
                    else:
                        found, found_leaf = found
                        found.children[found_leaf] = node.children[1]
                    return True
                else:
                    if not split_sum(node.children[0], (node, 0), is_left, found, sum_count):
                        return split_sum(node.children[1], (node, 1), is_left, found, sum_count)
                    return True
            else:
                raise RuntimeError("Split error: found unknown node: %s" % str(node))

        def split_and_update(out_prods):
            split, splittable = ({}, {})
            for stmt, stmt_info in out_prods.items():
                it_vars, parent, loops = stmt_info
                stmt_left = dcopy(stmt)
                stmt_right = dcopy(stmt)
                expr_left = Par(stmt_left.children[1])
                expr_right = Par(stmt_right.children[1])
                sleft = split_sum(expr_left.children[0], (expr_left, 0), True, None, 0)
                sright = split_sum(expr_right.children[0], (expr_right, 0), False, None, 0)

                if sleft and sright:
                    # Append the left-split expression. Re-use loop nest
                    parent.children[parent.children.index(stmt)] = stmt_left
                    # Append the right-split (reminder) expression. Create new loop nest
                    split_loop = dcopy([f for f in self.fors if f.it_var() == it_vars[0]][0])
                    split_inner_loop = split_loop.children[0].children[0].children[0]
                    split_inner_loop.children[0] = stmt_right
                    place = self.int_loop.children[0] if self.int_loop else self.pre_header
                    place.children.append(split_loop)
                    stmt_right_loops = [split_loop, split_loop.children[0].children[0]]
                    # Update outer product dictionaries
                    splittable[stmt_right] = (it_vars, split_inner_loop, stmt_right_loops)
                    if check_sum(stmt_left.children[1]):
                        splittable[stmt_left] = (it_vars, parent, loops)
                    else:
                        split[stmt_left] = (it_vars, parent, loops)
                else:
                    split[stmt] = stmt_info
            return split, splittable

        if not self.asm_expr:
            return

        new_asm_expr = {}
        splittable = self.asm_expr
        if length and not self._has_zeros:
            # Split into at most ``length`` blocks
            for i in range(length-1):
                split, splittable = split_and_update(splittable)
                new_asm_expr.update(split)
                if not splittable:
                    break
            if splittable:
                new_asm_expr.update(splittable)
        else:
            # Split everything into blocks of length 1
            cut = 1
            while splittable:
                split, splittable = split_and_update(splittable)
                new_asm_expr.update(split)
            new_asm_expr.update(splittable)
            if self._has_zeros:
                # Group assembly expressions that have the same iteration space
                new_asm_expr = self._group_itspaces(new_asm_expr)
        self.asm_expr = new_asm_expr

    def _group_itspaces(self, asm_expr):
        """Group the expressions in ``asm_expr`` that iterate along the same space
        and return an updated version of the dictionary containing the assembly
        expressions in the kernel."""
        def get_nonzero_bounds(node):
            if isinstance(node, Symbol):
                return (node.rank[-1], self._has_zeros[node.symbol])
            elif isinstance(node, Par):
                return get_nonzero_bounds(node.children[0])
            elif isinstance(node, Prod):
                return tuple([get_nonzero_bounds(n) for n in node.children])
            else:
                raise RuntimeError("Group iter space error: unknown node: %s" % str(node))

        def get_size_and_ofs(itspace):
            """Given an ``itspace`` in the form (('itvar', (bound_a, bound_b), ...)),
            return ((('it_var', bound_b - bound_a), ...), (('it_var', bound_a), ...))"""
            itspace_info = []
            for var, bounds in itspace:
                itspace_info.append(((var, bounds[1] - bounds[0]), (var, bounds[0])))
            return tuple(zip(*itspace_info))

        def update_ofs(node, ofs):
            """Given a dictionary ``ofs`` s.t. {'itvar': ofs}, update the various
            iteration variables in the symbols rooted in ``node``."""
            if isinstance(node, Symbol):
                new_ofs = []
                old_ofs = ((1, 0) for r in node.rank) if not node.offset else node.offset
                for r, o in zip(node.rank, old_ofs):
                    new_ofs.append((o[0], ofs[r] if r in ofs else o[1]))
                node.offset = tuple(new_ofs)
            else:
                for n in node.children:
                    update_ofs(n, ofs)

        # If two iteration spaces have:
        # - Same size and same bounds: then generate a single statement, e.g.
        #   for i, for j
        #     A[i][j] += B[i][j] + C[i][j]
        # - Same size but different bounds: then generate two statements in the same
        #   iteration space:
        #   for i, for j
        #     A[i][j] += B[i][j]
        #     A[i+k][j+k] += C[i+k][j+k]
        # - Different size: then generate two iteration spaces
        # So, group increments according to the size of their iteration space, and
        # also save the offset within that iteration space
        itspaces = defaultdict(list)
        for expr, expr_info in asm_expr.items():
            nonzero_bounds = get_nonzero_bounds(expr.children[1])
            itspace_info = get_size_and_ofs(nonzero_bounds)
            itspaces[itspace_info[0]].append((expr, expr_info, itspace_info[1]))

        # Create the new iteration spaces
        to_remove = []
        new_asm_expr = {}
        for its, asm_exprs in itspaces.items():
            itvar_to_size = dict(its)
            expr, expr_info, ofs = asm_exprs[0]
            it_vars, parent, loops = expr_info
            # Reuse and modify an existing loop nest
            outer_loop_size = itvar_to_size[loops[0].it_var()]
            inner_loop_size = itvar_to_size[loops[1].it_var()]
            loops[0].cond.children[1] = c_sym(outer_loop_size + 1)
            loops[1].cond.children[1] = c_sym(inner_loop_size + 1)
            # Update memory offsets in the expression
            update_ofs(expr, dict(ofs))
            new_asm_expr[expr] = expr_info
            # Track down loops that will have to be removed
            for _expr, _expr_info, _ofs in asm_exprs[1:]:
                to_remove.append(_expr_info[2][0])
                parent.children.append(_expr)
                new_asm_expr[_expr] = expr_info
                update_ofs(_expr, dict(_ofs))
        # Remove old loops
        parent = self.int_loop.children[0] if self.int_loop else self.pre_header
        for i in to_remove:
            parent.children.remove(i)
        # Update the dictionary of assembly expressions in the kernel
        return new_asm_expr

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

    def _merge_perfect_loop_nests(self, node, eg):
        """Merge loop nests rooted in ``node`` having the same iteration space.
        This assumes that the statements rooted in ``node`` are in SSA form:
        no data dependency analysis is performed, i.e. the safety of the
        transformation must be checked by the caller. Also, the loop nests are
        assumed to be perfect; again, this must be ensured by the caller.

        :arg node: root of the tree to inspect for merging loops
        :arg eg: expression graph, used to check there are no read-after-write
                 dependencies between two loops.
        """

        def find_iteration_space(node):
            """Return the iteration space of the loop nest rooted in ``node``,
            as tuple of 3-tuple, in which each 3-tuple is of the form
            (start, bound, increment)."""
            if isinstance(node, For):
                itspace = (node.start(), node.end(), node.increment())
                child_itspace = find_iteration_space(node.children[0].children[0])
                return (itspace, child_itspace) if child_itspace else (itspace,)

        def writing_syms(node):
            """Return a list of symbols that are being written to in the tree
            rooted in ``node``."""
            if isinstance(node, Symbol):
                return [node]
            elif isinstance(node, FlatBlock):
                return []
            elif isinstance(node, (Assign, Incr, Decr)):
                return writing_syms(node.children[0])
            elif isinstance(node, Decl):
                if node.init and not isinstance(node.init, EmptyStatement):
                    return writing_syms(node.sym)
                else:
                    return []
            else:
                written_syms = []
                for n in node.children:
                    written_syms.extend(writing_syms(n))
                return written_syms

        def merge_loops(root, loop_a, loop_b):
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

        def update_iteration_variables(node, it_vars):
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
                    update_iteration_variables(n, it_vars)

        # {((start, bound, increment), ...) --> [outer_loop]}
        found_nests = defaultdict(list)
        written_syms = []
        # Collect some info visiting the tree rooted in node
        for n in node.children:
            if isinstance(n, For):
                # Track structure of iteration spaces
                found_nests[find_iteration_space(n)].append(n)
            else:
                # Track written variables
                written_syms.extend(writing_syms(n))

        # A perfect loop nest L1 is mergeable in a loop nest L2 if
        # - their iteration space is identical; implicitly true because the keys,
        #   in the dictionary, are iteration spaces.
        # - between the two nests, there are no statements that read from values
        #   computed in L1. This is checked next.
        # Here, to simplify the data flow analysis, the last loop in the tree
        # rooted in node is selected as L2
        for itspace, loop_nests in found_nests.items():
            if len(loop_nests) == 1:
                # At least two loops are necessary for merging to be meaningful
                continue
            mergeable = []
            merging_in = loop_nests[-1]
            for ln in loop_nests[:-1]:
                is_mergeable = True
                # Get the symbols written to in the loop nest ln
                ln_written_syms = writing_syms(ln)
                # Get the symbols written to between ln and merging_in (included)
                _written_syms = [writing_syms(l) for l in loop_nests[loop_nests.index(ln)+1:-1]]
                _written_syms = [i for l in _written_syms for i in l]  # list flattening
                _written_syms += written_syms
                for ws, lws in itertools.product(_written_syms, ln_written_syms):
                    if eg.has_dep(ws, lws):
                        is_mergeable = False
                        break
                # Track mergeable loops
                if is_mergeable:
                    mergeable.append(ln)
            # If there is at least one mergeable loops, do the merging
            for l in reversed(mergeable):
                merged, l_itvars, m_itvars = merge_loops(node, l, merging_in)
                update_iteration_variables(merged, dict(zip(l_itvars, m_itvars)))


class AssemblyRewriter(object):
    """Provide operations to re-write an assembly expression:

    * Loop-invariant code motion: find and hoist sub-expressions which are
      invariant with respect to an assembly loop
    * Expansion: transform an expression ``(a + b)*c`` into ``(a*c + b*c)``
    * Distribute: transform an expression ``a*b + a*c`` into ``a*(b+c)``"""

    def __init__(self, expr, int_loop, syms, decls, parent):
        """Initialize the AssemblyRewriter.

        :arg expr:     provide generic information related to an assembly expression,
                       including the depending for loops.
        :arg int_loop: the loop along which integration is performed.
        :arg syms:     list of AST symbols used to evaluate the local element matrix.
        :arg decls:    list of AST declarations of the various symbols in ``syms``.
        :arg parent:   the parent AST node of the assembly loop nest.
        """
        self.expr, self.expr_info = expr
        self.int_loop = int_loop
        self.syms = syms
        self.decls = decls
        self.parent, self.parent_decls = parent
        self.hoisted = OrderedDict()
        # Properties of the assembly expression
        self._licm = 0
        self._expanded = False
        # The expression graph tracks symbols dependencies
        self.eg = ExpressionGraph()

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
                    self.eg.add_dependency(s, e, n_replaced[str(s)] > 1)

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
        ee = ExpressionExpander(self.hoisted, self.eg, self.parent)
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

    def zeros(self):
        """Track the propagation of zero columns along the computation and re-write
        the assembly expressions so as to avoid useless floating point operations
        over zero values."""

        def track_nonzero_columns(node, nonzeros_in_syms):
            """Return the first and last indices of non-zero columns resulting from
            the evaluation of the expression rooted in node. If there are no zero
            columns or if the expression is not made of bi-dimensional arrays,
            return (None, None)."""
            if isinstance(node, Symbol):
                if node.offset:
                    raise RuntimeError("Zeros error: offsets not supported: %s" % str(node))
                return nonzeros_in_syms.get(node.symbol)
            elif isinstance(node, Par):
                return track_nonzero_columns(node.children[0], nonzeros_in_syms)
            else:
                nz_bounds = [track_nonzero_columns(n, nonzeros_in_syms) for n in node.children]
                if isinstance(node, (Prod, Div)):
                    indices = [nz for nz in nz_bounds if nz and nz != (None, None)]
                    if len(indices) == 0:
                        return (None, None)
                    elif len(indices) > 1:
                        raise RuntimeError("Zeros error: unexpected operation: %s" % str(node))
                    else:
                        return indices[0]
                elif isinstance(node, Sum):
                    indices = [None, None]
                    for nz in nz_bounds:
                        if nz is not None:
                            indices[0] = nz[0] if not indices[0] else min(nz[0], indices[0])
                            indices[1] = nz[1] if not indices[1] else max(nz[1], indices[1])
                    return tuple(indices)
                else:
                    raise RuntimeError("Zeros error: unsupported operation: %s" % str(node))

        # Initialize a dict mapping symbols to their zero columns with the info
        # already available in the kernel's declarations
        nonzeros_in_syms = {}
        for i, j in self.parent_decls.items():
            nz_bounds = j[0].get_nonzero_columns()
            if nz_bounds:
                nonzeros_in_syms[i] = nz_bounds
                if nz_bounds == (-1, -1):
                    # A fully zero-valued two dimensional array
                    nonzeros_in_syms[i] = j[0].sym.rank

        # If zeros were not found, then just give up
        if not nonzeros_in_syms:
            return {}

        # Now track zeros in the temporaries storing hoisted sub-expressions
        for i, j in self.hoisted.items():
            nz_bounds = track_nonzero_columns(j[0], nonzeros_in_syms) or (None, None)
            if None not in nz_bounds:
                # There are some zero-columns in the array, so track the bounds
                # of *non* zero-columns
                nonzeros_in_syms[i] = nz_bounds
            else:
                # Dense array or scalar cases: need to ignore scalars
                sym_size = j[1].size()[-1]
                if sym_size:
                    nonzeros_in_syms[i] = (0, sym_size)

        # Record the fact that we are tracking zeros
        return nonzeros_in_syms


class ExpressionExpander(object):
    """Expand assembly expressions such that: ::

        Y[j] = f(...)
        (X[i]*Y[j])*F + ...

    becomes: ::

        Y[j] = f(...)*F
        (X[i]*Y[j]) + ..."""

    CONST = -1
    ITVAR = -2

    def __init__(self, var_info, eg, expr):
        self.var_info = var_info
        self.eg = eg
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
            self.expanded_decls[new_const_decl.sym.symbol] = (new_const_decl, ast_plan.LOCAL_VAR)
            self.expanded_syms.append(new_const_decl.sym)
            place.insert(place.index(inv_for), new_const_decl)
            self.found_consts[const_str] = const_sym
            const = const_sym

        # No dependencies, just perform the expansion
        if not self.eg.has_dep(sym):
            old_expr.children[0] = Prod(Par(old_expr.children[0]), dcopy(const))
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
        self.eg.add_dependency(sym, new_expr, 0)

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
