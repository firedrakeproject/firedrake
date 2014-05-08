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

from collections import defaultdict
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
                opts = node.pragma.split(" ", 2)
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
                    raise RuntimeError("Unrecognised pragma found '%s'", node.pragma)

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

    def generalized_licm(self, level):
        """Generalized loop-invariant code motion.

        :arg level: The optimization level (0, 1, 2, 3). The higher, the more
                    invasive is the re-writing of the assembly expressions,
                    trying to hoist as much invariant code as possible.
        """

        parent = (self.pre_header, self.kernel_decls)
        for expr in self.asm_expr.items():
            ew = AssemblyRewriter(expr, self.int_loop, self.sym, self.decls, parent)
            if level > 0:
                ew.licm()
            if level > 1:
                ew.expand()
                ew.distribute()
                ew.licm()

    def slice_loop(self, slice_factor=None):
        """Perform slicing of the innermost loop to enhance register reuse.
        For example, given a loop:

        for i = 0 to N
          f()

        the following sequence of loops is generated:

        for i = 0 to k
          f()
        for i = k to 2k
          f()
        ...
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

    def split(self, cut=1, length=0):
        """Split assembly to improve resources utilization (e.g. vector registers).
        The splitting ``cuts`` the expressions into ``length`` blocks of ``cut``
        outer products.

        For example:
        for i
          for j
            A[i][j] += X[i]*Y[j] + Z[i]*K[j] + B[i]*X[j]
        with cut=1, length=1 this would be transformed into:
        for i
          for j
            A[i][j] += X[i]*Y[j]
        for i
          for j
            A[i][j] += Z[i]*K[j] + B[i]*X[j]

        If ``length`` is 0, then ``cut`` is ignored, and the expression is fully cut
        into chunks containing a single outer product."""

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
                raise RuntimeError("Splitting expression, but actually found an unknown \
                                    node: %s" % node.gencode())

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
            return split, splittable

        if not self.asm_expr:
            return

        new_asm_expr = {}
        splittable = self.asm_expr
        if length:
            # Split into at most length blocks
            for i in range(length-1):
                split, splittable = split_and_update(splittable)
                new_asm_expr.update(split)
                if not splittable:
                    break
            if splittable:
                new_asm_expr.update(splittable)
        else:
            # Split everything into blocks of length 1
            while splittable:
                split, splittable = split_and_update(splittable)
                new_asm_expr.update(split)
        self.asm_expr = new_asm_expr


class AssemblyRewriter(object):
    """Provide operations to re-write an assembly expression:
        - Loop-invariant code motion: find and hoist sub-expressions which are
        invariant with respect to an assembly loop
        - Expansion: transform an expression (a + b)*c into (a*c + b*c)
        - Distribute: transform an expression a*b + a*c into a*(b+c)"""

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
        self.hoisted = {}
        # Properties of the assembly expression
        self._licm = 0
        self._expanded = False
        # The expression graph tracks symbols dependencies
        self.eg = ExpressionGraph()

    def licm(self):
        """Perform loop-invariant code motion.

        Invariant expressions found in the loop nest are moved "after" the
        outermost independent loop and "after" the fastest varying dimension
        loop. Here, "after" means that if the loop nest has two loops i and j,
        and j is in the body of i, then i comes after j (i.e. the loop nest
        has to be read from right to left).

        For example, if a sub-expression E depends on [i, j] and the loop nest
        has three loops [i, j, k], then E is hoisted out from the body of k to
        the body of i). All hoisted expressions are then wrapped within a
        suitable loop in order to exploit compiler autovectorization. Note that
        this applies to constant sub-expressions as well, in which case hoisting
        after the outermost loop takes place."""

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
        expression a*(5+c) + b*(a+4), return {a: 2, b: 1, c: 1}."""

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
        """Expand assembly expressions such that:

        Y[j] = f(...)
        (X[i]*Y[j])*F + ...

        becomes:

        Y[j] = f(...)*F
        (X[i]*Y[j]) + ...

        This may be useful for several purposes:
        - Relieve register pressure; when, for example, (X[i]*Y[j]) is computed
        in a loop L' different than the loop L'' in which Y[j] is evaluated,
        and cost(L') > cost(L'')
        - It is also a step towards exposing well-known linear algebra operations,
        like matrix-matrix multiplies."""

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
        self._expanded = True

    def distribute(self):
        """Apply to the distributivity property to the assembly expression.
        E.g. A[i]*B[j] + A[i]*C[j] becomes A[i]*(B[j] + C[j])."""

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


class ExpressionExpander(object):
    """Expand assembly expressions such that:

    Y[j] = f(...)
    (X[i]*Y[j])*F + ...

    becomes:

    Y[j] = f(...)*F
    (X[i]*Y[j]) + ..."""

    CONST = -1
    ITVAR = -2

    def __init__(self, var_info, eg, expr):
        self.var_info = var_info
        self.eg = eg
        self.counter = 0
        self.parent = expr
        self.expanded_decls = {}

    def _do_expand(self, sym, const):
        """Perform the actual expansion. If there are no dependencies, then
        the already hoisted expression is expanded. Otherwise, if the symbol to
        be expanded occurs multiple times in the expression, or it depends on
        other hoisted symbols that will also be expanded, create a new symbol."""

        old_expr, var_decl, inv_for, place = self.var_info[sym.symbol]

        # No dependencies, just perform the expansion
        if not self.eg.has_dep(sym):
            old_expr.children[0] = Prod(Par(old_expr.children[0]), const)
            return

        # Create a new symbol, expression, and declaration
        new_expr = Par(Prod(dcopy(sym), const))
        new_node = Assign(sym, new_expr)
        sym.symbol += "_EXP%d" % self.counter
        new_var_decl = dcopy(var_decl)
        new_var_decl.sym.symbol = sym.symbol
        # Append new expression and declaration
        inv_for.children[0].children.append(new_node)
        place.insert(place.index(var_decl), new_var_decl)
        self.expanded_decls[new_var_decl.sym.symbol] = (new_var_decl, ast_plan.LOCAL_VAR)
        # Update tracked information
        self.var_info[sym.symbol] = (new_expr, new_var_decl, inv_for, place)
        self.eg.add_dependency(sym, new_expr, 0)

        self.counter += 1

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

    def has_dep(self, sym):
        """Return True if ``sym`` has a read-after-write dependency with some
        other symbols. This is the case if ``sym`` has either a self dependency
        or at least one input edge, meaning that other symbols depend on it."""

        sym = sym.symbol
        return sym in self.deps and zip(*self.deps.in_edges(sym))
