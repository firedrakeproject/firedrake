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

    def generalized_licm(self):
        """Generalized loop-invariant code motion."""

        nest = (self.fors, self.sym, self.decls)
        parent = (self.pre_header, self.kernel_decls)
        for expr in self.asm_expr.items():
            ew = AssemblyRewriter(expr, nest, parent)
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

    def split(self, cut, length):
        """Split outer product RHS to improve resources utilization (e.g.
        vector registers)."""

        def split_sum(node, parent, is_left, found, sum_count):
            """Exploit sum's associativity to cut node when a sum is found."""
            if isinstance(node, Symbol):
                return False
            elif isinstance(node, Par) and found:
                return False
            elif isinstance(node, Par) and not found:
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
                    found = parent
                if sum_count == cut:
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
                raise RuntimeError("Splitting expression, shouldn't be here.")

        def split_and_update(asm_expr):
            split, splittable = ({}, {})
            for stmt, stmt_info in asm_expr.items():
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
                    self.fors[0].children[0].children.append(split_loop)
                    stmt_right_loops = [split_loop, split_loop.children[0].children[0]]
                    # Update outer product dictionaries
                    splittable[stmt_right] = (it_vars, split_inner_loop, stmt_right_loops)
                    split[stmt_left] = (it_vars, parent, loops)
                    return split, splittable
                else:
                    return asm_expr, {}

        if not self.asm_expr:
            return

        new_asm_expr = {}
        splittable = self.asm_expr
        for i in range(length-1):
            split, splittable = split_and_update(splittable)
            new_asm_expr.update(split)
            if not splittable:
                break
        if splittable:
            new_asm_expr.update(splittable)
        self.asm_expr = new_asm_expr


class AssemblyRewriter(object):
    """Rewrite assembly expressions according to the following expansion
    rules."""

    def __init__(self, expr, nest, parent):
        self.expr, self.expr_info = expr
        self.nest_loops, self.nest_syms, self.nest_decls = nest
        self.parent, self.parent_decls = parent
        self.hoisted = {}

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

        def extract_const(node, expr_dep):
            if isinstance(node, Symbol):
                return (node.loop_dep, node.symbol != self.expr.children[0].symbol)
            if isinstance(node, Par):
                return (extract_const(node.children[0], expr_dep))

            # Traverse the expression tree
            left, right = node.children
            dep_left, invariant_l = extract_const(left, expr_dep)
            dep_right, invariant_r = extract_const(right, expr_dep)

            if dep_left == dep_right:
                # Children match up, keep traversing the tree in order to see
                # if this sub-expression is actually a child of a larger
                # loop-invariant sub-expression
                return (dep_left, True)
            elif len(dep_left) == 0:
                # The left child does not depend on any iteration variable,
                # so it's loop invariant
                return (dep_right, True)
            elif len(dep_right) == 0:
                # The right child does not depend on any iteration variable,
                # so it's loop invariant
                return (dep_left, True)
            else:
                # Iteration variables of the two children do not match, add
                # the children to the dict of invariant expressions iff
                # they were invariant w.r.t. some loops and not just symbols
                if invariant_l:
                    left = Par(left) if isinstance(left, Symbol) else left
                    expr_dep[dep_left].append(left)
                if invariant_r:
                    right = Par(right) if isinstance(right, Symbol) else right
                    expr_dep[dep_right].append(right)
                return ((), False)

        def replace_const(node, syms_dict):
            if isinstance(node, Symbol):
                if str(Par(node)) in syms_dict:
                    return True
                else:
                    return False
            if isinstance(node, Par):
                if str(node) in syms_dict:
                    return True
                else:
                    return replace_const(node.children[0], syms_dict)
            # Found invariant sub-expression
            if node in syms_dict:
                return True

            # Traverse the expression tree and replace
            left = node.children[0]
            right = node.children[1]
            if replace_const(left, syms_dict):
                left = Par(left) if isinstance(left, Symbol) else left
                node.children[0] = syms_dict[str(left)]
            if replace_const(right, syms_dict):
                right = Par(right) if isinstance(right, Symbol) else right
                node.children[1] = syms_dict[str(right)]
            return False

        # Extract read-only sub-expressions that do not depend on at least
        # one loop in the loop nest
        expr_dep = defaultdict(list)
        typ = self.parent_decls[self.expr.children[0].symbol][0].typ
        extract_const(self.expr.children[1], expr_dep)

        for dep, expr in expr_dep.items():
            # 1) Determine the loops that should wrap invariant statements
            # and where such for blocks should be placed in the loop nest
            n_dep_for = None
            fast_for = None
            # Collect some info about the loops
            for l in self.nest_loops:
                if l.it_var() == dep[-1]:
                    fast_for = fast_for or l
                if l.it_var() not in dep:
                    n_dep_for = n_dep_for or l
                if l.it_var() == self.expr_info[0][0]:
                    op_loop = l
            if not fast_for or not n_dep_for:
                continue
            # Find where to put the new invariant for
            pre_loop = None
            for l in self.nest_loops:
                if l.it_var() not in [fast_for.it_var(), n_dep_for.it_var()]:
                    pre_loop = l
                else:
                    break
            if pre_loop:
                place = pre_loop.children[0]
                ofs = place.children.index(op_loop)
                wl = [fast_for]
            else:
                place = self.parent
                ofs = place.children.index(self.nest_loops[0])
                wl = [l for l in self.nest_loops if l.it_var() in dep]

            # 2) Remove identical sub-expressions
            expr = dict([(str(e), e) for e in expr]).values()

            # 3) Create the new loop
            sym_rank = tuple([l.size() for l in wl],)
            syms = [Symbol("LI_%s_%s" % (wl[0].it_var(), i), sym_rank)
                    for i in range(len(expr))]
            var_decl = [Decl(typ, _s) for _s in syms]
            for_rank = tuple([l.it_var() for l in reversed(wl)])
            for_sym = [Symbol(_s.sym.symbol, for_rank) for _s in var_decl]
            for_ass = [Assign(_s, e) for _s, e in zip(for_sym, expr)]
            block = Block(for_ass, open_scope=True)
            for l in wl:
                inv_for = For(dcopy(l.init), dcopy(l.cond), dcopy(l.incr), block)
                block = Block([inv_for], open_scope=True)

            # 4) Update the lists of symbols accessed and of decls
            self.nest_syms.update([d.sym for d in var_decl])
            lv = ast_plan.LOCAL_VAR
            self.nest_decls.update(dict(zip([d.sym.symbol for d in var_decl],
                                        [(v, lv) for v in var_decl])))

            # 5) Append the new node at the right level in the loop nest
            new_block = var_decl + [inv_for] + place.children[ofs:]
            place.children = place.children[:ofs] + new_block

            # 6) Track hoisted symbols
            self.hoisted.update(zip(for_sym, [(i, inv_for) for i in expr]))

            # 7) Replace invariant sub-trees with the proper tmp variable
            replace_const(self.expr.children[1], dict(zip([str(i) for i in expr], for_sym)))

    def count_occurrences(self):
        """For each variable in the assembly expression, count how many times
        it appears as involved in some operations. For example, for the
        expression a*(5+c) + b*(a+4), return {a: 2, b: 1, c: 1}."""

        def count(node, counter):
            if isinstance(node, Symbol):
                node = str(node)
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
