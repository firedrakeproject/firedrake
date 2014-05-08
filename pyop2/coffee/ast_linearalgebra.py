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

from collections import OrderedDict
from copy import deepcopy as dcopy

from pyop2.coffee.ast_base import *
import ast_plan


class AssemblyLinearAlgebra(object):

    """Convert assembly code into sequences of calls to external dense linear
    algebra libraries. Currently, MKL BLAS and ATLAS BLAS are supported."""

    def __init__(self, ao, kernel_decls):
        self.kernel_decls = kernel_decls
        self.header = ao.pre_header
        self.int_loop = ao.int_loop
        self.asm_expr = ao.asm_expr

    def blas(self, blas):
        """Transform perfect loop nests representing matrix-matrix multiplies into
        calls to BLAS dgemm. Involved matrices' layout is modified accordingly.

        :arg blas: the BLAS library that should be used (currently, only mkl)."""

        def update_syms(node, parent, syms_to_change, ofs_info, to_transpose):
            """Change the storage layout of symbols involved in MMMs."""
            if isinstance(node, Symbol):
                if node.symbol in syms_to_change:
                    if isinstance(parent, Decl):
                        node.rank = (int(node.rank[0])*int(node.rank[1]),)
                    else:
                        if node.symbol in to_transpose:
                            node.offset = ((ofs_info.values()[0], node.rank[0]),)
                            node.rank = (node.rank[-1],)
                        else:
                            node.offset = ((ofs_info[node.rank[-1]], node.rank[-1]),)
                            node.rank = (node.rank[0],)
            elif isinstance(node, (Par, For)):
                update_syms(node.children[0], node, syms_to_change, ofs_info, to_transpose)
            elif isinstance(node, Decl):
                update_syms(node.sym, node, syms_to_change, ofs_info, to_transpose)
            elif isinstance(node, (Assign, Incr)):
                update_syms(node.children[0], node, syms_to_change, ofs_info, to_transpose)
                update_syms(node.children[1], node, syms_to_change, ofs_info, to_transpose)
            elif isinstance(node, (Root, Block, Expr)):
                for n in node.children:
                    update_syms(n, node, syms_to_change, ofs_info, to_transpose)
            else:
                pass

        def check_prod(node):
            """Return (e1, e2) if the node is a product between two symbols s1
            and s2, () otherwise.
            For example:
            - Par(Par(Prod(s1, s2))) -> (s1, s2)
            - Prod(s1, s2) -> (s1, s2)
            - Sum -> ()
            - Prod(Sum, s1) -> ()"""
            if isinstance(node, Par):
                return check_prod(node.children[0])
            elif isinstance(node, Prod):
                left, right = (node.children[0], node.children[1])
                if isinstance(left, Expr) and isinstance(right, Expr):
                    return (left, right)
                return ()
            return ()

        # There must be at least three loops to extract a MMM
        if not (self.int_loop and self.asm_expr):
            return

        outer_loop = self.int_loop
        ofs = self.header.children.index(outer_loop)
        found_mmm = False

        # 1) Split potential MMM into different perfect loop nests
        to_remove, to_transpose = ([], [])
        to_transform = {}
        for middle_loop in outer_loop.children[0].children:
            if not isinstance(middle_loop, For):
                continue
            found = False
            inner_loop = middle_loop.children[0].children
            if not (len(inner_loop) == 1 and isinstance(inner_loop[0], For)):
                continue
            # Found a perfect loop nest, now check body operation
            body = inner_loop[0].children[0].children
            if not (len(body) == 1 and isinstance(body[0], Incr)):
                continue
            # The body is actually a single statement, as expected
            lhs = body[0].children[0].rank
            rhs = check_prod(body[0].children[1])
            if not rhs:
                continue
            # Check memory access pattern
            rhs_l, rhs_r = (rhs[0].rank, rhs[1].rank)
            if lhs[0] == rhs_l[0] and lhs[1] == rhs_r[1] and rhs_l[1] == rhs_r[0] or \
                    lhs[0] == rhs_r[1] and lhs[1] == rhs_r[0] and rhs_l[1] == rhs_r[0]:
                found = True
            elif lhs[0] == rhs_l[1] and lhs[1] == rhs_r[1] and rhs_l[0] == rhs_r[0] or \
                    lhs[0] == rhs_r[1] and lhs[1] == rhs_l[1] and rhs_l[0] == rhs_r[0]:
                found = True
                to_transpose.append(rhs[0].symbol)
            if found:
                new_outer = dcopy(outer_loop)
                new_outer.children[0].children = [middle_loop]
                to_remove.append(middle_loop)
                self.header.children.insert(ofs, new_outer)
                loop_itvars = (outer_loop.it_var(), middle_loop.it_var(), inner_loop[0].it_var())
                loop_sizes = (outer_loop.size(), middle_loop.size(), inner_loop[0].size())
                loop_info = OrderedDict(zip(loop_itvars, loop_sizes))
                to_transform[new_outer] = (body[0].children[0], rhs, loop_info)
                found_mmm = True
        # Clean up
        for l in to_remove:
            outer_loop.children[0].children.remove(l)
        if not outer_loop.children[0].children:
            self.header.children.remove(outer_loop)

        # 2) Delegate to BLAS
        to_change_layout = []
        for l, mmm in to_transform.items():
            lhs, rhs, loop_info = mmm
            blas_interface = ast_plan.blas_interface
            dgemm = blas_interface['dgemm'] % \
                {'m1size': loop_info[rhs[0].rank[-1]],
                 'm2size': loop_info[rhs[1].rank[-1]],
                 'm3size': loop_info[rhs[0].rank[0]],
                 'm1': rhs[0].symbol,
                 'm2': rhs[1].symbol,
                 'm3': lhs.symbol}
            self.header.children[self.header.children.index(l)] = FlatBlock(dgemm)
            to_change = [rhs[0].symbol, rhs[1].symbol, lhs.symbol]
            to_change_layout.extend([s for s in to_change if s not in to_change_layout])
        # Change the storage layout of involved matrices
        if to_change_layout:
            update_syms(self.header, None, to_change_layout, loop_info, to_transpose)
            update_syms(self.kernel_decls[lhs.symbol][0], None, to_change_layout,
                        loop_sizes, to_transpose)

        return found_mmm
