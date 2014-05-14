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
from copy import deepcopy as dcopy

from ast_base import *
import ast_plan as ap


class AssemblyVectorizer(object):

    """ Loop vectorizer """

    def __init__(self, assembly_optimizer, intrinsics, compiler):
        self.asm_opt = assembly_optimizer
        self.intr = intrinsics
        self.comp = compiler
        self.padded = []

    def alignment(self, decl_scope):
        """Align all data structures accessed in the loop nest to the size in
        bytes of the vector length."""

        for d, s in decl_scope.values():
            if d.sym.rank and s != ap.PARAM_VAR:
                d.attr.append(self.comp["align"](self.intr["alignment"]))

    def padding(self, decl_scope):
        """Pad all data structures accessed in the loop nest to the nearest
        multiple of the vector length. Adjust trip counts and bounds of all
        innermost loops where padded arrays are written to. Since padding
        enforces data alignment of multi-dimensional arrays, add suitable
        pragmas to inner loops to inform the backend compiler about this
        property."""

        used_syms = [s.symbol for s in self.asm_opt.sym]
        acc_decls = [d for s, d in decl_scope.items() if s in used_syms]

        # Padding
        for d, s in acc_decls:
            if d.sym.rank:
                if s == ap.PARAM_VAR:
                    d.sym.rank = tuple([vect_roundup(r) for r in d.sym.rank])
                else:
                    rounded = vect_roundup(d.sym.rank[-1])
                    d.sym.rank = d.sym.rank[:-1] + (rounded,)
                self.padded.append(d.sym)

        iloops = inner_loops(self.asm_opt.pre_header)
        # Add pragma alignment
        for l in iloops:
            l.pragma = self.comp["decl_aligned_for"]

        # Loop adjustment
        for l in iloops:
            for stm in l.children[0].children:
                sym = stm.children[0]
                if sym.rank and sym.rank[-1] == l.it_var():
                    bound = l.cond.children[1]
                    l.cond.children[1] = c_sym(vect_roundup(bound.symbol))

    def outer_product(self, opts, factor=1):
        """Compute outer products according to opts.
        opts = V_OP_PADONLY : no peeling, just use padding
        opts = V_OP_PEEL : peeling for autovectorisation
        opts = V_OP_UAJ : set unroll_and_jam factor
        opts = V_OP_UAJ_EXTRA : as above, but extra iters avoid remainder loop
        factor is an additional parameter to specify things like unroll-and-
        jam factor. Note that factor is just a suggestion to the compiler,
        which can freely decide to use a higher or lower value."""

        if not self.asm_opt.asm_expr:
            return

        for stmt, stmt_info in self.asm_opt.asm_expr.items():
            # First, find outer product loops in the nest
            it_vars, parent, loops = stmt_info

            vect_len = self.intr["dp_reg"]
            rows = loops[0].size()
            unroll_factor = factor if opts in [ap.V_OP_UAJ, ap.V_OP_UAJ_EXTRA] else 1

            op = OuterProduct(stmt, loops, self.intr, self.asm_opt)

            # Vectorisation
            rows_per_it = vect_len*unroll_factor
            if opts == ap.V_OP_UAJ:
                if rows_per_it <= rows:
                    body, layout = op.generate(rows_per_it)
                else:
                    # Unroll factor too big
                    body, layout = op.generate(vect_len)
            elif opts == ap.V_OP_UAJ_EXTRA:
                if rows <= rows_per_it or vect_roundup(rows) % rows_per_it > 0:
                    # Cannot unroll too much
                    body, layout = op.generate(vect_len)
                else:
                    body, layout = op.generate(rows_per_it)
            elif opts in [ap.V_OP_PADONLY, ap.V_OP_PEEL]:
                body, layout = op.generate(vect_len)
            else:
                raise RuntimeError("Don't know how to vectorize option %s" % opts)

            # Construct the remainder loop
            if opts != ap.V_OP_UAJ_EXTRA and rows > rows_per_it and rows % rows_per_it > 0:
                # peel out
                loop_peel = dcopy(loops)
                # Adjust main, layout and remainder loops bound and trip
                bound = loops[0].cond.children[1].symbol
                bound -= bound % rows_per_it
                loops[0].cond.children[1] = c_sym(bound)
                layout.cond.children[1] = c_sym(bound)
                loop_peel[0].init.init = c_sym(bound)
                loop_peel[0].incr.children[1] = c_sym(1)
                loop_peel[1].incr.children[1] = c_sym(1)
                # Append peeling loop after the main loop
                parent_loop = self.asm_opt.fors[0]
                parent_loop.children[0].children.append(loop_peel[0])

            # Insert the vectorized code at the right point in the loop nest
            blk = parent.children
            ofs = blk.index(stmt)
            parent.children = blk[:ofs] + body + blk[ofs + 1:]

        # Append the layout code after the loop nest
        if layout:
            parent = self.asm_opt.pre_header.children
            parent.insert(parent.index(self.asm_opt.fors[0]) + 1, layout)


class OuterProduct():

    """Generate outer product vectorisation of a statement. """

    OP_STORE_IN_MEM = 0
    OP_REGISTER_INC = 1

    def __init__(self, stmt, loops, intr, nest):
        self.stmt = stmt
        self.intr = intr
        # Outer product loops
        self.loops = loops
        # The whole loop nest in which outer product loops live
        self.nest = nest

    class Alloc(object):

        """Handle allocation of register variables. """

        def __init__(self, intr, tensor_size):
            nres = max(intr["dp_reg"], tensor_size)
            self.ntot = intr["avail_reg"]
            self.res = [intr["reg"](v) for v in range(nres)]
            self.var = [intr["reg"](v) for v in range(nres, self.ntot)]
            self.i = intr

        def get_reg(self):
            if len(self.var) == 0:
                l = self.ntot * 2
                self.var += [self.i["reg"](v) for v in range(self.ntot, l)]
                self.ntot = l
            return self.var.pop(0)

        def free_regs(self, regs):
            for r in reversed(regs):
                self.var.insert(0, r)

        def get_tensor(self):
            return self.res

    def _swap_reg(self, step, vrs):
        """Swap values in a vector register. """

        # Find inner variables
        regs = [reg for node, reg in vrs.items()
                if node.rank and node.rank[-1] == self.loops[1].it_var()]

        if step in [0, 2]:
            return [Assign(r, self.intr["l_perm"](r, "5")) for r in regs]
        elif step == 1:
            return [Assign(r, self.intr["g_perm"](r, r, "1")) for r in regs]
        elif step == 3:
            return []

    def _vect_mem(self, vrs, decls):
        """Return a list of vector variable declarations representing
        loads, sets, broadcasts.

        :arg vrs:   Dictionary that associates scalar variables to vector.
                    variables, for which it will be generated a corresponding
                    intrinsics load/set/broadcast.
        :arg decls: List of scalar variables for which an intrinsics load/
                    set/broadcast has already been generated. Used to avoid
                    regenerating the same line. Can be updated.
        """
        stmt = []
        for node, reg in vrs.items():
            if node.rank and node.rank[-1] in [i.it_var() for i in self.loops]:
                exp = self.intr["symbol_load"](node.symbol, node.rank, node.offset)
            else:
                exp = self.intr["symbol_set"](node.symbol, node.rank, node.offset)
            if not decls.get(node.gencode()):
                decls[node.gencode()] = reg
                stmt.append(Decl(self.intr["decl_var"], reg, exp))
        return stmt

    def _vect_expr(self, node, ofs, regs, decls, vrs):
        """Turn a scalar expression into its intrinsics equivalent.
        Also return dicts of allocated vector variables.

        :arg node:  AST Expression which is inspected to generate an equivalent
                    intrinsics-based representation.
        :arg ofs:   Contains the offset of the entry in the left hand side that
                    is being computed.
        :arg regs:  Register allocator.
        :arg decls: List of scalar variables for which an intrinsics load/
                    set/broadcast has already been generated. Used to determine
                    which vector variable contains a certain scalar, if any.
        :arg vrs:   Dictionary that associates scalar variables to vector
                    variables. Updated every time a new scalar variable is
                    encountered.
        """

        if isinstance(node, Symbol):
            if node.rank and self.loops[0].it_var() == node.rank[-1]:
                # The symbol depends on the outer loop dimension, so add offset
                n_ofs = tuple([(1, 0) for i in range(len(node.rank)-1)]) + ((1, ofs),)
                node = Symbol(node.symbol, dcopy(node.rank), n_ofs)
            node_ide = node.gencode()
            if node_ide not in decls:
                reg = [k for k in vrs.keys() if k.gencode() == node_ide]
                if not reg:
                    vrs[node] = c_sym(regs.get_reg())
                    return vrs[node]
                else:
                    return vrs[reg[0]]
            else:
                return decls[node_ide]
        elif isinstance(node, Par):
            return self._vect_expr(node.children[0], ofs, regs, decls, vrs)
        else:
            left = self._vect_expr(node.children[0], ofs, regs, decls, vrs)
            right = self._vect_expr(node.children[1], ofs, regs, decls, vrs)
            if isinstance(node, Sum):
                return self.intr["add"](left, right)
            elif isinstance(node, Sub):
                return self.intr["sub"](left, right)
            elif isinstance(node, Prod):
                return self.intr["mul"](left, right)
            elif isinstance(node, Div):
                return self.intr["div"](left, right)

    def _incr_tensor(self, tensor, ofs, regs, out_reg, mode):
        """Add the right hand side contained in out_reg to tensor.

        :arg tensor:  The left hand side of the expression being vectorized.
        :arg ofs:     Contains the offset of the entry in the left hand side that
                      is being computed.
        :arg regs:    Register allocator.
        :arg out_reg: Register variable containing the left hand side.
        :arg mode:    It can be either `OP_STORE_IN_MEM`, for which stores in
                      memory are performed, or `OP_REGISTER_INC`, by means of
                      which left hand side's values are accumulated in a register.
                      Usually, `OP_REGISTER_INC` is not recommended unless the
                      loop sizes are extremely small.
        """
        if mode == self.OP_STORE_IN_MEM:
            # Store in memory
            sym = tensor.symbol
            rank = tensor.rank
            ofs = ((1, ofs), (1, 0))
            load = self.intr["symbol_load"](sym, rank, ofs)
            return self.intr["store"](Symbol(sym, rank, ofs),
                                      self.intr["add"](load, out_reg))
        elif mode == self.OP_REGISTER_INC:
            # Accumulate on a vector register
            reg = Symbol(regs.get_tensor()[ofs], ())
            return Assign(reg, self.intr["add"](reg, out_reg))

    def _restore_layout(self, regs, tensor, mode):
        """Restore the storage layout of the tensor.

        :arg regs:    Register allocator.
        :arg tensor:  The left hand side of the expression being vectorized.
        :arg mode:    It can be either `OP_STORE_IN_MEM`, for which load/stores in
                      memory are performed, or `OP_REGISTER_INC`, by means of
                      which left hand side's values are read from registers.
        """

        code = []
        t_regs = [Symbol(r, ()) for r in regs.get_tensor()]
        n_regs = len(t_regs)

        # Determine tensor symbols
        tensor_syms = []
        for i in range(n_regs):
            rank = (tensor.rank[0] + "+" + str(i), tensor.rank[1])
            tensor_syms.append(Symbol(tensor.symbol, rank))

        # Load LHS values from memory
        if mode == self.OP_STORE_IN_MEM:
            for i, j in zip(tensor_syms, t_regs):
                load_sym = self.intr["symbol_load"](i.symbol, i.rank)
                code.append(Decl(self.intr["decl_var"], j, load_sym))

        # In-register restoration of the tensor
        # TODO: AVX only at the present moment
        # TODO: here some __m256 vars could not be declared if rows < 4
        perm = self.intr["g_perm"]
        uphi = self.intr["unpck_hi"]
        uplo = self.intr["unpck_lo"]
        typ = self.intr["decl_var"]
        vect_len = self.intr["dp_reg"]
        # Do as many times as the unroll factor
        spins = int(ceil(n_regs / float(vect_len)))
        for i in range(spins):
            # In-register permutations
            tmp = [Symbol(regs.get_reg(), ()) for r in range(vect_len)]
            code.append(Decl(typ, tmp[0], uphi(t_regs[1], t_regs[0])))
            code.append(Decl(typ, tmp[1], uplo(t_regs[0], t_regs[1])))
            code.append(Decl(typ, tmp[2], uphi(t_regs[2], t_regs[3])))
            code.append(Decl(typ, tmp[3], uplo(t_regs[3], t_regs[2])))
            code.append(Assign(t_regs[0], perm(tmp[1], tmp[3], 32)))
            code.append(Assign(t_regs[1], perm(tmp[0], tmp[2], 32)))
            code.append(Assign(t_regs[2], perm(tmp[3], tmp[1], 49)))
            code.append(Assign(t_regs[3], perm(tmp[2], tmp[0], 49)))
            regs.free_regs([s.symbol for s in tmp])

            # Store LHS values in memory
            for j in range(min(vect_len, n_regs - i * vect_len)):
                ofs = i * vect_len + j
                code.append(self.intr["store"](tensor_syms[ofs], t_regs[ofs]))

        return code

    def generate(self, rows):
        """Generate the outer-product intrinsics-based vectorisation code. """

        cols = self.intr["dp_reg"]

        # Determine order of loops w.r.t. the local tensor entries.
        # If j-k are the inner loops and A[j][k], then increments of
        # A are performed within the k loop, otherwise we would lose too many
        # vector registers for keeping tmp values. On the other hand, if i is
        # the innermost loop (i.e. loop nest is j-k-i), stores in memory are
        # done outside of ip, i.e. immediately before the outer product's
        # inner loop terminates.
        if self.loops[1].it_var() == self.nest.fors[-1].it_var():
            mode = self.OP_STORE_IN_MEM
            tensor_size = cols
        else:
            mode = self.OP_REGISTER_INC
            tensor_size = rows

        tensor = self.stmt.children[0]
        expr = self.stmt.children[1]

        # Get source-level variables
        regs = self.Alloc(self.intr, tensor_size)

        # Adjust loops' increment
        self.loops[0].incr.children[1] = c_sym(rows)
        self.loops[1].incr.children[1] = c_sym(cols)

        stmt = []
        decls = {}
        vrs = {}
        rows_per_col = rows / cols
        rows_to_peel = rows % cols
        peeling = 0
        for i in range(cols):
            # Handle extra rows
            if peeling < rows_to_peel:
                nrows = rows_per_col + 1
                peeling += 1
            else:
                nrows = rows_per_col
            for j in range(nrows):
                # Vectorize, declare allocated variables, increment tensor
                ofs = j * cols
                v_expr = self._vect_expr(expr, ofs, regs, decls, vrs)
                stmt.extend(self._vect_mem(vrs, decls))
                incr = self._incr_tensor(tensor, i + ofs, regs, v_expr, mode)
                stmt.append(incr)
            # Register shuffles
            if rows_per_col + (rows_to_peel - peeling) > 0:
                stmt.extend(self._swap_reg(i, vrs))

        # Set initialising and tensor layout code
        layout = self._restore_layout(regs, tensor, mode)
        if mode == self.OP_STORE_IN_MEM:
            # Tensor layout
            layout_loops = dcopy(self.loops)
            layout_loops[0].incr.children[1] = c_sym(cols)
            layout_loops[0].children = [Block([layout_loops[1]], open_scope=True)]
            layout_loops[1].children = [Block(layout, open_scope=True)]
            layout = layout_loops[0]
        elif mode == self.OP_REGISTER_INC:
            # Initialiser
            for r in regs.get_tensor():
                decl = Decl(self.intr["decl_var"], Symbol(r, ()), self.intr["setzero"])
                self.loops[1].children[0].children.insert(0, decl)
            # Tensor layout
            self.loops[1].children[0].children.extend(layout)
            layout = None

        return (stmt, layout)


# Utility functions

def vect_roundup(x):
    """Return x rounded up to the vector length. """
    word_len = ap.intrinsics.get("dp_reg") or 1
    return int(ceil(x / float(word_len))) * word_len


def inner_loops(node):
    """Find inner loops in the subtree rooted in node."""

    def find_iloops(node, loops):
        if isinstance(node, Perfect):
            return False
        elif isinstance(node, (Block, Root)):
            return any([find_iloops(s, loops) for s in node.children])
        elif isinstance(node, For):
            found = find_iloops(node.children[0], loops)
            if not found:
                loops.append(node)
            return True

    loops = []
    find_iloops(node, loops)
    return loops
