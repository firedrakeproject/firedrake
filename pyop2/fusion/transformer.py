# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2016, Imperial College London and
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

"""Core loop fusion mechanisms."""

from copy import deepcopy as dcopy

import pyop2.base as base
from pyop2.caching import Cached
from pyop2.profiling import timed_region

from extended import lazy_trace_name, Kernel
from filters import Filter
from scheduler import *

from coffee import base as ast
from coffee.utils import ItSpace
from coffee.visitors import FindInstances, SymbolReferences


class Inspector(Cached):

    """An Inspector constructs a Schedule to fuse or tile a sequence of loops.

    .. note:: For tiling, the Inspector relies on the SLOPE library."""

    _cache = {}
    _modes = ['soft', 'hard', 'tile', 'only_tile', 'only_omp']

    @classmethod
    def _cache_key(cls, name, loop_chain, **options):
        key = (name,)
        if name != lazy_trace_name:
            # Special case: the Inspector comes from a user-defined /loop_chain/
            key += (options['mode'], options['tile_size'],
                    options['use_glb_maps'], options['use_prefetch'], options['coloring'])
            key += (loop_chain[0].kernel.cache_key,)
            return key
        # Inspector extracted from lazy evaluation trace
        for loop in loop_chain:
            if isinstance(loop, base._LazyMatOp):
                continue
            key += (loop.kernel.cache_key,)
            key += (loop.it_space.cache_key, loop.it_space.iterset.sizes)
            for arg in loop.args:
                if arg._is_global:
                    key += (arg.data.dim, arg.data.dtype, arg.access)
                elif arg._is_dat:
                    if isinstance(arg.idx, base.IterationIndex):
                        idx = (arg.idx.__class__, arg.idx.index)
                    else:
                        idx = arg.idx
                    map_arity = arg.map.arity if arg.map else None
                    key += (arg.data.dim, arg.data.dtype, map_arity, idx, arg.access)
                elif arg._is_mat:
                    idxs = (arg.idx[0].__class__, arg.idx[0].index,
                            arg.idx[1].index)
                    map_arities = (arg.map[0].arity, arg.map[1].arity)
                    key += (arg.data.dims, arg.data.dtype, idxs, map_arities, arg.access)
        return key

    def __init__(self, name, loop_chain, **options):
        """Initialize an Inspector object.

        :arg name: a name for the Inspector
        :arg loop_chain: an iterator for the loops that will be fused/tiled
        :arg options: a set of parameters to drive fusion/tiling
            * mode: can take any of the values in ``Inspector._modes``, namely
                soft, hard, tile, only_tile, only_omp:
                * soft: consecutive loops over the same iteration set that do
                    not present RAW or WAR dependencies through indirections
                    are fused.
                * hard: ``soft`` fusion; then, loops over different iteration sets
                    are also fused, provided that there are no RAW or WAR
                    dependencies.
                * tile: ``soft`` and ``hard`` fusion; then, tiling through the
                    SLOPE library takes place.
                * only_tile: only tiling through the SLOPE library (i.e., no fusion)
                * only_omp: ompize individual parloops through the SLOPE library
            * tile_size: starting average tile size
            * extra_halo: are we providing SLOPE with extra halo to be efficient
                and allow it to minimize redundant computation ?
        """
        if self._initialized:
            return
        self._name = name
        self._loop_chain = loop_chain
        self._mode = options.pop('mode')
        self._options = options
        self._schedule = PlainSchedule(name, [loop.kernel for loop in self._loop_chain])

    def inspect(self):
        """Inspect the loop chain and produce a :class:`Schedule`."""
        if self._initialized:
            # An inspection plan is in cache.
            return self._schedule
        elif self._heuristic_skip_inspection():
            # Not in cache, and too premature for running a potentially costly inspection
            del self._name
            del self._loop_chain
            del self._mode
            del self._options
            return self._schedule

        # Is `mode` legal ?
        if self.mode not in Inspector._modes:
            raise RuntimeError("Inspection accepts only %s fusion modes", Inspector._modes)

        with timed_region("ParLoopChain `%s`: inspector" % self._name):
            if self.mode in ['soft', 'hard', 'tile']:
                self._soft_fuse()
            if self.mode in ['hard', 'tile']:
                self._hard_fuse()
            if self.mode in ['tile', 'only_tile', 'only_omp']:
                self._tile()

        # A schedule has been computed. The Inspector is initialized and therefore
        # retrievable from cache. We then blow away everything we don't need any more.
        self._initialized = True
        del self._name
        del self._loop_chain
        del self._mode
        del self._options
        return self._schedule

    def _heuristic_skip_inspection(self):
        """Decide, heuristically, whether to run an inspection or not.
        If tiling is not requested, then inspection is performed.
        If tiling is requested, then inspection is performed on the third
        invocation. The fact that an inspection for the same loop chain
        is requested multiple times suggests the parloops originate in a
        time stepping loop. The cost of building tiles in SLOPE-land would
        then be amortized over several iterations."""
        self._ninsps = self._ninsps + 1 if hasattr(self, '_ninsps') else 1
        if self.mode in ['tile', 'only_tile'] and self._ninsps < 3:
            return True
        return False

    def _soft_fuse(self):
        """Fuse consecutive loops over the same iteration set by concatenating
        kernel bodies and creating new :class:`ParLoop` objects representing
        the fused sequence.

        The conditions under which two loops over the same iteration set can
        be soft fused are:

            * They are both direct, OR
            * One is direct and the other indirect

        This is detailed in the paper::

            "Mesh Independent Loop Fusion for Unstructured Mesh Applications"

        from C. Bertolli et al.
        """

        def fuse(self, loops, loop_chain_index):
            # Naming convention: here, we are fusing ASTs in /fuse_asts/ within
            # /base_ast/. Same convention will be used in the /hard_fuse/ method
            kernels = [l.kernel for l in loops]
            fuse_asts = [k._original_ast if k._code else k._ast for k in kernels]
            # Fuse the actual kernels' bodies
            base_ast = dcopy(fuse_asts[0])
            retval = FindInstances.default_retval()
            base_fundecl = FindInstances(ast.FunDecl).visit(base_ast, ret=retval)[ast.FunDecl]
            if len(base_fundecl) != 1:
                raise RuntimeError("Fusing kernels, but found unexpected AST")
            base_fundecl = base_fundecl[0]
            for unique_id, _fuse_ast in enumerate(fuse_asts[1:], 1):
                fuse_ast = dcopy(_fuse_ast)
                retval = FindInstances.default_retval()
                fuse_fundecl = FindInstances(ast.FunDecl).visit(fuse_ast, ret=retval)[ast.FunDecl]
                if len(fuse_fundecl) != 1:
                    raise RuntimeError("Fusing kernels, but found unexpected AST")
                fuse_fundecl = fuse_fundecl[0]
                # 1) Extend function name
                base_fundecl.name = "%s_%s" % (base_fundecl.name, fuse_fundecl.name)
                # 2) Concatenate the arguments in the signature
                base_fundecl.args.extend(fuse_fundecl.args)
                # 3) Uniquify symbols identifiers
                retval = SymbolReferences.default_retval()
                fuse_symbols = SymbolReferences().visit(fuse_ast, ret=retval)
                for decl in fuse_fundecl.args:
                    for symbol, _ in fuse_symbols[decl.sym.symbol]:
                        symbol.symbol = "%s_%d" % (symbol.symbol, unique_id)
                # 4) Scope and concatenate bodies
                base_fundecl.children[0] = ast.Block(
                    [ast.Block(base_fundecl.children[0].children, open_scope=True),
                     ast.FlatBlock("\n\n// Begin of fused kernel\n\n"),
                     ast.Block(fuse_fundecl.children[0].children, open_scope=True)])
            # Eliminate redundancies in the /fused/ kernel signature
            Filter().kernel_args(loops, base_fundecl)
            # Naming convention
            fused_ast = base_ast
            return Kernel(kernels, fused_ast, loop_chain_index)

        fused, fusing = [], [self._loop_chain[0]]
        for i, loop in enumerate(self._loop_chain[1:]):
            base_loop = fusing[-1]
            if base_loop.it_space != loop.it_space or \
                    (base_loop.is_indirect and loop.is_indirect):
                # Fusion not legal
                fused.append((fuse(self, fusing, len(fused)), i+1))
                fusing = [loop]
            elif (base_loop.is_direct and loop.is_direct) or \
                    (base_loop.is_direct and loop.is_indirect) or \
                    (base_loop.is_indirect and loop.is_direct):
                # This loop is fusible. Also, can speculative go on searching
                # for other loops to fuse
                fusing.append(loop)
            else:
                raise RuntimeError("Unexpected loop chain structure while fusing")
        if fusing:
            fused.append((fuse(self, fusing, len(fused)), len(self._loop_chain)))

        fused_kernels, offsets = zip(*fused)
        self._schedule = FusionSchedule(self._name, self._schedule, fused_kernels, offsets)
        self._loop_chain = self._schedule(self._loop_chain)

    def _hard_fuse(self):
        """Fuse consecutive loops over different iteration sets that do not
        present RAW, WAR or WAW dependencies. For examples, two loops like: ::

            par_loop(kernel_1, it_space_1,
                     dat_1_1(INC, ...),
                     dat_1_2(READ, ...),
                     ...)

            par_loop(kernel_2, it_space_2,
                     dat_2_1(INC, ...),
                     dat_2_2(READ, ...),
                     ...)

        where ``dat_1_1 == dat_2_1`` and, possibly (but not necessarily),
        ``it_space_1 != it_space_2``, can be hard fused. Note, in fact, that
        the presence of ``INC`` does not imply a real WAR dependency, because
        increments are associative."""

        reads = lambda l: set([a.data for a in l.args if a.access in [READ, RW]])
        writes = lambda l: set([a.data for a in l.args if a.access in [RW, WRITE, MIN, MAX]])
        incs = lambda l: set([a.data for a in l.args if a.access in [INC]])

        def has_raw_or_war(loop1, loop2):
            # Note that INC after WRITE is a special case of RAW dependency since
            # INC cannot take place before WRITE.
            return reads(loop2) & writes(loop1) or writes(loop2) & reads(loop1) or \
                incs(loop1) & (writes(loop2) - incs(loop2)) or \
                incs(loop2) & (writes(loop1) - incs(loop1))

        def has_iai(loop1, loop2):
            return incs(loop1) & incs(loop2)

        def fuse(base_loop, loop_chain, fused):
            """Try to fuse one of the loops in ``loop_chain`` with ``base_loop``."""
            for loop in loop_chain:
                if has_raw_or_war(loop, base_loop):
                    # Can't fuse across loops preseting RAW or WAR dependencies
                    return []
                if loop.it_space == base_loop.it_space:
                    warning("Ignoring unexpected sequence of loops in loop fusion")
                    continue
                # Is there an overlap in any of the incremented regions? If that is
                # the case, then fusion can really be beneficial
                common_inc_data = has_iai(base_loop, loop)
                if not common_inc_data:
                    continue
                common_incs = [a for a in base_loop.args + loop.args
                               if a.data in common_inc_data]
                # Hard fusion potentially doable provided that we own a map between
                # the iteration spaces involved
                maps = list(set(flatten([a.map for a in common_incs])))
                maps += [m.factors for m in maps if hasattr(m, 'factors')]
                maps = list(flatten(maps))
                set1, set2 = base_loop.it_space.iterset, loop.it_space.iterset
                fused_map = [m for m in maps if set1 == m.iterset and set2 == m.toset]
                if fused_map:
                    fused.append((base_loop, loop, fused_map[0], common_incs[1]))
                    return loop_chain[:loop_chain.index(loop)+1]
                fused_map = [m for m in maps if set1 == m.toset and set2 == m.iterset]
                if fused_map:
                    fused.append((loop, base_loop, fused_map[0], common_incs[0]))
                    return loop_chain[:loop_chain.index(loop)+1]
            return []

        # First, find fusible kernels
        fusible, skip = [], []
        for i, l in enumerate(self._loop_chain, 1):
            if l in skip:
                # /l/ occurs between (hard) fusible loops, let's leave it where
                # it is for safeness
                continue
            skip = fuse(l, self._loop_chain[i:], fusible)
        if not fusible:
            return

        # Then, create a suitable hard-fusion kernel
        # The hard fused kernel will have the following structure:
        #
        # wrapper (args: Union(kernel1, kernel2, extra):
        #   staging of pointers
        #   ...
        #   fusion (staged pointers, ..., extra)
        #   insertion (...)
        #
        # Where /extra/ represents additional arguments, like the map from
        # /kernel1/ iteration space to /kernel2/ iteration space. The /fusion/
        # function looks like:
        #
        # fusion (...):
        #   kernel1 (buffer, ...)
        #   for i = 0 to arity:
        #     if not already_executed[i]:
        #       kernel2 (buffer[..], ...)
        #
        # Where /arity/ is the number of /kernel2/ iterations incident to
        # /kernel1/ iterations.
        fused = []
        for base_loop, fuse_loop, fused_map, fused_inc_arg in fusible:
            # Start with analyzing the kernel ASTs. Note: fusion occurs on fresh
            # copies of the /base/ and /fuse/ ASTs. This is because the optimization
            # of the /fused/ AST should be independent of that of individual ASTs,
            # and subsequent cache hits for non-fused ParLoops should always retrive
            # the original, unmodified ASTs. This is important not just for the
            # sake of performance, but also for correctness of padding, since hard
            # fusion changes the signature of /fuse/ (in particular, the buffers that
            # are provided for computation on iteration spaces)
            finder = FindInstances((ast.FunDecl, ast.PreprocessNode))
            base, fuse = base_loop.kernel, fuse_loop.kernel
            base_ast = dcopy(base._original_ast) if base._code else dcopy(base._ast)
            retval = FindInstances.default_retval()
            base_info = finder.visit(base_ast, ret=retval)
            base_headers = base_info[ast.PreprocessNode]
            base_fundecl = base_info[ast.FunDecl]
            fuse_ast = dcopy(fuse._original_ast) if fuse._code else dcopy(fuse._ast)
            retval = FindInstances.default_retval()
            fuse_info = finder.visit(fuse_ast, ret=retval)
            fuse_headers = fuse_info[ast.PreprocessNode]
            fuse_fundecl = fuse_info[ast.FunDecl]
            if len(base_fundecl) != 1 or len(fuse_fundecl) != 1:
                raise RuntimeError("Fusing kernels, but found unexpected AST")
            base_fundecl = base_fundecl[0]
            fuse_fundecl = fuse_fundecl[0]

            # Create /fusion/ arguments and signature
            body = ast.Block([])
            fusion_name = '%s_%s' % (base_fundecl.name, fuse_fundecl.name)
            fusion_args = dcopy(base_fundecl.args + fuse_fundecl.args)
            fusion_fundecl = ast.FunDecl(base_fundecl.ret, fusion_name, fusion_args, body)

            # Filter out duplicate arguments, and append extra arguments to the fundecl
            binding = WeakFilter().kernel_args([base_loop, fuse_loop], fusion_fundecl)
            fusion_fundecl.args += [ast.Decl('int*', 'executed'),
                                    ast.Decl('int*', 'fused_iters'),
                                    ast.Decl('int', 'i')]

            # Which args are actually used in /fuse/, but not in /base/ ?
            # The gather for such arguments is moved to /fusion/, to avoid any
            # usless LOAD from memory
            retval = SymbolReferences.default_retval()
            base_symbols = SymbolReferences().visit(base_fundecl.body, ret=retval)
            retval = SymbolReferences.default_retval()
            fuse_symbols = SymbolReferences().visit(fuse_fundecl.body, ret=retval)
            base_funcall_syms, unshared = [], OrderedDict()
            for arg, decl in binding.items():
                if decl.sym.symbol in set(fuse_symbols) - set(base_symbols):
                    base_funcall_sym = ast.Symbol('NULL')
                    unshared.setdefault(decl, arg)
                else:
                    base_funcall_sym = ast.Symbol(decl.sym.symbol)
                if arg in base_loop.args:
                    base_funcall_syms.append(base_funcall_sym)
            for decl, arg in unshared.items():
                decl.typ = 'double*'
                decl.sym.symbol = arg.c_arg_name()
                fusion_fundecl.args.insert(fusion_fundecl.args.index(decl) + 1,
                                           ast.Decl('int*', arg.c_map_name(0, 0)))

            # Append the invocation of /base/; then, proceed with the invocation
            # of the /fuse/ kernels
            body.children.append(ast.FunCall(base_fundecl.name, *base_funcall_syms))

            for idx in range(fused_map.arity):

                fused_iter = 'fused_iters[%d]' % idx
                fuse_funcall = ast.FunCall(fuse_fundecl.name)
                if_cond = ast.Not(ast.Symbol('executed', (fused_iter,)))
                if_update = ast.Assign(ast.Symbol('executed', (fused_iter,)), 1)
                if_body = ast.Block([fuse_funcall, if_update], open_scope=True)
                if_exec = ast.If(if_cond, [if_body])
                body.children.extend([ast.FlatBlock('\n'), if_exec])

                # Modify the /fuse/ kernel
                # This is to take into account that many arguments are shared with
                # /base/, so they will only staged once for /base/. This requires
                # tweaking the way the arguments are declared and accessed in /fuse/.
                # For example, the shared incremented array (called /buffer/ in
                # the pseudocode in the comment above) now needs to take offsets
                # to be sure the locations that /base/ is supposed to increment are
                # actually accessed. The same concept apply to indirect arguments.
                init = lambda v: '{%s}' % ', '.join([str(j) for j in v])
                for i, fuse_loop_arg in enumerate(fuse_loop.args):
                    fuse_kernel_arg = binding[fuse_loop_arg]
                    buffer = '%s_vec' % fuse_kernel_arg.sym.symbol

                    # How should I use the temporaries ?
                    if fuse_loop_arg.access == INC:
                        op = ast.Incr
                        lvalue, rvalue = fuse_kernel_arg.sym.symbol, buffer
                        extend_if_body = lambda body, block: body.children.extend(block)
                        buffer_decl = ast.Decl('%s' % fuse_loop_arg.ctype, buffer)
                    elif fuse_loop_arg.access == READ:
                        op = ast.Assign
                        lvalue, rvalue = buffer, fuse_kernel_arg.sym.symbol
                        extend_if_body = lambda body, block: \
                            [body.children.insert(0, b) for b in reversed(block)]
                        buffer_decl = ast.Decl('%s*' % fuse_loop_arg.ctype, buffer)

                    # Now handle arguments depending on their type ...
                    if fuse_loop_arg._is_mat:
                        # ... Handle Mats
                        staging = []
                        for b in fused_inc_arg._block_shape:
                            for rc in b:
                                lvalue = ast.Symbol(lvalue, (idx, idx),
                                                    ((rc[0], 'j'), (rc[1], 'k')))
                                rvalue = ast.Symbol(rvalue, ('j', 'k'))
                                staging = ItSpace(mode=0).to_for([(0, rc[0]), (0, rc[1])],
                                                                 ('j', 'k'),
                                                                 [op(lvalue, rvalue)])[:1]
                        # Set up the temporary
                        buffer_decl.sym.rank = fuse_kernel_arg.sym.rank
                        if fuse_loop_arg.access == INC:
                            buffer_decl.init = ast.ArrayInit(init([init([0.0])]))

                    elif fuse_loop_arg._is_indirect:
                        # ... Handle indirect arguments. At the C level, these arguments
                        # are of pointer type, so simple pointer arithmetic is used
                        # to ensure the kernel accesses are to the correct locations
                        fuse_arity = fuse_loop_arg.map.arity
                        base_arity = fuse_arity*fused_map.arity
                        cdim = fuse_loop_arg.data.dataset.cdim
                        size = fuse_arity*cdim
                        # Set the proper storage layout before invoking /fuse/
                        ofs_vals = [[base_arity*j + k for k in range(fuse_arity)]
                                    for j in range(cdim)]
                        ofs_vals = [[fuse_arity*j + k for k in flatten(ofs_vals)]
                                    for j in range(fused_map.arity)]
                        ofs_vals = list(flatten(ofs_vals))
                        indices = [ofs_vals[idx*size + j] for j in range(size)]
                        # Set up the temporary and stage (gather) data into it
                        buffer_decl.sym.rank = (size,)
                        if fuse_loop_arg.access == INC:
                            buffer_decl.init = ast.ArrayInit(init([0.0]))
                            staging = [op(ast.Symbol(lvalue, (k,)), ast.Symbol(rvalue, (j,)))
                                       for j, k in enumerate(indices)]
                        elif fuse_kernel_arg in unshared:
                            staging = unshared[fuse_kernel_arg].c_vec_init(False).split('\n')
                            staging = [j for i, j in enumerate(staging) if i in indices]
                            rvalues = [ast.FlatBlock(i.split('=')[1]) for i in staging]
                            lvalues = [ast.Symbol(buffer, (i,)) for i in range(len(staging))]
                            staging = [ast.Assign(i, j) for i, j in zip(lvalues, rvalues)]
                        else:
                            staging = [op(ast.Symbol(lvalue, (j,)), ast.Symbol(rvalue, (k,)))
                                       for j, k in enumerate(indices)]

                    else:
                        # Nothing special to do for direct arguments
                        continue

                    # Update the If-then AST body
                    extend_if_body(if_exec.children[0], staging)
                    if_exec.children[0].children.insert(0, buffer_decl)
                    fuse_funcall.children.append(ast.Symbol(buffer))

            # Create a /fusion.Kernel/ object as well as the schedule
            fused_headers = set([str(h) for h in base_headers + fuse_headers])
            fused_ast = ast.Root([ast.PreprocessNode(h) for h in fused_headers] +
                                 [base_fundecl, fuse_fundecl, fusion_fundecl])
            kernels = [base, fuse]
            loop_chain_index = (self._loop_chain.index(base_loop),
                                self._loop_chain.index(fuse_loop))
            # Track position of Args that need a postponed gather
            # Can't track Args themselves as they change across different parloops
            fargs = {fusion_args.index(i): ('postponed', False) for i in unshared.keys()}
            fargs.update({len(set(binding.values())): ('onlymap', True)})
            fused.append((Kernel(kernels, fused_ast, loop_chain_index), fused_map, fargs))

        # Finally, generate a new schedule
        self._schedule = HardFusionSchedule(self._name, self._schedule, fused)
        self._loop_chain = self._schedule(self._loop_chain, only_hard=True)

    def _tile(self):
        """Tile consecutive loops over different iteration sets characterized
        by RAW and WAR dependencies. This requires interfacing with the SLOPE
        library."""

        def inspect_set(s, insp_sets, extra_halo):
            """Inspect the iteration set of a loop and store set info suitable
            for SLOPE in /insp_sets/. Further, check that such iteration set has
            a sufficiently depth halo region for correct execution in the case a
            SLOPE MPI backend is enabled."""
            # Get and format some iterset info
            partitioning, superset, s_name = None, None, s.name
            if isinstance(s, Subset):
                superset = s.superset.name
                s_name = "%s_ss" % s.name
            if hasattr(s, '_partitioning'):
                partitioning = s._partitioning
            # If not an MPI backend, return "standard" values for core, exec, and
            # non-exec regions (recall that SLOPE expects owned to be part of exec)
            if slope.get_exec_mode() not in ['OMP_MPI', 'ONLY_MPI']:
                exec_size = s.exec_size - s.core_size
                nonexec_size = s.total_size - s.exec_size
                infoset = s_name, s.core_size, exec_size, nonexec_size, superset
            else:
                if not hasattr(s, '_deep_size'):
                    raise RuntimeError("SLOPE backend (%s) requires deep halos",
                                       slope.get_exec_mode())
                # Assume [1, ..., N] levels of halo depth
                level_N = s._deep_size[-1]
                core_size = level_N[0]
                exec_size = level_N[2] - core_size
                nonexec_size = level_N[3] - level_N[2]
                if extra_halo and nonexec_size == 0:
                    level_E = s._deep_size[-2]
                    exec_size = level_E[2] - core_size
                    nonexec_size = level_E[3] - level_E[2]
                infoset = s_name, core_size, exec_size, nonexec_size, superset
            insp_sets[infoset] = partitioning
            return infoset

        tile_size = self._options.get('tile_size', 1)
        extra_halo = self._options.get('extra_halo', False)
        coloring = self._options.get('coloring', 'default')
        use_prefetch = self._options.get('use_prefetch', 0)
        log = self._options.get('log', False)
        rank = MPI.comm.rank

        # The SLOPE inspector, which needs be populated with sets, maps,
        # descriptors, and loop chain structure
        inspector = slope.Inspector(self._name)

        # Build inspector and argument types and values
        # Note: we need ordered containers to be sure that SLOPE generates
        # identical code for all ranks
        arguments = []
        insp_sets, insp_maps, insp_loops = OrderedDict(), OrderedDict(), []
        for loop in self._loop_chain:
            slope_desc = set()
            # 1) Add sets
            iterset = loop.it_space.iterset
            iterset = iterset.subset if hasattr(iterset, 'subset') else iterset
            infoset = inspect_set(iterset, insp_sets, extra_halo)
            iterset_name, is_superset = infoset[0], infoset[4]
            # If iterating over a subset, we fake an indirect parloop from the
            # (iteration) subset to the superset. This allows the propagation of
            # tiling across the hierarchy of sets (see SLOPE for further info)
            if is_superset:
                inspect_set(iterset.superset, insp_sets, extra_halo)
                map_name = "%s_tosuperset" % iterset_name
                insp_maps[iterset_name] = (map_name, iterset_name,
                                           iterset.superset.name, iterset.indices)
                slope_desc.add((map_name, INC._mode))
            for a in loop.args:
                # 2) Add access descriptors
                maps = as_tuple(a.map, Map)
                if not maps:
                    # Simplest case: direct loop
                    slope_desc.add(('DIRECT', a.access._mode))
                else:
                    # Add maps (there can be more than one per argument if the arg
                    # is actually a Mat - in which case there are two maps - or if
                    # a MixedMap) and relative descriptors
                    for i, map in enumerate(maps):
                        for j, m in enumerate(map):
                            map_name = "%s%d_%d" % (m.name, i, j)
                            insp_maps[m.name] = (map_name, m.iterset.name,
                                                 m.toset.name, m.values_with_halo)
                            slope_desc.add((map_name, a.access._mode))
                            inspect_set(m.iterset, insp_sets, extra_halo)
                            inspect_set(m.toset, insp_sets, extra_halo)
            # 3) Add loop
            insp_loops.append((loop.kernel.name, iterset_name, list(slope_desc)))
        # Provide structure of loop chain to SLOPE
        arguments.extend([inspector.add_sets(insp_sets.keys())])
        arguments.extend([inspector.add_maps(insp_maps.values())])
        inspector.add_loops(insp_loops)

        # Set a specific tile size
        arguments.extend([inspector.set_tile_size(tile_size)])

        # Tell SLOPE the rank of the MPI process
        arguments.extend([inspector.set_mpi_rank(rank)])

        # Get type and value of additional arguments that SLOPE can exploit
        arguments.extend(inspector.add_extra_info())

        # Add any available partitioning
        partitionings = [(s[0], v) for s, v in insp_sets.items() if v is not None]
        arguments.extend([inspector.add_partitionings(partitionings)])

        # Arguments types and values
        argtypes, argvalues = zip(*arguments)

        # Set a tile partitioning strategy
        inspector.set_part_mode('chunk')

        # Set a tile coloring strategy
        inspector.set_coloring(coloring)

        # Inform about the prefetch distance that needs be guaranteed
        inspector.set_prefetch_halo(use_prefetch)

        # Generate the C code
        src = inspector.generate_code()

        # Return type of the inspector
        rettype = slope.Executor.meta['py_ctype_exec']

        # Compiler and linker options
        slope_dir = os.environ['SLOPE_DIR']
        compiler = coffee.plan.compiler.get('name')
        cppargs = slope.get_compile_opts(compiler)
        cppargs += ['-I%s/%s' % (slope_dir, slope.get_include_dir())]
        ldargs = ['-L%s/%s' % (slope_dir, slope.get_lib_dir()),
                  '-l%s' % slope.get_lib_name(),
                  '-lrt']

        # Compile and run inspector
        fun = compilation.load(src, "cpp", "inspector", cppargs, ldargs,
                               argtypes, rettype, compiler)
        inspection = fun(*argvalues)

        # Log the inspector output
        if log and rank == 0:
            filename = os.path.join("log", "%s.txt" % self._name)
            summary = os.path.join("log", "summary.txt")
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'w') as f, open(summary, 'a') as s:
                # Estimate tile footprint
                template = '| %25s | %22s | %-11s |\n'
                f.write('*** Tile footprint ***\n')
                f.write(template % ('iteration set', 'memory footprint (KB)', 'megaflops'))
                f.write('-' * 68 + '\n')
                tot_footprint, tot_flops = 0, 0
                for loop in self._loop_chain:
                    flops, footprint = loop.num_flops/(1000*1000), 0
                    for arg in loop.args:
                        dat_size = arg.data.nbytes
                        map_size = 0 if arg._is_direct else arg.map.values_with_halo.nbytes
                        tot_dat_size = (dat_size + map_size)/1000
                        footprint += tot_dat_size
                    tot_footprint += footprint
                    f.write(template % (loop.it_space.name, str(footprint), str(flops)))
                    tot_flops += flops
                f.write('** Summary: %d KBytes moved, %d Megaflops performed\n' %
                        (tot_footprint, tot_flops))
                probSeed = 0 if MPI.parallel else len(self._loop_chain) / 2
                probNtiles = self._loop_chain[probSeed].it_space.exec_size / tile_size or 1
                f.write('** KB/tile: %d' % (tot_footprint/probNtiles))
                f.write('  (Estimated: %d tiles)\n' % probNtiles)
                f.write('-' * 68 + '\n')

                # Estimate data reuse
                template = '| %40s | %5s | %-70s |\n'
                f.write('*** Data reuse ***\n')
                f.write(template % ('field', 'type', 'loops'))
                f.write('-' * 125 + '\n')
                reuse = OrderedDict()
                for i, loop in enumerate(self._loop_chain):
                    for arg in loop.args:
                        values = reuse.setdefault(arg.data, [])
                        if i not in values:
                            values.append(i)
                        if arg._is_indirect:
                            values = reuse.setdefault(arg.map, [])
                            if i not in values:
                                values.append(i)
                for field, positions in reuse.items():
                    reused_in = ', '.join('%d' % j for j in positions)
                    field_type = 'map' if isinstance(field, Map) else 'data'
                    f.write(template % (field.name, field_type, reused_in))
                ideal_reuse = 0
                for field, positions in reuse.items():
                    size = field.values_with_halo.nbytes if isinstance(field, Map) \
                        else field.nbytes
                    # First position needs be cut away as it's the first touch
                    ideal_reuse += (size/1000)*len(positions[1:])
                out = '** Ideal reuse (i.e., no tile growth): %d / %d KBytes (%f %%)\n' % \
                    (ideal_reuse, tot_footprint, float(ideal_reuse)*100/tot_footprint)
                f.write(out)
                f.write('-' * 125 + '\n')
                s.write(out)

        # Finally, get the Executor representation, to be used at executor
        # code generation time
        executor = slope.Executor(inspector)

        kernel = Kernel(tuple(loop.kernel for loop in self._loop_chain))
        self._schedule = TilingSchedule(self._name, self._schedule, kernel, inspection,
                                        executor, **self._options)

    @property
    def mode(self):
        return self._mode

    @property
    def schedule(self):
        return self._schedule
