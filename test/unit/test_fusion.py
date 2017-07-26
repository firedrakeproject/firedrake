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


import pytest
import numpy as np
import random
from contextlib import contextmanager

from pyop2 import op2
from pyop2.base import _trace as trace
from pyop2 import configuration
import pyop2.fusion.interface
from pyop2.fusion.interface import fuse, lazy_trace_name, loop_chain, slope


from coffee import base as ast
from coffee.utils import ItSpace

nelems = 100


@pytest.fixture
def iterset():
    return op2.Set(nelems, "iterset")


@pytest.fixture
def bigiterset():
    return op2.Set(2*nelems, "bigiterset")


@pytest.fixture
def indset():
    return op2.Set(nelems, "indset")


@pytest.fixture
def diterset(iterset):
    return op2.DataSet(iterset, 1, "diterset")


@pytest.fixture
def x(iterset):
    return op2.Dat(iterset, list(range(nelems)), np.uint32, "x")


@pytest.fixture
def y(iterset):
    return op2.Dat(iterset, list(range(nelems)), np.uint32, "y")


@pytest.fixture
def z(iterset):
    return op2.Dat(iterset, list(range(nelems)), np.uint32, "z")


@pytest.fixture
def ix(indset):
    return op2.Dat(indset, list(range(nelems)), np.uint32, "ix")


@pytest.fixture
def iy(indset):
    return op2.Dat(indset, list(range(nelems)), np.uint32, "iy")


@pytest.fixture
def x2(iterset):
    return op2.Dat(iterset ** 2, np.array([list(range(nelems)), list(range(nelems))],
                   dtype=np.uint32), np.uint32, "x2")


@pytest.fixture
def ix2(indset):
    return op2.Dat(indset ** 2, np.array([list(range(nelems)), list(range(nelems))],
                   dtype=np.uint32), np.uint32, "ix2")


@pytest.fixture
def bigx(bigiterset):
    return op2.Dat(bigiterset, list(range(2*nelems)), np.uint32, "bigx")


@pytest.fixture
def mapd():
    mapd = list(range(nelems))
    random.shuffle(mapd, lambda: 0.02041724)
    return mapd


@pytest.fixture
def mapd2():
    mapd = list(range(nelems))
    random.shuffle(mapd, lambda: 0.03345714)
    return mapd


@pytest.fixture
def iterset2indset(iterset, indset, mapd):
    u_map = np.array(mapd, dtype=np.uint32)
    return op2.Map(iterset, indset, 1, u_map, "iterset2indset")


@pytest.fixture
def indset2iterset(iterset, indset, mapd2):
    u_map = np.array(mapd2, dtype=np.uint32)
    return op2.Map(indset, iterset, 1, u_map, "indset2iterset")


@pytest.fixture
def bigiterset2indset(bigiterset, indset, mapd):
    u_map = np.array(np.concatenate((mapd, mapd)), dtype=np.uint32)
    return op2.Map(bigiterset, indset, 1, u_map, "bigiterset2indset")


@pytest.fixture
def bigiterset2iterset(bigiterset, iterset):
    u_map = np.array(np.concatenate((list(range(nelems)), list(range(nelems)))), dtype=np.uint32)
    return op2.Map(bigiterset, iterset, 1, u_map, "bigiterset2iterset")


@pytest.fixture
def ker_init():
    return ast.FunDecl('void', 'ker_init',
                       [ast.Decl('int', 'B', qualifiers=['unsigned'], pointers=[''])],
                       ast.Block([ast.Assign(ast.Symbol('B', (0,)), 0)]))


@pytest.fixture
def ker_write():
    return ast.FunDecl('void', 'ker_write',
                       [ast.Decl('int', 'A', qualifiers=['unsigned'], pointers=[''])],
                       ast.Block([ast.Assign(ast.Symbol('A', (0,)), 1)]))


@pytest.fixture
def ker_write2d():
    return ast.FunDecl('void', 'ker_write2d',
                       [ast.Decl('int', 'V', qualifiers=['unsigned'], pointers=[''])],
                       ast.Block([ast.Assign(ast.Symbol('V', (0,)), 1),
                                  ast.Assign(ast.Symbol('V', (1,)), 2)]))


@pytest.fixture
def ker_inc():
    return ast.FunDecl('void', 'ker_inc',
                       [ast.Decl('int', 'B', qualifiers=['unsigned'], pointers=['']),
                        ast.Decl('int', 'A', qualifiers=['unsigned'], pointers=[''])],
                       ast.Block([ast.Incr(ast.Symbol('B', (0,)), ast.Symbol('A', (0,)))]))


@pytest.fixture
def ker_ind_inc():
    return ast.FunDecl('void', 'ker_ind_inc',
                       [ast.Decl('int', 'B', qualifiers=['unsigned'], pointers=['', '']),
                        ast.Decl('int', 'A', qualifiers=['unsigned'], pointers=[''])],
                       ast.Block([ast.Incr(ast.Symbol('B', (0, 0)), ast.Symbol('A', (0,)))]))


@pytest.fixture
def ker_loc_reduce():
    body = ast.Incr('a', ast.Prod(ast.Symbol('V', ('i',)), ast.Symbol('B', (0,))))
    body = \
        [ast.Decl('int', 'a', '0')] +\
        ItSpace().to_for([(0, 2)], ('i',), [body]) +\
        [ast.Assign(ast.Symbol('A', (0,)), 'a')]
    return ast.FunDecl('void', 'ker_loc_reduce',
                       [ast.Decl('int', 'A', qualifiers=['unsigned'], pointers=['']),
                        ast.Decl('int', 'V', qualifiers=['unsigned'], pointers=['']),
                        ast.Decl('int', 'B', qualifiers=['unsigned'], pointers=[''])],
                       ast.Block(body))


@pytest.fixture
def ker_reduce_ind_read():
    body = ast.Incr('a', ast.Prod(ast.Symbol('V', (0, 'i')), ast.Symbol('B', (0,))))
    body = \
        [ast.Decl('int', 'a', '0')] +\
        ItSpace().to_for([(0, 2)], ('i',), [body]) +\
        [ast.Incr(ast.Symbol('A', (0,)), 'a')]
    return ast.FunDecl('void', 'ker_reduce_ind_read',
                       [ast.Decl('int', 'A', qualifiers=['unsigned'], pointers=['']),
                        ast.Decl('int', 'V', qualifiers=['unsigned'], pointers=['', '']),
                        ast.Decl('int', 'B', qualifiers=['unsigned'], pointers=[''])],
                       ast.Block(body))


@pytest.fixture
def ker_ind_reduce():
    incr = ast.Incr(ast.Symbol('A', ('i',)), ast.Symbol('B', (0, 0)))
    body = ItSpace().to_for([(0, 2)], ('i',), [incr])
    return ast.FunDecl('void', 'ker_ind_reduce',
                       [ast.Decl('int', 'A', qualifiers=['unsigned'], pointers=['']),
                        ast.Decl('int', 'B', qualifiers=['unsigned'], pointers=['', ''])],
                       ast.Block(body))


@contextmanager
def loop_fusion(force=None):
    configuration['loop_fusion'] = True

    yield

    if force:
        trace._trace = fuse(lazy_trace_name, trace._trace, mode=force)

    configuration['loop_fusion'] = False


class TestSoftFusion:

    """
    Soft fusion tests. Only loops over the same iteration space presenting
    no indirect read-after-write or write-after-read dependencies may be
    fused.
    """

    def test_fusible_direct_loops(self, ker_init, ker_write, ker_inc,
                                  iterset, x, y, z, skip_greedy):
        """Check that loops over the same iteration space presenting no indirect
        data dependencies are fused and produce the correct result."""
        op2.par_loop(op2.Kernel(ker_init, "ker_init"), iterset, y(op2.WRITE))
        op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, x(op2.WRITE))
        op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), iterset,
                     y(op2.INC), x(op2.READ))
        y.data

        with loop_fusion(force='soft'):
            op2.par_loop(op2.Kernel(ker_init, "ker_init"), iterset, z(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, x(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), iterset,
                         z(op2.INC), x(op2.READ))
        assert np.all(y._data == z.data)

    def test_fusible_fake_indirect_RAW(self, ker_write, ker_inc, iterset,
                                       x, ix, iterset2indset, skip_greedy):
        """Check that two loops over the same iteration space with a "fake" dependency
        are fused. Here, the second loop performs an indirect increment, but since the
        incremented Dat is different than that read in the first loop, loop fusion is
        applicable."""
        with loop_fusion(force='soft'):
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, x(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), iterset,
                         ix(op2.INC, iterset2indset[0]),
                         x(op2.READ))
        assert len(trace._trace) == 1
        assert sum(ix.data) == nelems + sum(range(nelems))

    def test_fusible_fake_indirect_IAI(self, ker_inc, ker_write, iterset,
                                       x, ix, iy, iterset2indset, skip_greedy):
        """Check that two loops over the same iteration space with a "fake" dependency
        are fused. Here, the first loop performs an indirect increment to D1, while the
        second loop performs an indirect increment to D2, but since D1 != D2, loop
        incremented Dat is different than that read in the first loop, loop fusion is
        applicable."""
        with loop_fusion(force='soft'):
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, x(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), iterset,
                         ix(op2.INC, iterset2indset[0]),
                         x(op2.READ))
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), iterset,
                         iy(op2.INC, iterset2indset[0]),
                         x(op2.READ))
        assert len(trace._trace) == 1
        assert np.all(ix.data == iy.data)

    def test_fusible_nontrivial_kernel(self, ker_write2d, ker_loc_reduce, ker_write,
                                       iterset, x2, y, z, skip_greedy):
        """Check that loop fusion works properly when it comes to modify variable
        names within non-trivial kernels to avoid clashes."""
        with loop_fusion(force='soft'):
            op2.par_loop(op2.Kernel(ker_write2d, "ker_write2d"), iterset, x2(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, z(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_loc_reduce, "ker_loc_reduce"), iterset,
                         y(op2.INC), x2(op2.READ), z(op2.READ))
        assert len(trace._trace) == 1
        assert sum(y.data) == nelems * 3

    def test_unfusible_indirect_RAW(self, ker_inc, iterset, x, y, ix,
                                    iterset2indset, skip_greedy):
        """Check that two loops over the same iteration space are not fused to an
        indirect read-after-write dependency."""
        with loop_fusion(force='soft'):
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), iterset,
                         ix(op2.INC, iterset2indset[0]),
                         x(op2.READ))
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), iterset,
                         y(op2.INC),
                         ix(op2.READ, iterset2indset[0]))
        assert len(trace._trace) == 2
        y.data
        assert len(trace._trace) == 0

    def test_unfusible_different_itspace(self, ker_write, iterset, indset,
                                         x, ix, skip_greedy):
        """Check that two loops over different iteration spaces are not fused."""
        with loop_fusion(force='soft'):
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, x(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), indset, ix(op2.WRITE))
        assert len(trace._trace) == 2
        ix.data
        x.data


class TestHardFusion:

    """
    Hard fusion tests. On top of soft fusion, loops presenting incr-after-incr
    dependencies may be fused, even though they iterate over different spaces.
    """

    def test_unfusible_direct_read(self, ker_inc, iterset, indset,
                                   iterset2indset, ix, iy, x, skip_greedy):
        """Check that loops characterized by an inc-after-inc dependency are not
        fused if one of the two loops is direct or the non-base loop performs at
        least one direct read."""
        with loop_fusion(force='hard'):
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), indset,
                         ix(op2.INC), iy(op2.READ))
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), iterset,
                         ix(op2.INC, iterset2indset[0]), x(op2.READ))
        assert len(trace._trace) == 2
        ix.data

    def test_fusible_IAI(self, ker_inc, ker_init, iterset, indset, bigiterset,
                         iterset2indset, bigiterset2indset, bigiterset2iterset,
                         ix, iy, skip_greedy):
        """Check that two indirect loops with no direct reads characterized by
        an inc-after-inc dependency are applied hard fusion."""
        bigiterset2indset.factors = [bigiterset2iterset]

        op2.par_loop(op2.Kernel(ker_init, "ker_init"), indset, ix(op2.WRITE))
        ix.data
        with loop_fusion(force='hard'):
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), bigiterset,
                         ix(op2.INC, bigiterset2indset[0]),
                         iy(op2.READ, bigiterset2indset[0]))
            op2.par_loop(op2.Kernel(ker_inc, "ker_inc"), iterset,
                         ix(op2.INC, iterset2indset[0]),
                         iy(op2.READ, iterset2indset[0]))
        assert len(trace._trace) == 1
        assert sum(ix.data) == sum(range(nelems)) * 3

        bigiterset2indset.factors = []


@pytest.mark.skipif(slope is None, reason="SLOPE required to test tiling")
class TestTiling:

    """
    Tiling tests. A sequence of loops with no synchronization points can be fused
    through tiling. The SLOPE library must be accessible.
    """

    def test_fallback_if_no_slope(self, ker_init, ker_reduce_ind_read, ker_write,
                                  ker_write2d, iterset, indset, iterset2indset,
                                  ix2, x, y, z, skip_greedy):
        """Check that no tiling takes place if SLOPE is not available, although the
        loops can still be executed in the standard fashion."""
        pyop2.fusion.interface.slope = None
        with loop_fusion(force="tile"):
            op2.par_loop(op2.Kernel(ker_init, "ker_init"), iterset, y(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, z(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write2d, "ker_write2d"), indset, ix2(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_reduce_ind_read, "ker_reduce_ind_read"), iterset,
                         y(op2.INC), ix2(op2.READ, iterset2indset), z(op2.READ))
        assert len(trace._trace) == 4
        assert sum(y.data) == nelems * 3
        pyop2.fusion.interface.slope = slope

    @pytest.mark.parametrize(('nu', 'ts'),
                             [(0, 1),
                              (1, 1), (1, nelems//10), (1, nelems),
                              (2, 1), (2, nelems//10), (2, nelems)])
    def test_simple_tiling(self, ker_init, ker_reduce_ind_read, ker_write,
                           ker_write2d, iterset, indset, iterset2indset,
                           ix2, x, y, z, skip_greedy, nu, ts):
        """Check that tiling produces the correct output in a sequence of four
        loops. First two loops are soft-fusible; the remaining three loops are
        fused through tiling. Multiple tile sizes (ts) and unroll factors (nu)
        are tried to check the correctness of different fusion strategies."""

        def time_loop_body():
            op2.par_loop(op2.Kernel(ker_init, "ker_init"), iterset, y(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, z(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write2d, "ker_write2d"), indset, ix2(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_reduce_ind_read, "ker_reduce_ind_read"), iterset,
                         y(op2.INC), ix2(op2.READ, iterset2indset), z(op2.READ))

        # Tiling is skipped until the same sequence is seen three times
        for t in range(2):
            with loop_chain("simple_nu%d" % nu, mode='tile', tile_size=ts, num_unroll=nu):
                time_loop_body()
        assert sum(y.data) == nelems * 3

        for t in range(4):
            with loop_chain("simple_nu%d" % nu, mode='tile', tile_size=ts, num_unroll=nu):
                time_loop_body()
        assert sum(y.data) == nelems * 3

    @pytest.mark.parametrize('sl', [0, 1])
    def test_war_dependency(self, ker_ind_reduce, ker_reduce_ind_read, ker_write,
                            ker_write2d, iterset, indset, sl, iterset2indset,
                            indset2iterset, x, y, ix2, skip_greedy):
        """Check that tiling works properly in presence of write-after-read dependencies."""

        op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, y(op2.WRITE))

        # Tiling is skipped until the same sequence is seen three times
        for t in range(3):
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, x(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write2d, "ker_write2d"), indset, ix2(op2.WRITE))
            with loop_chain("tiling_war", mode='tile',
                            tile_size=nelems//10, num_unroll=1, seed_loop=sl):
                op2.par_loop(op2.Kernel(ker_ind_reduce, "ker_ind_reduce"),
                             indset, ix2(op2.INC), x(op2.READ, indset2iterset))
                op2.par_loop(op2.Kernel(ker_reduce_ind_read, "ker_reduce_ind_read"),
                             iterset, x(op2.INC), ix2(op2.READ, iterset2indset),
                             y(op2.READ))
            assert sum(sum(ix2.data)) == nelems * (1 + 2) + nelems * 2
            assert sum(x.data) == sum(sum(ix2.data)) + nelems

    @pytest.mark.parametrize(('nu', 'ts', 'fs', 'sl'),
                             [(0, 1, (0, 5, 1), 0),
                              (1, nelems//10, (0, 5, 1), 0)])
    def test_advanced_tiling(self, ker_init, ker_reduce_ind_read, ker_ind_reduce,
                             ker_write, ker_write2d, ker_inc, iterset, indset,
                             iterset2indset, indset2iterset, ix2, y, z, skip_greedy,
                             nu, ts, fs, sl):
        """Check that tiling produces the correct output in a sequence of six
        loops. Loops perform direct writes, direct increments, and indirect increments;
        both RAW and WAR dependencies are present. Multiple tile sizes (ts), unroll
        factors (nu), and fusion schemes (fs) are tried to check the correctness of
        different optimization strategies."""

        # Tiling is skipped until the same sequence is seen three times
        for t in range(4):
            with loop_chain("advanced_nu%d" % nu, mode='tile',
                            tile_size=ts, num_unroll=nu, explicit_mode=fs, seed_loop=sl):
                op2.par_loop(op2.Kernel(ker_init, "ker_init"), iterset, y(op2.WRITE))
                op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, z(op2.WRITE))
                op2.par_loop(op2.Kernel(ker_write2d, "ker_write2d"), indset, ix2(op2.WRITE))
                op2.par_loop(op2.Kernel(ker_reduce_ind_read, "ker_reduce_ind_read"), iterset,
                             y(op2.INC), ix2(op2.READ, iterset2indset), z(op2.READ))
                op2.par_loop(op2.Kernel(ker_ind_reduce, "ker_ind_reduce"), indset,
                             ix2(op2.INC), y(op2.READ, indset2iterset))
                op2.par_loop(op2.Kernel(ker_reduce_ind_read, "ker_reduce_ind_read"), iterset,
                             z(op2.INC), ix2(op2.READ, iterset2indset), y(op2.READ))
            assert sum(z.data) == nelems * 27 + nelems
            assert sum(y.data) == nelems * 3
            assert sum(sum(ix2.data)) == nelems * 9

    @pytest.mark.parametrize('sl', [0, 1, 2])
    def test_acyclic_raw_dependency(self, ker_ind_inc, ker_write, iterset,
                                    bigiterset, indset, iterset2indset, indset2iterset,
                                    bigiterset2iterset, x, y, bigx, ix, sl, skip_greedy):
        """Check that tiling produces the correct output in a sequence of loops
        characterized by read-after-write dependencies. SLOPE is told to ignore
        write-after-read dependencies; this test shows that the resulting
        inspector/executor scheme created through SLOPE is anyway correct."""

        # Tiling is skipped until the same sequence is seen three times
        for t in range(3):
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, x(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), iterset, y(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), bigiterset, bigx(op2.WRITE))
            op2.par_loop(op2.Kernel(ker_write, "ker_write"), indset, ix(op2.WRITE))
            with loop_chain("tiling_acyclic_raw", mode='tile', tile_size=nelems//10,
                            num_unroll=1, seed_loop=sl, ignore_war=True):
                op2.par_loop(op2.Kernel(ker_ind_inc, 'ker_ind_inc'), bigiterset,
                             x(op2.INC, bigiterset2iterset), bigx(op2.READ))
                op2.par_loop(op2.Kernel(ker_ind_inc, 'ker_ind_inc'), iterset,
                             ix(op2.INC, iterset2indset), x(op2.READ))
                op2.par_loop(op2.Kernel(ker_ind_inc, 'ker_ind_inc'), indset,
                             y(op2.INC, indset2iterset), ix(op2.READ))
            assert sum(x.data) == nelems * 3
            assert sum(ix.data) == nelems * 4
            assert sum(y.data) == nelems * 5


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
