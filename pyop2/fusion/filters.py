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

"""Classes for handling duplicate arguments in parallel loops and kernels."""


from collections import OrderedDict

from pyop2.base import READ, RW, WRITE
from pyop2.utils import flatten

from coffee.utils import ast_make_alias
from coffee import base as ast


class Filter(object):

    def _key(self, arg):
        """Arguments accessing the same :class:`base.Dat` with the same
        :class:`base.Map` are considered identical."""
        if arg._is_dat:
            return (arg.data, arg.map)
        elif arg._is_mat:
            return (arg.data,) + tuple(arg.map)
        else:
            return (arg.data,)

    def loop_args(self, loops):
        """Merge and return identical :class:`base.Arg`s appearing in ``loops``.
        Merging two :class:`base.Arg`s means discarding duplicates and taking the
        set union of the access modes (if Arg1 accesses Dat1 in READ mode and Arg2
        accesses Dat1 in WRITE mode, then a single argument is returned with
        access mode RW). Uniqueness is determined by ``self._key``."""

        loop_args = [loop.args for loop in loops]
        filtered_args = OrderedDict()
        for args in loop_args:
            for a in args:
                fa = filtered_args.setdefault(self._key(a), a)
                if a.access != fa.access:
                    if READ in [a.access, fa.access]:
                        # If a READ and some sort of write (MIN, MAX, RW, WRITE,
                        # INC), then the access mode becomes RW
                        fa.access = RW
                    elif WRITE in [a.access, fa.access]:
                        # Can't be a READ, so just stick to WRITE regardless of what
                        # the other access mode is
                        fa.access = WRITE
                    else:
                        # Neither READ nor WRITE, so access modes are some
                        # combinations of RW, INC, MIN, MAX. For simplicity,
                        # just make it RW.
                        fa.access = RW
        return filtered_args

    def kernel_args(self, loops, fundecl):
        """Filter out identical kernel parameters in ``fundecl`` based on the
        :class:`base.Arg`s used in ``loops``."""

        loop_args = list(flatten([l.args for l in loops]))
        unique_loop_args = self.loop_args(loops)
        kernel_args = fundecl.args
        binding = OrderedDict(zip(loop_args, kernel_args))
        new_kernel_args, args_maps = [], []
        for loop_arg, kernel_arg in binding.items():
            unique_loop_arg = unique_loop_args[self._key(loop_arg)]

            # Do nothing if only a single instance of a given Arg is present
            if loop_arg is unique_loop_arg:
                new_kernel_args.append(kernel_arg)
                continue

            # Set up a proper /binding/
            tobind_kernel_arg = binding[unique_loop_arg]
            if tobind_kernel_arg.is_const:
                # Need to remove the /const/ qualifier from the C declaration
                # if the same argument is now written in the fused kernel.
                # Otherwise, /const/ may be appended (if necessary)
                if loop_arg._is_written:
                    tobind_kernel_arg.qual.remove('const')
                elif 'const' not in kernel_arg.qual:
                    kernel_arg.qual.append('const')
            binding[loop_arg] = tobind_kernel_arg

            # An alias may at this point be required
            if kernel_arg.sym.symbol != tobind_kernel_arg.sym.symbol:
                alias = ast_make_alias(tobind_kernel_arg, kernel_arg.sym.symbol)
                args_maps.append(alias)

        fundecl.args[:] = new_kernel_args
        if args_maps:
            args_maps.insert(0, ast.FlatBlock('// Args aliases\n'))
            args_maps.append(ast.FlatBlock('\n'))
        fundecl.body = args_maps + fundecl.body

        return binding


class WeakFilter(Filter):

    def _key(self, arg):
        """Arguments accessing the same :class:`base.Dat` are considered identical,
        irrespective of the :class:`base.Map` used (if any)."""
        return arg.data
