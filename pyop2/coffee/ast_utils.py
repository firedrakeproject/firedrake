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

"""Utility functions for AST transformation."""

import resource
import operator

from pyop2.logger import warning


def increase_stack(asm_opt):
    """"Increase the stack size it the total space occupied by the kernel's local
    arrays is too big."""
    # Assume the size of a C type double is 8 bytes
    double_size = 8
    # Assume the stack size is 1.7 MB (2 MB is usually the limit)
    stack_size = 1.7*1024*1024

    size = 0
    for asm in asm_opt:
        decls = asm.decls.values()
        if decls:
            size += sum([reduce(operator.mul, d.sym.rank) for d in zip(*decls)[0]
                         if d.sym.rank])

    if size*double_size > stack_size:
        # Increase the stack size if the kernel's stack size seems to outreach
        # the space available
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY,
                                                       resource.RLIM_INFINITY))
        except resource.error:
            warning("Stack may blow up, and could not increase its size.")
            warning("In case of failure, lower COFFEE's licm level to 1.")
