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

from petsc4py import PETSc

class KspSolver(object):

    def __init__(self):
        self._ksp = PETSc.KSP()
        self._ksp.create(PETSc.COMM_WORLD)
        self._pc = self._ksp.getPC()

    def set_parameters(self, parameters):
        self._ksp.setType(parameters['linear_solver'])
        self._pc.setType(parameters['preconditioner'])
        self._ksp.rtol = parameters['relative_tolerance']
        self._ksp.atol = parameters['absolute_tolerance']
        self._ksp.divtol = parameters['divergence_tolerance']
        self._ksp.max_it = parameters['maximum_iterations']

    def solve(self, A, x, b):
        m = A._handle
        px = PETSc.Vec()
        px.createWithArray(x.data)
        pb = PETSc.Vec()
        pb.createWithArray(b.data)
        self._ksp.setOperators(m)
        self._ksp.solve(pb, px)

    def get_converged_reason(self):
        return self._ksp.getConvergedReason()

    def get_iteration_number(self):
        return self._ksp.getIterationNumber()
