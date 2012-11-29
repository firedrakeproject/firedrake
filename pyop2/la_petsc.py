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

class KspSolver(PETSc.KSP):

    def __init__(self, parameters=None):
        self.create(PETSc.COMM_WORLD)
        self._parameters = parameters or {}
        self._param_actions = { 'linear_solver'       : self.setType,
                                'preconditioner'      : self._setPC,
                                'relative_tolerance'  : self._setRtol,
                                'absolute_tolerance'  : self._setAtol,
                                'divergence_tolerance': self._setDivtol,
                                'maximum_iterations'  : self._setMaxIt }

    def _setPC(self, v):
        self.getPC().setType(v)

    def _setRtol(self, v):
        self.rtol = v

    def _setAtol(self, v):
        self.atol = v

    def _setDivtol(self, v):
        self.divtol = v

    def _setMaxIt(self, v):
        self.max_it = v

    def _set_parameters(self):
        for k, v in self._parameters.iteritems():
            try:
                f = self._param_actions[k]
                f(v)
            except KeyError:
                print "Warning: unknown solver parameter %s" % k

    def update_parameters(self, parameters):
        self._parameters.update(parameters)

    def solve(self, A, x, b):
        self._set_parameters()
        px = PETSc.Vec().createWithArray(x.data)
        pb = PETSc.Vec().createWithArray(b.data)
        self.setOperators(A.handle)
        super(KspSolver, self).solve(pb, px)
