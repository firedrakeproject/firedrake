from __future__ import absolute_import

import numpy

from ufl import as_vector
from ufl.classes import Zero
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.corealg.map_dag import MultiFunction

from firedrake.ufl_expr import Argument


class ExtractSubBlock(MultiFunction):

    """Extract a sub-block from a form.

    :arg test_indices: The indices of the test function to extract.
    :arg trial_indices: The indices of the trial function to extract.
    """

    def __init__(self, test_indices=(), trial_indices=()):
        self.blocks = {0: test_indices,
                       1: trial_indices}
        super(ExtractSubBlock, self).__init__()

    def split(self, form):
        """Split the form.

        :arg form: the form to split.

        Returns either ``None`` (if no part of the form matched) or
        else a single :class:`ufl.classes.Form`.
        """
        args = form.arguments()
        if len(args) == 0:
            raise ValueError("Can't split functional")
        if all(len(a.function_space()) == 1 for a in args):
            assert (len(idx) == 1 for idx in self.blocks.values())
            assert (idx[0] == 0 for idx in self.blocks.values())
            return (form, )
        f = map_integrand_dags(self, form)
        if len(f.integrals()) == 0:
            return None
        return f

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        return o

    def argument(self, o):
        from ufl import split
        from firedrake import MixedFunctionSpace, FunctionSpace
        V = o.function_space()
        if len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o

        V_is = V.split()
        indices = self.blocks[o.number()]

        if len(indices) == 1:
            W = V_is[indices[0]]
            W = FunctionSpace(W.mesh(), W.ufl_element())
            a = (Argument(W, o.number(), part=o.part()), )
        else:
            W = MixedFunctionSpace([V_is[i] for i in indices])
            a = split(Argument(W, o.number(), part=o.part()))
        args = []
        for i in range(len(V_is)):
            if i in indices:
                c = indices.index(i)
                a_ = a[c]
                if len(a_.ufl_shape) == 0:
                    args += [a_]
                else:
                    args += [a_[j] for j in numpy.ndindex(a_.ufl_shape)]
            else:
                args += [Zero()
                         for j in numpy.ndindex(
                         V_is[i].ufl_element().value_shape())]
        return as_vector(args)
