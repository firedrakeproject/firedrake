from __future__ import absolute_import, print_function, division
from six.moves import range, zip

import numpy
import collections
import operator
from functools import reduce

from ufl import as_vector, dx
from ufl.classes import Zero, Indexed, MultiIndex, FixedIndex, ListTensor
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.corealg.map_dag import MultiFunction

from firedrake.ufl_expr import Argument


class ExtractSubBlock(MultiFunction):

    """Extract a sub-block from a form."""

    def split(self, form, argument_indices):
        """Split a form.

        :arg form: the form to split.
        :arg argument_indices: indices of test and trial spaces to extract.
            This should be 0-, 1-, or 2-tuple (whose length is the
            same as the number of arguments as the ``form``) whose
            entries are either an integer index, or else an iterable
            of indices.

        Returns a new :class:`ufl.classes.Form` on the selected subspace.
        """
        args = form.arguments()
        self._arg_cache = {}
        self.blocks = dict(zip((0, 1), argument_indices))
        if len(args) == 0:
            # Functional can't be split
            return form
        if all(len(a.function_space()) == 1 for a in args):
            assert (len(idx) == 1 for idx in self.blocks.values())
            assert (idx[0] == 0 for idx in self.blocks.values())
            return form
        f = map_integrand_dags(self, form)
        return f

    def safe_split(self, form, argument_indices):
        """Split a form with no loss of rank."""
        subform = self.split(form, argument_indices)
        zero_form = create_zero_form(form.arguments(), argument_indices)
        if subform.arguments() != zero_form.arguments():
            assert len(subform.integrals()) == 0
            return zero_form
        else:
            return subform

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        return o

    def indexed(self, o, expr, multiindex):
        indices = list(multiindex)
        while indices and isinstance(expr, ListTensor) and isinstance(indices[0], FixedIndex):
            index = indices.pop(0)
            expr = expr.ufl_operands[int(index)]

        if indices == list(multiindex):
            return self.expr(o, expr, multiindex)
        elif indices:
            return Indexed(expr, MultiIndex(tuple(indices)))
        else:
            return expr

    def list_tensor(self, o, *ops):
        all_indexed = all(isinstance(op, Indexed) for op in ops)
        same_indexed = all_indexed and len(set(op.ufl_operands[0] for op in ops)) == 1
        seq_indices = same_indexed and all([FixedIndex(i)] == list(op.ufl_operands[1]) for i, op in enumerate(ops))
        if seq_indices:
            expr, = set(op.ufl_operands[0] for op in ops)
            if expr.ufl_shape == (len(ops),):
                return expr

        return self.expr(o, *ops)

    def argument(self, o):
        from ufl import split
        from firedrake import MixedFunctionSpace, FunctionSpace
        V = o.function_space()
        if len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o

        if o in self._arg_cache:
            return self._arg_cache[o]

        V_is = V.split()
        indices = self.blocks[o.number()]

        try:
            indices = tuple(indices)
            nidx = len(indices)
        except TypeError:
            # Only one index provided.
            indices = (indices, )
            nidx = 1

        if nidx == 1:
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
        return self._arg_cache.setdefault(o, as_vector(args))


SplitForm = collections.namedtuple("SplitForm", ["indices", "form"])


def split_form(form):
    """Split a form into a tuple of sub-forms defined on the component spaces.

    Each entry is a :class:`SplitForm` tuple of the indices into the
    component arguments and the form defined on that block.

    For example, consider the following code:

    .. code-block:: python

        V = FunctionSpace(m, 'CG', 1)
        W = V*V*V
        u, v, w = TrialFunctions(W)
        p, q, r = TestFunctions(W)
        a = q*u*dx + p*w*dx

    Then splitting the form returns a tuple of two forms.

    .. code-block:: python

       ((0, 2), w*p*dx),
        (1, 0), q*u*dx))

    Due to the limited amount of simplification that UFL does, some of
    the returned forms may eventually evaluate to zero.  The form
    compiler will remove these in its more complex simplification
    stages.
    """
    splitter = ExtractSubBlock()
    args = form.arguments()
    shape = tuple(len(a.function_space()) for a in args)
    forms = []
    for idx in numpy.ndindex(shape):
        f = splitter.split(form, idx)
        if len(f.integrals()) > 0:
            forms.append(SplitForm(indices=idx, form=f))
    return tuple(forms)


def create_zero_form(arguments, argument_indices):
    from firedrake import MixedFunctionSpace, FunctionSpace
    zeros = []
    for o, indices in zip(arguments, argument_indices):
        V = o.function_space()
        V_is = V.split()

        try:
            indices = tuple(indices)
            nidx = len(indices)
        except TypeError:
            # Only one index provided.
            indices = (indices,)
            nidx = 1

        if nidx == 1:
            W = V_is[indices[0]]
            W = FunctionSpace(W.mesh(), W.ufl_element())
        else:
            W = MixedFunctionSpace([V_is[i] for i in indices])
        a = Argument(W, o.number(), part=o.part())

        scalar = a[next(numpy.ndindex(a.ufl_shape))]
        zero = Indexed(as_vector([0, scalar]), MultiIndex((FixedIndex(0),)))
        zeros.append(zero)
    return reduce(operator.mul, zeros)*dx
