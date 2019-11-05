
import numpy
import collections

from ufl import as_vector
from ufl.classes import Zero, FixedIndex, ListTensor
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.corealg.map_dag import MultiFunction, map_expr_dags

from firedrake.ufl_expr import Argument


class ExtractSubBlock(MultiFunction):

    """Extract a sub-block from a form."""

    class IndexInliner(MultiFunction):
        """Inline fixed index of list tensors"""
        expr = MultiFunction.reuse_if_untouched

        def multi_index(self, o):
            return o

        def indexed(self, o, child, multiindex):
            indices = multiindex.indices()
            if isinstance(child, ListTensor) and all(isinstance(i, FixedIndex) for i in indices):
                if len(indices) == 1:
                    return child.ufl_operands[indices[0]._value]
                else:
                    return ListTensor(*(child.ufl_operands[i._value] for i in multiindex.indices()))
            return self.expr(o, child, multiindex)

    index_inliner = IndexInliner()

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
        self.blocks = dict(enumerate(argument_indices))
        if len(args) == 0:
            # Functional can't be split
            return form
        if all(len(a.function_space()) == 1 for a in args):
            assert (len(idx) == 1 for idx in self.blocks.values())
            assert (idx[0] == 0 for idx in self.blocks.values())
            return form
        f = map_integrand_dags(self, form)
        return f

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        return o

    def expr_list(self, o, *operands):
        # Inline list tensor indexing.
        # This fixes a problem where we extract a subblock from
        # derivative(foo, ...) and end up with the "Argument" looking like
        # [v_0, v_2, v_3][1, 2]
        return self.expr(o, *map_expr_dags(self.index_inliner, operands))

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


def split_form(form, diagonal=False):
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
    if diagonal:
        assert len(shape) == 2
    for idx in numpy.ndindex(shape):
        f = splitter.split(form, idx)
        if len(f.integrals()) > 0:
            if diagonal:
                i, j = idx
                if i != j:
                    continue
                idx = (i, )
            forms.append(SplitForm(indices=idx, form=f))
    return tuple(forms)
