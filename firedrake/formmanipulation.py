
import numpy
import collections

from ufl import as_vector, split
from ufl.classes import Zero, FixedIndex, ListTensor, ZeroBaseForm
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms import expand_derivatives
from ufl.corealg.map_dag import MultiFunction, map_expr_dags

from pyop2 import MixedDat
from pyop2.utils import as_tuple

from firedrake.petsc import PETSc
from firedrake.functionspace import MixedFunctionSpace
from firedrake.cofunction import Cofunction


def subspace(V, indices):
    """Construct a collapsed subspace using components from V."""
    if len(indices) == 1:
        W = V[indices[0]]
    else:
        W = MixedFunctionSpace([V[i] for i in indices])
    return W.collapse()


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
                    return child[indices[0]]
                elif len(indices) == len(child.ufl_operands) and all(k == int(i) for k, i in enumerate(indices)):
                    return child
                else:
                    return ListTensor(*(child[i] for i in indices))
            return self.expr(o, child, multiindex)

    index_inliner = IndexInliner()

    def _subspace_argument(self, a):
        return type(a)(subspace(a.function_space(), self.blocks[a.number()]),
                       a.number(), part=a.part())

    @PETSc.Log.EventDecorator()
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
        self.blocks = dict(enumerate(map(as_tuple, argument_indices)))
        if len(args) == 0:
            # Functional can't be split
            return form
        if all(len(a.function_space()) == 1 for a in args):
            assert (len(idx) == 1 for idx in self.blocks.values())
            assert (idx[0] == 0 for idx in self.blocks.values())
            return form
        # TODO find a way to distinguish empty Forms avoiding expand_derivatives
        f = map_integrand_dags(self, form)
        if expand_derivatives(f).empty():
            # Get ZeroBaseForm with the right shape
            f = ZeroBaseForm(tuple(map(self._subspace_argument, form.arguments())))
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

    def coefficient_derivative(self, o, expr, coefficients, arguments, cds):
        argument, = arguments
        if (isinstance(argument, Zero)
            or (isinstance(argument, ListTensor)
                and all(isinstance(a, Zero) for a in argument.ufl_operands))):
            # If we're only taking a derivative wrt part of an argument in
            # a mixed space other bits might come back as zero. We want to
            # propagate a zero in that case.
            return Zero(o.ufl_shape, o.ufl_free_indices, o.ufl_index_dimensions)
        else:
            return self.reuse_if_untouched(o, expr, coefficients, arguments, cds)

    @PETSc.Log.EventDecorator()
    def argument(self, o):
        V = o.function_space()

        if len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o

        if o in self._arg_cache:
            return self._arg_cache[o]

        indices = self.blocks[o.number()]

        a = self._subspace_argument(o)
        asplit = (a, ) if len(indices) == 1 else split(a)

        args = []
        for i in range(len(V)):
            if i in indices:
                asub = asplit[indices.index(i)]
                args.extend(asub[j] for j in numpy.ndindex(asub.ufl_shape))
            else:
                args.extend(Zero() for j in numpy.ndindex(V[i].value_shape))
        return self._arg_cache.setdefault(o, as_vector(args))

    def cofunction(self, o):
        V = o.function_space()

        if len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o

        # We only need the test space for Cofunction
        indices = self.blocks[0]
        W = subspace(V, indices)
        if len(W) == 1:
            return Cofunction(W, val=o.dat[indices[0]])
        else:
            return Cofunction(W, val=MixedDat(o.dat[i] for i in indices))


SplitForm = collections.namedtuple("SplitForm", ["indices", "form"])


@PETSc.Log.EventDecorator()
def split_form(form, diagonal=False):
    """Split a form into a tuple of sub-forms defined on the component spaces.

    Each entry is a :class:`SplitForm` tuple of the indices into the
    component arguments and the form defined on that block.

    For example, consider the following code:

    .. code-block:: python3

        V = FunctionSpace(m, 'CG', 1)
        W = V*V*V
        u, v, w = TrialFunctions(W)
        p, q, r = TestFunctions(W)
        a = q*u*dx + p*w*dx

    Then splitting the form returns a tuple of two forms.

    .. code-block:: python3

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
    rank = len(shape)
    if diagonal:
        assert rank == 2
        rank = 1
    for idx in numpy.ndindex(shape):
        if diagonal:
            i, j = idx
            if i != j:
                continue
        f = splitter.split(form, idx)
        forms.append(SplitForm(indices=idx[:rank], form=f))
    return tuple(forms)
