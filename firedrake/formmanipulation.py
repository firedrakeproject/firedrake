
import numpy
import collections

from ufl import as_vector, FormSum, Form, split
from ufl.classes import Zero, FixedIndex, ListTensor, ZeroBaseForm
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.corealg.map_dag import MultiFunction, map_expr_dags

from pyop2 import MixedDat

from firedrake.petsc import PETSc
from firedrake.ufl_expr import Argument
from firedrake.cofunction import Cofunction
from firedrake.functionspace import FunctionSpace, MixedFunctionSpace, DualSpace
from firedrake.matrix import AssembledMatrix
from firedrake import slate


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
                    return child.ufl_operands[indices[0]._value]
                else:
                    return ListTensor(*(child.ufl_operands[i._value] for i in multiindex.indices()))
            return self.expr(o, child, multiindex)

    index_inliner = IndexInliner()

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
        self.blocks = dict(enumerate(argument_indices))
        if len(args) == 0:
            # Functional can't be split
            return form
        if all(len(a.function_space()) == 1 for a in args):
            assert (len(idx) == 1 for idx in self.blocks.values())
            assert (idx[0] == 0 for idx in self.blocks.values())
            return form

        if isinstance(form, slate.slate.TensorBase):
            return slate.slate.Block(form, argument_indices)

        # TODO find a way to distinguish empty Forms avoiding expand_derivatives
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

        V_is = V.subfunctions
        indices = self.blocks[o.number()]

        # Only one index provided.
        if isinstance(indices, int):
            indices = (indices, )

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
                         for j in numpy.ndindex(V_is[i].value_shape)]
        return self._arg_cache.setdefault(o, as_vector(args))

    def cofunction(self, o):
        V = o.function_space()

        # Not on a mixed space, just return ourselves.
        if len(V) == 1:
            return o

        # We only need the test space for Cofunction
        indices = self.blocks[0]
        V_is = V.subfunctions

        # Only one index provided.
        if isinstance(indices, int):
            indices = (indices, )

        # for two-forms, the cofunction should only
        # be returned for the diagonal blocks, so
        # if we are asked for an off-diagonal block
        # then we return a zero form, analogously to
        # the off components of arguments.
        if len(self.blocks) == 2:
            itest, itrial = self.blocks
            on_diag = (itest == itrial)
        else:
            on_diag = True

        # if we are on the diagonal, then return a Cofunction
        # in the relevant subspace that points to the data in
        # the full space. This means that the right hand side
        # of the fieldsplit problem will be correct.
        if on_diag:
            if len(indices) == 1:
                i = indices[0]
                W = V_is[i]
                W = DualSpace(W.mesh(), W.ufl_element())
                c = Cofunction(W, val=o.subfunctions[i].dat)
            else:
                W = MixedFunctionSpace([V_is[i] for i in indices])
                c = Cofunction(W, val=MixedDat(o.dat[i] for i in indices))
        else:
            c = ZeroBaseForm(o.arguments())

        return c


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
    if diagonal:
        assert len(shape) == 2
    for idx in numpy.ndindex(shape):
        f = splitter.split(form, idx)

        # does f actually contain anything?
        if isinstance(f, Cofunction):
            flen = 1
        elif isinstance(f, FormSum):
            flen = len(f.components())
        elif isinstance(f, Form):
            flen = len(f.integrals())
        else:
            raise ValueError(
                "ExtractSubBlock.split should have returned an instance of "
                "either Form, FormSum, or Cofunction")

        if flen > 0:
            if diagonal:
                i, j = idx
                if i != j:
                    continue
                idx = (i, )
            forms.append(SplitForm(indices=idx, form=f))
    return tuple(forms)
