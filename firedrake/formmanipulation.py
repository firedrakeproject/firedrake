import numpy
import collections
import itertools

from ufl import as_vector, MixedElement
from ufl.classes import Zero, FixedIndex, ListTensor
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.analysis import extract_type
from ufl.corealg.map_dag import MultiFunction, map_expr_dags

from firedrake.ufl_expr import Argument, derivative, apply_derivatives
from firedrake.function import Function
from firedrake.subspace import IndexedSubspace, extract_indexed_subspaces
from firedrake.projected import FiredrakeProjected, propagate_projection
from firedrake.petsc import PETSc


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

    def firedrake_projected(self, o, A):
        t = set()
        t.update(extract_type(A, Argument))
        t.update(extract_type(A, Function))
        t = tuple(t)
        try:
            t, = t
        except ValueError:
            raise RuntimeError("`FiredrakeProjected` must act on one and only one Argument/Function.")
        if not isinstance(t, Argument):
            # Only split subspace if argument.
            return o
        if o in self._arg_cache:
            return self._arg_cache[o]
        subspace = o.subspace()
        index = self.blocks[t.number()]
        if subspace.is_zero(index):
            return Zero(o.ufl_shape, o.ufl_free_indices, o.ufl_index_dimensions)
        elif subspace.is_identity(index):
            return A
        else:
            indexed_subspace = IndexedSubspace(subspace, index)
            return self._arg_cache.setdefault(o, FiredrakeProjected(A, indexed_subspace))

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
        if len(f.integrals()) > 0:
            if diagonal:
                i, j = idx
                if i != j:
                    continue
                idx = (i, )
            forms.append(SplitForm(indices=idx, form=f))
    return tuple(forms)


# -- Split form according to subspaces


def split_form_projected(form):
    """Split form according to subspaces.

    :arg form: the form to split.

    If a projected function is found, replace it with an additional argument
    and remember the association of the resulting form (of one higher rank),
    the additional argument, the projected function, the subspace that the
    function is projected onto.

    For instance consider the following:

    .. code-block:: python

        V = FunctionSpace(m, 'CG', 1)
        Vsub = ScalarSubspace(V)
        v = TestFunction(V)
        u = Function(V)
        form = u * v * dx + u * Projected(v, Vsub) * dx + Projected(u, Vsub) * v * dx

    Then split_form_projected(form) returns:

    .. code-block:: python

       (subforms, subspaces, extraargs, functions)

    where:

    .. code-block:: python

        subforms = (u * v * dx, u * v * dx, TrialFunction(V) * v * dx, )
        subspaces = ((None, ), (Vsub, ), (None, Vsub), )
        extraargs = ((), (), (TrialFunction(V), ), )
        functions = ((), (), (u, ), )
    """
    form = apply_algebra_lowering(form)
    form = apply_derivatives(form)
    form = propagate_projection(form)
    nargs = len(form.arguments())
    # Collect (subspace, arg) pair if Projected(arg, subspace) is found in the form.
    subspace_argument_set = extract_indexed_subspaces(form, cls=Argument)
    subspaces_tuple = tuple((None, ) + tuple(s for s, a in subspace_argument_set if a.number() == i) for i in range(nargs))
    # -- Decompose form according to test/trial subspaces.
    subforms = []
    subspaces = []
    extraargs = []
    functions = []
    for subspace_tuple in itertools.product(*subspaces_tuple):
        _subforms, _subspaces, _extraargs, _functions = split_form_projected_argument(form, subspace_tuple)
        subforms.extend(_subforms)
        subspaces.extend(_subspaces)
        extraargs.extend(_extraargs)
        functions.extend(_functions)
    return subforms, subspaces, extraargs, functions


# -- SplitFormProjectedArgument


class SplitFormProjectedArgument(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    def terminal(self, o):
        return o

    def argument(self, o):
        if self.subspaces[o.number()] is None:
            return o
        else:
            shape = o.ufl_shape
            fi = o.ufl_free_indices
            fid = o.ufl_index_dimensions
            return Zero(shape=shape, free_indices=fi, index_dimensions=fid)

    expr = MultiFunction.reuse_if_untouched

    def firedrake_projected(self, o, A):
        a, = o.ufl_operands
        assert isinstance(a, (Argument, Function))
        if isinstance(a, Function):
            return o
        elif self.subspaces[a.number()] is not None and o.subspace() == self.subspaces[a.number()]:
            return a
        else:
            shape = o.ufl_shape
            fi = o.ufl_free_indices
            fid = o.ufl_index_dimensions
            return Zero(shape=shape, free_indices=fi, index_dimensions=fid)

    def split(self, form, subspaces):
        """Split form according to test/trial subspaces.

        :arg form: the form to split.
        :arg subspaces: subspaces of test and trial spaces to extract.
            This should be 0-, 1-, or 2-tuple (whose length is the
            same as the number of arguments as the ``form``). The
            tuple can contain `None`s for extraction of non-projected
            arguments.

        Returns a new :class:`ufl.classes.Form`.
        """
        args = form.arguments()
        if len(subspaces) != len(args):
            raise ValueError("Length of subspaces and arguments must match.")
        if len(args) == 0:
            return form
        self.subspaces = subspaces
        f = map_integrand_dags(self, form)
        return f


def split_form_projected_argument(form, subspace_tuple):
    nargs = len(form.arguments())
    splitter = SplitFormProjectedArgument()
    subform = splitter.split(form, subspace_tuple)
    if subform.integrals() == ():
        return [], [], [], []
    subspace_function_set = extract_indexed_subspaces(subform, cls=Function)
    # Return if projected functions are not found.
    if len(subspace_function_set) == 0:
        return [subform, ], [subspace_tuple, ], [(), ], [(), ]
    # Further split subform if projected functions are found.
    subforms = []
    subspaces = []
    extraargs = []
    functions = []
    # Extract part that does not contain projected functions.
    subsubform = split_form_non_projected_function(subform)
    if subsubform.integrals():
        subforms.append(subsubform)
        subspaces.append(subspace_tuple)
        extraargs.append(())
        functions.append(())
    # Split according to projected functions.
    for s, f in subspace_function_set:
        # Replace f with a new argument.
        f_arg = Argument(f.function_space(), nargs)
        subsubform = split_form_projected_function(subform, s, f, f_arg)
        if subsubform.integrals():
            subsubsubforms = split_form(subsubform)
            for idx, subsubsubform in subsubsubforms:
                assert all(idx[i] == 0 for i in range(nargs))
                assert all(form.arguments()[i] == subsubsubform.arguments()[i] for i in range(nargs))
                subforms.append(subsubsubform)
                if type(s.ufl_element()) == MixedElement:
                    subspaces.append(subspace_tuple + (IndexedSubspace(s, idx[nargs]), ))
                else:
                    assert idx[nargs] == 0
                    subspaces.append(subspace_tuple + (s, ))
                extraargs.append((subsubsubform.arguments()[nargs], ))
                # Here we just remember the parent function, f.
                # The index is found in the associated IndexedSubspace.
                # When submesh lands, f.split() will return a function
                # on a WithGeometry that remembers its parent and the index.
                functions.append((f, ))
    return subforms, subspaces, extraargs, functions


# -- SplitFormProjectedFunction


class SplitFormProjectedFunction(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    def terminal(self, o):
        return o

    expr = MultiFunction.reuse_if_untouched

    def firedrake_projected(self, o, A):
        a, = o.ufl_operands
        assert isinstance(a, Function)
        if o.subspace() is self._subspace and a is self._function:
            return self._dummy_function
        else:
            shape = o.ufl_shape
            fi = o.ufl_free_indices
            fid = o.ufl_index_dimensions
            return Zero(shape=shape, free_indices=fi, index_dimensions=fid)

    def split(self, form, subspace, function, dummy_function):
        self._subspace = subspace
        self._function = function
        self._dummy_function = dummy_function
        f = map_integrand_dags(self, form)
        return f


def split_form_projected_function(form, subspace, function, function_arg):
    """Split form according to the projected function."""
    dummy_function = Function(function.function_space())
    splitter = SplitFormProjectedFunction()
    subform = splitter.split(form, subspace, function, dummy_function)
    subform = derivative(subform, dummy_function, du=function_arg)
    return subform


# -- Split form removing all projected functions.


class SplitFormNonProjectedFunction(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    def terminal(self, o):
        return o

    expr = MultiFunction.reuse_if_untouched

    def firedrake_projected(self, o, A):
        a, = o.ufl_operands
        assert isinstance(a, Function)
        shape = o.ufl_shape
        fi = o.ufl_free_indices
        fid = o.ufl_index_dimensions
        return Zero(shape=shape, free_indices=fi, index_dimensions=fid)

    def split(self, form):
        return map_integrand_dags(self, form)


def split_form_non_projected_function(form):
    splitter = SplitFormNonProjectedFunction()
    return splitter.split(form)
