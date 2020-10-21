import itertools
import ufl
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.constantvalue import Zero
from ufl.core.ufl_type import ufl_type
from ufl.core.operator import Operator
from ufl.form import Form
from ufl import MixedElement
from ufl.precedence import parstr
import firedrake
from firedrake.ufl_expr import Argument, derivative, apply_derivatives
from firedrake.function import Function
from firedrake.subspace import AbstractSubspace, Subspaces, ComplementSubspace, ScalarSubspace, IndexedSubspace


__all__ = ['Projected']


@ufl_type(num_ops=1, is_terminal_modifier=True, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class FiredrakeProjected(Operator):
    __slots__ = (
        "ufl_shape",
        "ufl_free_indices",
        "ufl_index_dimensions",
        "_subspace",
    )

    def __new__(cls, expression, subspace):
        if isinstance(expression, Zero):
            # Zero-simplify indexed Zero objects
            shape = expression.ufl_shape
            fi = expression.ufl_free_indices
            fid = expression.ufl_index_dimensions
            return Zero(shape=shape, free_indices=fi, index_dimensions=fid)
        else:
            return Operator.__new__(cls)

    def __init__(self, expression, subspace):
        # Store operands
        Operator.__init__(self, (expression, ))
        self._subspace = subspace

    def ufl_element(self):
        "Shortcut to get the finite element of the function space of the operand."
        return self.ufl_operands[0].ufl_element()

    def subspace(self):
        return self._subspace

    def _ufl_expr_reconstruct_(self, expression):
        return self._ufl_class_(expression, self.subspace())

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, FiredrakeProjected):
            return False
        else:
            return self.ufl_operands[0] == other.ufl_operands[0] and \
                   self.subspace() == other.subspace()

    def __repr__(self):
        return "%s(%s, %s)" % (self._ufl_class_.__name__, repr(self.ufl_operands[0]), repr(self.subspace()))

    def __str__(self):
        return "%s[%s]" % (parstr(self.ufl_operands[0], self),
                           self._subspace)


def Projected(form_argument, subspace):
    """Return `FiredrakeProjected` objects."""
    if isinstance(subspace, ComplementSubspace):
        return form_argument - Projected(form_argument, subspace.complement)
    if isinstance(subspace, (list, tuple)):
        subspace = Subspaces(*subspace)
    if isinstance(subspace, Subspaces):
        ms = tuple(Projected(form_argument, s) for s in subspace)
        return functools.reduce(lambda a, b: a + b, ms)
    elif isinstance(subspace, (AbstractSubspace, ufl.classes.ListTensor)):
        #TODO: ufl.classes.ListTensor can be removed if we trest splitting appropriately.
        return FiredrakeProjected(form_argument, subspace)
    else:
        raise TypeError("Must be `AbstractSubspace`, `Subspaces`, list, or tuple, not %s." % subspace.__class__.__name__)


from ufl.classes import FormArgument
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.algorithms.map_integrands import map_integrand_dags

from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import unique_pre_traversal


# -- Propagate FiredrakeProjected to directly wrap FormArguments


class FiredrakeProjectedRuleset(MultiFunction):
    def __init__(self, subspace):
        MultiFunction.__init__(self)
        self._subspace = subspace

    def terminal(self, o):
        return o

    expr = MultiFunction.reuse_if_untouched

    def form_argument(self, o):
        "Must act directly on form arguments."
        return FiredrakeProjected(o, self._subspace)


class FiredrakeProjectedRuleDispatcher(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    def terminal(self, o):
        return o

    expr = MultiFunction.reuse_if_untouched

    def firedrake_projected(self, o, A):
        rules = FiredrakeProjectedRuleset(o.subspace())
        return map_expr_dag(rules, A)


def propagate_projection(expression):
    "Propagate FiredrakeProjected nodes to wrap form arguments directly."
    rules = FiredrakeProjectedRuleDispatcher()
    return map_integrand_dags(rules, expression)


# -- Wrap split functions

def split_form_projected(form):
    form = apply_algebra_lowering(form)
    form = apply_derivatives(form)
    form = propagate_projection(form)
    nargs = len(form.arguments())
    # -- Collect every (subspace, arg) pair if Projected(arg, subspace) is found in the form.
    subspace_argument_set = extract_indexed_subspaces(form, cls=Argument) 
    subspaces_list = tuple((None, ) + tuple(s for s, a in subspace_argument_set if a.number() == i) for i in range(nargs))
    # -- Decompose form according to test/trial subspaces.
    subforms = []
    subspaces = []
    extraargs = []
    functions = []
    for subspace_tuple in itertools.product(*subspaces_list):
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
        a = set()
        a.update(extract_type(o.ufl_operands[0], Argument))
        a.update(extract_type(o.ufl_operands[0], Function))
        a = tuple(a)
        if len(a) != 1:
            raise RuntimeError("`FiredrakeProjected` must act on one and only one Argument/Function.")
        a = a[0]
        if isinstance(a, Function):
            return o
        if o.subspace() is self.subspaces[a.number()]:
            return o.ufl_operands[0]
        else:
            shape = o.ufl_shape
            fi = o.ufl_free_indices
            fid = o.ufl_index_dimensions
            return Zero(shape=shape, free_indices=fi, index_dimensions=fid)

    def split(self, form, subspaces):
        """Split a sub-form according to test/trial subspaces.

        :arg form: the sub-form to split.
        :arg subspaces: subspaces of test and trial spaces to extract.
            This should be 0-, 1-, or 2-tuple (whose length is the
            same as the number of arguments as the ``form``). The
            tuple can contain `None`s for extraction of non-projected
            arguments.

        Returns a new :class:`ufl.classes.Form` on the selected subspace.
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
    # Further split subform if projected functions are found.
    subspace_function_set = extract_indexed_subspaces(subform, cls=Function)
    if len(subspace_function_set) == 0:
        return [subform, ], [subspace_tuple, ], [(), ], [(), ]
    subforms = []
    subspaces = []
    extraargs = []
    functions = []
    subsubform = split_form_non_projected_function(subform)
    if subsubform.integrals():
        subforms.append(subsubform)
        subspaces.append(subspace_tuple)
        extraargs.append(())
        functions.append(())
    for s, f in subspace_function_set:
        # Replace f with a new argument.
        f_arg = Argument(f.function_space(), nargs)
        subsubform = split_form_projected_function(subform, s, f, f_arg)
        if subsubform.integrals():
            subsubsubforms = firedrake.formmanipulation.split_form(subsubform)
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


class ProjectedFunctionReplacer(MultiFunction):
    def __init__(self, function, dummy_function):
        MultiFunction.__init__(self)
        self._function = function
        self._dummy_function = dummy_function

    def terminal(self, o):
        return o

    expr = MultiFunction.reuse_if_untouched

    def coefficient(self, o):
        assert o is self._function
        return self._dummy_function


class SplitFormProjectedFunction(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    def terminal(self, o):
        return o

    expr = MultiFunction.reuse_if_untouched

    def firedrake_projected(self, o, A):
        a = set()
        a.update(extract_type(o.ufl_operands[0], Argument))
        a.update(extract_type(o.ufl_operands[0], Function))
        a = tuple(a)
        if len(a) != 1:
            raise RuntimeError("`FiredrakeProjected` must act on one and only one Argument/Function.")
        a = a[0]
        if isinstance(a, Argument):
            raise RuntimeError("Projected arguments must have been removed.")
        elif o.subspace() is self._subspace and a is self._function:
            rules = ProjectedFunctionReplacer(self._function, self._dummy_function)
            return map_expr_dag(rules, A)
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
        a = set()
        a.update(extract_type(o.ufl_operands[0], Argument))
        a.update(extract_type(o.ufl_operands[0], Function))
        a = tuple(a)
        if len(a) != 1:
            raise RuntimeError("`FiredrakeProjected` must act on one and only one Argument/Function.")
        a = a[0]
        if isinstance(a, Argument):
            raise RuntimeError("Projected arguments must have been removed.")
        shape = o.ufl_shape
        fi = o.ufl_free_indices
        fid = o.ufl_index_dimensions
        return Zero(shape=shape, free_indices=fi, index_dimensions=fid)

    def split(self, form):
        return map_integrand_dags(self, form)


def split_form_non_projected_function(form):
    splitter = SplitFormNonProjectedFunction()
    return splitter.split(form)


# -- Helper functions


def extract_subspaces(a, cls=object):
    subspaces_and_objects = extract_indexed_subspaces(a, cls=cls)
    subspaces = set(s if s.parent is None else s.parent for s, o in subspaces_and_objects)
    return tuple(sorted(subspaces, key=lambda x: x.count()))


def extract_indexed_subspaces(a, cls=object):
    from firedrake.slate.slate import TensorBase, Tensor
    if isinstance(a, TensorBase):
        if isinstance(a, Tensor):
            return extract_indexed_subspaces(a.form, cls=cls)
        _set = set()
        for op in a.operands:
            _set.update(extract_indexed_subspaces(op, cls=cls))
        return _set
    elif isinstance(a, Form):
        return set((o.subspace(), o.ufl_operands[0])
                    for e in iter_expressions(a)
                    for o in unique_pre_traversal(e)
                    if isinstance(o, FiredrakeProjected) and isinstance(o.ufl_operands[0], cls))
    else:
        raise TypeError("Unexpected type: %s" % str(type(a)))
