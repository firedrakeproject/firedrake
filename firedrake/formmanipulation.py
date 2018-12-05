from pyop2.utils import as_tuple
import numpy
import collections
import firedrake


def extract_sub_block(form, argument_indices):
    """Extract a subblock from a form.

    :arg form: the form to extract from.
    :arg argument_indices: indices of test and trial spaces to extract.
        This should be 0-, 1-, or 2-tuple (whose length is the
        same as the number of arguments as the ``form``) whose
        entries are either an integer index, or else an iterable
        of indices.

    Returns a new :class:`ufl.classes.Form` on the selected subspace.
    """
    argument_indices = tuple(as_tuple(i) for i in argument_indices)
    args = form.arguments()
    if all(len(a.function_space()) == 1 for a in args):
        for indices in argument_indices:
            idx, = indices      # assert singleton
            assert idx == 0
        return form
    slices = []
    for arg in args:
        start = 0
        slicez = []
        for a in firedrake.split(arg):
            end = start + numpy.prod(a.ufl_shape, dtype=int)
            slicez.append(slice(start, end))
            start = end
        slices.append(tuple(slicez))

    mapping = {}
    for (idx, arg, slicez) in zip(argument_indices, args, slices):
        # Default every piece to zero.
        replacement = numpy.full(arg.ufl_shape, firedrake.zero(), dtype=object)
        W = arg.function_space()
        V = firedrake.MixedFunctionSpace(W[i] for i in idx)
        A = firedrake.Argument(V, arg.number(), part=arg.part())
        if len(V) == 1:
            A = (A, )
        else:
            A = firedrake.split(A)
        # For the pieces we're selecting, replace the zero with the relevant indexed slice.
        for i, a in zip(idx, A):
            sub = replacement[slicez[i]].reshape(a.ufl_shape)
            for j in numpy.ndindex(a.ufl_shape):
                sub[j] = a[j]
        mapping[arg] = firedrake.as_vector(replacement)
    return firedrake.replace(form, mapping)


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
    args = form.arguments()
    shape = tuple(len(a.function_space()) for a in args)
    forms = []
    for idx in numpy.ndindex(shape):
        f = extract_sub_block(form, idx)
        if len(f.integrals()) > 0:
            forms.append(SplitForm(indices=idx, form=f))
    return tuple(forms)
