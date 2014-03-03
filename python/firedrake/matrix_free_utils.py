from pyop2.ffc_interface import compile_form
from solving import _do_assemble
import core_types


def _matrix_diagonal(form, tensor=None, bcs=None):
    """Compute the diagonal entries of a bilinear form, without assembling it.

    :arg form: a bilinear form
    :arg tensor: an optional :class:`Function` to place the result in.
    :arg bcs: an optional list of boundary conditions to apply to the
        assembled diagonal."""
    fd = form.compute_form_data()

    test, trial = fd.original_arguments

    if test.function_space() != trial.function_space():
        raise RuntimeError("""It only makes sense to extract a matrix
        diagonal if the test and trial spaces are the same""")
    kernels = compile_form(form, "a_kernels", {"extract_diagonal": True})

    integrals = fd.preprocessed_form.integrals()
    coords = integrals[0].measure().domain_data()

    if tensor is None:
        out = core_types.Function(test.function_space())
    else:
        out = tensor

    result = lambda: out
    _do_assemble(result=result,
                 tensor=out.dat,
                 bcs=None,
                 kernels=kernels,
                 integrals=integrals,
                 fd=fd,
                 m=test.function_space().mesh(),
                 coords=coords,
                 is_vec=True,
                 test=test)

    return out
