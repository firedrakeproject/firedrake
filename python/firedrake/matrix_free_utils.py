from pyop2 import op2
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
    kernels = compile_form(form, "a_kernels")

    new_kernels = []

    integrals = fd.preprocessed_form.integrals()
    coords = integrals[0].measure().domain_data()

    coeffs = fd.original_coefficients
    # FIXME: this should really be an FFC-level change.
    for kernel, integral in zip(kernels, integrals):
        args = []
        arg_names = []
        for i, c in enumerate(coeffs):
            args.append("%s **arg%d" % (c.dat.ctype, i))
            arg_names.append("arg%d" % i)
        dt = integral.measure().domain_type()
        if dt == "exterior_facet" or dt == "interior_facet":
            args.append("unsigned int *local_facet")
            arg_names.append("local_facet")

        if len(args) == 0:
            args = ""
            arg_names = ""
        else:
            args = ", ".join([a for a in args]) + ","
            arg_names = ", ".join([a for a in arg_names]) + ","
        k = op2.Kernel("""
        %(orig_kernel)s
        inline void diag_%(orig_kernel_name)s(double A[1], double **coords, %(args)s int k) {
            double t[1][1];
            t[0][0] = 0;
            %(orig_kernel_name)s(t, coords, %(arg_names)s k, k);
            A[0] += t[0][0];
        }
        """ % {'orig_kernel': kernel.code,
               'args': args,
               'orig_kernel_name': kernel.name,
               'arg_names': arg_names}, name='diag_%s' % kernel.name)
        new_kernels.append(k)

    if tensor is None:
        out = core_types.Function(test.function_space())
    else:
        out = tensor

    result = lambda: out
    _do_assemble(result=result,
                 tensor=out.dat,
                 bcs=None,
                 kernels=new_kernels,
                 integrals=integrals,
                 fd=fd,
                 m=test.function_space().mesh(),
                 coords=coords,
                 is_vec=True,
                 test=test)

    return out
