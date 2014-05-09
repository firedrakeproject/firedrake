"""This module implements parallel loops reading and writing
:class:`.Function`\s. This provides a mechanism for implementing
non-finite element operations such as slope limiters."""
from pyop2 import READ, WRITE, RW, INC  # NOQA get flake8 to ignore unused import.
import pyop2
import pyop2.coffee.ast_base as ast

__all__ = ['par_loop', 'READ', 'WRITE', 'RW', 'INC']

"""Map a measure to the correct maps."""
_maps = {
    'cell': {
        'nodes': lambda x: x.cell_node_map(),
        'itspace': lambda mesh, measure: mesh.cell_set
    },
    'interior_facet': {
        'nodes': lambda x: x.interior_facet_node_map(),
        'itspace': lambda mesh, measure: mesh.interior_facets.measure_set(measure)
    },
    'exterior_facet': {
        'nodes': lambda x: x.exterior_facet_node_map(),
        'itspace': lambda mesh, measure: mesh.exterior_facets.measure_set(measure)
    }
}


def _form_kernel(kernel, measure, args):

    kargs = []
    lkernel = kernel

    for var, (func, intent) in args.iteritems():
        ndof = func.function_space().fiat_element.space_dimension()
        lkernel = lkernel.replace(var+".dofs", str(ndof))
        kargs.append(ast.Decl("double *", ast.Symbol(var, (ndof,))))

    body = ast.FlatBlock(lkernel)

    return pyop2.Kernel(ast.FunDecl("void", "par_loop_kernel", kargs, body),
                        "par_loop_kernel")


def par_loop(kernel, measure, args):
    """A :func:`par_loop` is a user-defined operation which reads and
    writes :class:`.Function`\s by looping over the mesh cells or facets
    and accessing the degrees of freedom on adjacent entities.

    :arg kernel: is a string containing the C code to be executed.
    :arg measure: is a :class:`ufl.Measure` which determines the manner in which the iteration over the mesh is to occur.
    :arg args: is a dictionary mapping variable names in the kernel to :class:`.Functions` and indicates how these :class:`.Functions` are to be accessed.

    **Example**

    Assume that `A` is a :class:`.Function` in CG1 and `B` is a
    :class:`.Function` in DG0. Then the following code sets each DoF in
    `A` to the maximum value that `B` attains in the cells adjacent to
    that DoF::

      A.assign(numpy.finfo(0.).min)
      parloop('for (int i=0; i<A.dofs; i++;) A[i] = fmax(A[i], B[0]);', dx,
          {'A' : (A, RW), 'B', (B, READ)})


    **Argument definitions**

    Each item in the `args` dictionary maps a string to a tuple
    containing a :class:`.Function` and an argument intent. The string
    is the c language variable name by which this function will be
    accessed in the kernel. The argument intent indicates how the
    kernel will access this variable:

    `READ`
       The variable will be read but not written to.
    `WRITE`
       The variable will be written to but not read. If multiple kernel
       invocations write to the same DoF, then the order of these writes
       is undefined.
    `RW`
       The variable will be both read and written to. If multiple kernel
       invocations access the same DoF, then the order of these accesses
       is undefined, but it is guaranteed that no race will occur.
    `INC`
       The variable will be added into using +=. As before, the order in
       which the kernel invocations increment the variable is undefined,
       but there is a guarantee that no races will occur.

    **The measure**

    The measure determines the mesh entities over which the iteration
    will occur, and the size of the kernel stencil. The iteration will
    occur over the same mesh entities as if the measure had been used
    to define an integral, and the stencil will likewise be the same
    as the integral case. That is to say, if the measure is a volume
    measure, the kernel will be called once per cell and the DoFs
    accessible to the kernel will be those associated with the cell,
    its facets, edges and vertices. If the measure is a facet measure
    then the iteration will occur over the corresponding class of
    facets and the accessible DoFs will be those on the cell(s)
    adjacent to the facet, and on the facets, edges and vertices
    adjacent to those facets.

    For volume measures the DoFs are guaranteed to be in the FIAT
    local DoFs order. For facet measures, the DoFs will be in sorted
    first by the cell to which they are adjacent. Within each cell,
    they will be in FIAT order. Note that if a continuous
    :class:`.Function` is accessed via an internal facet measure, the
    DoFs on the interface between the two facets will be accessible
    twice: once via each cell. The orientation of the cell(s) relative
    to the current facet is currently arbitrary.

    **The kernel code**

    The kernel code is plain C in which the variables specified in the
    `args` dictionary are available to be read or written in according
    to the argument intent specified. Most basic C operations are
    permitted. However there are some restrictions:

    * Only functions from `math.h
      <http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/math.h.html>`_
      may be called.
    * Pointer operations other than dereferencing arrays are prohibited.

    The free variables are all of type `double**` in which the first
    index is the local node number, while the second index is the
    vector component. The latter only applies to :class:`.Function`\s
    over a :class:`VectorFunctionSpace`, for :class:`.Function`\s over
    a plain :class:`.FunctionSpace` the second index will always be 0.

    """

    _map = _maps[measure.domain_type()]

    mesh = measure.domain_data().function_space().mesh()

    op2args = [_form_kernel(kernel, measure, args)]

    op2args.append(_map['itspace'](mesh, measure))

    op2args += [func.dat(intent, _map['nodes'](func))
                for (func, intent) in args.itervalues()]

    return pyop2.par_loop(*op2args)
