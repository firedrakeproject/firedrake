"""This module implements parallel loops reading and writing
:class:`.Function`\s. This provides a mechanism for implementing
non-finite element operations such as slope limiters."""
from pyop2 import READ, WRITE, RW, INC  # NOQA get flake8 to ignore unused import.
import pyop2
from pyop2.ir.ast_base import *

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
        kargs.append(Decl("double *", Symbol(var, (ndof,))))

    body = FlatBlock(lkernel)

    return pyop2.Kernel(FunDecl("void", "par_loop_kernel", kargs, body),
                        "par_loop_kernel")


def par_loop(kernel, measure, args):
    """Syntax example::

    parloop('for int i=0; i<A.dofs; i++; A[i] = max(A[i], B[0]);', dx,
       {'A' : (A, READWRITE), 'B', (B, READ)})
    """

    _map = _maps[measure.domain_type()]

    mesh = measure.domain_data().function_space().mesh()

    op2args = [_form_kernel(kernel, measure, args)]

    op2args.append(_map['itspace'](mesh, measure))

    op2args += [func.dat(intent, _map['nodes'](func))
                for (func, intent) in args.itervalues()]

    return pyop2.par_loop(*op2args)
