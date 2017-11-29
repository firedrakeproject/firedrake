"""Custom kernels for non-standard Firedrake operations."""

from collections import namedtuple

import numpy as np


__all__ = ['get_transfer_kernels']


TransferKernel = namedtuple("TransferKernel",
                            ["partition", "join"])
TransferKernel.__doc__ = """\
A custom kernel for non-standard Firedrake operations. Each TransferKernel
has a 'mode' which catagorizes the operation the kernel performs. This
includes the following operations:

(1) partition - A kernel which partitions a function defined on
                some function space into nodes interior to each
                cell and nodes on the facets of each cell. This
                takes a single function and produces two new
                functions associated with the interior and facet
                nodes respectively.
(2) join - A kernel which joins two functions defined on an
           interior and facet space respectively into a single
           function defined on the whole space.

Using these kernels is done through standard Firedrake parloops. The
expected arguments for the kernel are labelled as follows:

'x' - function on the unmodified function space;
'x_int' - function on the function space associated with basis functions
          defined on the interior of each cell.
'x_facet' - function on the function space associated with basis functions
            defined on the mesh skeleton (facets only).

The appropriate kernel is returned when a mode is specified.
"""


def get_transfer_kernels(fs_dict):
    """Returns transfer kernel for partitioning data into
    interior and facet nodes respectively, as well as a kernel
    which joins data back together.

    :arg fs_dict: A dictionary of the form
    ```
    fs_dict = {'h1-space': V,
               'interior-space': Vo,
               'facet-space': Vf}
    ```
    where V, Vo, and Vf are the associated function spaces.
    """
    h1key = 'h1-space'
    intkey = 'interior-space'
    facetkey = 'facet-space'

    for key in [h1key, intkey, facetkey]:
        if key not in fs_dict:
            raise ValueError("Dictionary must contain the key '%s'" % key)

    # Offset for interior dof mapping is determined by inspecting the
    # entity dofs of V (original FE space) and the dofs of V_o. For
    # example, degree 5 CG element has entity dofs:
    #
    # {0: {0: [0], 1: [1], 2: [2]}, 1: {0: [3, 4, 5, 6], 1: [7, 8, 9, 10],
    #  2: [11, 12, 13, 14]}, 2: {0: [15, 16, 17, 18, 19, 20]}}.
    #
    # Looking at the cell dofs, we have a starting dof index of 15. The
    # interior element has dofs:
    #
    # {0: {0: [], 1: [], 2: []}, 1: {0: [], 1:[], 2:[]},
    #  2: {0: [0, 1, 2, 3, 4, 5]}}
    #
    # with a starting dof index of 0. So the par_loop will need to be
    # adjusted by the difference: i + 15. The facet dofs do not need
    # an offset.

    # NOTE: This does not work for tensor product elements
    V = fs_dict['h1-space']
    if not V.ufl_element().cell().is_simplex():
        raise NotImplementedError("These transfer kernels only work for "
                                  "simplicial meshes.")

    dim = V.finat_element._element.ref_el.get_dimension()
    offset = V.finat_element.entity_dofs()[dim][0][0]

    Vo = fs_dict['interior-space']
    Vf = fs_dict['facet-space']

    args = (Vo.finat_element.space_dimension(), np.prod(Vo.shape),
            offset,
            Vf.finat_element.space_dimension(), np.prod(Vf.shape))

    join = """
    for (int i=0; i<%d; ++i){
        for (int j=0; j<%d; ++j){
            x[i + %d][j] = x_int[i][j];
        }
    }

    for (int i=0; i<%d; ++i){
        for (int j=0; j<%d; ++j){
            x[i][j] = x_facet[i][j];
        }
    }""" % args

    partition = """
    for (int i=0; i<%d; ++i){
        for (int j=0; j<%d; ++j){
            x_int[i][j] = x[i + %d][j];
        }
    }

    for (int i=0; i<%d; ++i){
        for (int j=0; j<%d; ++j){
            x_facet[i][j] = x[i][j];
        }
    }""" % args

    return TransferKernel(partition=partition, join=join)
