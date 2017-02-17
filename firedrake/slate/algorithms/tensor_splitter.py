from __future__ import absolute_import, print_function, division

import collections

from firedrake.formmanipulation import split_form


SplitTensor = collections.namedtuple("SplitTensor",
                                     ["indices",
                                      "tensor"])


def split_terminal(tensor):
    """
    """
    tensors = []
    for splitform in split_form(tensor.form):
        idx = splitform.indices
        f = splitform.form
        if len(f.integrals()) > 0:
            tensors.append(SplitTensor(indices=idx,
                                       tensor=tensor.reconstruct(f)))

    return tuple(tensors)
