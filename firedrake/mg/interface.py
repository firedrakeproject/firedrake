from __future__ import absolute_import

import numpy as np
import ufl

from pyop2 import op2

import firedrake
import firedrake.utils
from . import utils


__all__ = ["prolong", "restrict", "inject",
           "prolong_matrix", "restrict_matrix",
           "inject_matrix"]


def prolong_matrix(coarse, fine, cbcs=None, fbcs=None):
    hierarchy, lvl = utils.get_level(coarse.function_space())
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")

    fine_map = hierarchy.cell_node_map(lvl)
    coarse_map = coarse.cell_node_map()
    sparsity = op2.Sparsity((fine.function_space().dof_dset,
                             coarse.function_space().dof_dset),
                            (fine_map,
                             coarse_map),
                            "prolongation_%s_%s" % (coarse.function_space().name,
                                                    fine.function_space().name),
                            nest=False)
    mat = op2.Mat(sparsity, np.float64)

    fine_map = hierarchy.cell_node_map(lvl, fbcs)
    coarse_map = coarse.cell_node_map(cbcs)
    op2.par_loop(hierarchy._prolong_matrix,
                 coarse.cell_set,
                 mat(op2.WRITE, (fine_map[op2.i[0]],
                                 coarse_map[op2.i[1]])))
    mat.assemble()
    mat._force_evaluation()
    return mat


def restrict_matrix(fine, coarse):
    if hasattr(fine, "_restriction"):
        return fine._restriction
    hierarchy, lvl = utils.get_level(coarse.function_space())
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")

    fine_map = hierarchy.cell_node_map(lvl)
    sparsity = op2.Sparsity((coarse.function_space().dof_dset,
                             fine.function_space().dof_dset),
                            (coarse.cell_node_map(),
                             fine_map),
                            "restriction_%s_%s" % (fine.function_space().name,
                                                   coarse.function_space().name),
                            nest=False)
    mat = op2.Mat(sparsity, np.float64)

    op2.par_loop(hierarchy._restrict_matrix,
                 coarse.cell_set,
                 mat(op2.WRITE, (coarse.cell_node_map()[op2.i[0]],
                                 fine_map[op2.i[1]])))
    mat.assemble()
    mat._force_evaluation()
    fine._restriction = mat
    return mat


def inject_matrix(fine, coarse):
    # Matrix has nfine rows, ncoarse cols
    if hasattr(fine, "_injection"):
        return fine._injection
    hierarchy, lvl = utils.get_level(coarse.function_space())
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")

    fine_map = hierarchy.cell_node_map(lvl)
    sparsity = op2.Sparsity((fine.function_space().dof_dset,
                             coarse.function_space().dof_dset),
                            (fine_map,
                             coarse.cell_node_map()),
                            "injection_%s_%s" % (fine.function_space().name,
                                                 coarse.function_space().name),
                            nest=False)
    mat = op2.Mat(sparsity, np.float64)

    op2.par_loop(hierarchy._inject_matrix,
                 coarse.cell_set,
                 mat(op2.WRITE, (fine_map[op2.i[1]],
                                 coarse.cell_node_map()[op2.i[0]])))
    mat.assemble()
    mat._force_evaluation()
    fine._injection = mat
    return mat
    
    
@firedrake.utils.known_pyop2_safe
def prolong(coarse, fine):
    cfs = coarse.function_space()
    hierarchy, lvl = utils.get_level(cfs)
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(fine.function_space())
    if flvl != lvl + 1:
        raise ValueError("Can only prolong from level %d to level %d, not %d" %
                         (lvl, lvl + 1, flvl))
    if hierarchy is not fhierarchy:
        raise ValueError("Can't prolong between functions from different hierarchies")
    if isinstance(hierarchy, firedrake.MixedFunctionSpaceHierarchy):
        for c, f in zip(coarse.split(), fine.split()):
            prolong(c, f)
        return
    op2.par_loop(hierarchy._prolong_kernel,
                 hierarchy._cell_sets[lvl],
                 fine.dat(op2.WRITE, hierarchy.cell_node_map(lvl)[op2.i[0]]),
                 coarse.dat(op2.READ, coarse.cell_node_map()))


@firedrake.utils.known_pyop2_safe
def restrict(fine, coarse):
    cfs = coarse.function_space()
    hierarchy, lvl = utils.get_level(cfs)
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(fine.function_space())
    if flvl != lvl + 1:
        raise ValueError("Can only restrict from level %d to level %d, not %d" %
                         (flvl, flvl - 1, lvl))
    if hierarchy is not fhierarchy:
        raise ValueError("Can't restrict between functions from different hierarchies")
    if isinstance(hierarchy, firedrake.MixedFunctionSpaceHierarchy):
        for f, c in zip(fine.split(), coarse.split()):
            restrict(f, c)
        return

    weights = hierarchy._restriction_weights
    # We hit each fine dof more than once since we loop
    # elementwise over the coarse cells.  So we need a count of
    # how many times we did this to weight the final contribution
    # appropriately.
    if not hierarchy._discontinuous and weights is None:
        if isinstance(hierarchy.ufl_element(), ufl.VectorElement):
            element = hierarchy.ufl_element().sub_elements()[0]
            restriction_fs = firedrake.FunctionSpaceHierarchy(hierarchy._mesh_hierarchy, element)
        else:
            restriction_fs = hierarchy
        weights = firedrake.FunctionHierarchy(restriction_fs)

        k = utils.get_count_kernel(hierarchy.cell_node_map(0).arity)

        # Count number of times each fine dof is hit
        for l in range(1, len(weights)):
            op2.par_loop(k, restriction_fs._cell_sets[l-1],
                         weights[l].dat(op2.INC, weights.cell_node_map(l-1)[op2.i[0]]))
            # Inverse, since we're using as weights not counts
            weights[l].assign(1.0/weights[l])
        hierarchy._restriction_weights = weights

    args = [coarse.dat(op2.INC, coarse.cell_node_map()[op2.i[0]]),
            fine.dat(op2.READ, hierarchy.cell_node_map(lvl))]

    if not hierarchy._discontinuous:
        weight = weights[lvl+1]
        args.append(weight.dat(op2.READ, hierarchy._restriction_weights.cell_node_map(lvl)))
    coarse.dat.zero()
    op2.par_loop(hierarchy._restrict_kernel, hierarchy._cell_sets[lvl],
                 *args)


@firedrake.utils.known_pyop2_safe
def inject(fine, coarse):
    cfs = coarse.function_space()
    hierarchy, lvl = utils.get_level(cfs)
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(fine.function_space())
    if flvl != lvl + 1:
        raise ValueError("Can only inject from level %d to level %d, not %d" %
                         (flvl, flvl - 1, lvl))
    if hierarchy is not fhierarchy:
        raise ValueError("Can't prolong between functions from different hierarchies")
    if isinstance(hierarchy, firedrake.MixedFunctionSpaceHierarchy):
        for f, c in zip(fine.split(), coarse.split()):
            inject(f, c)
        return
    op2.par_loop(hierarchy._inject_kernel, hierarchy._cell_sets[lvl],
                 coarse.dat(op2.WRITE, coarse.cell_node_map()[op2.i[0]]),
                 fine.dat(op2.READ, hierarchy.cell_node_map(lvl)))
