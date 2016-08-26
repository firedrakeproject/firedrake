from __future__ import absolute_import

import ufl

from pyop2 import op2

import firedrake
import firedrake.utils
from . import utils
from firedrake.function import Function

__all__ = ["prolong", "restrict", "inject"]


@firedrake.utils.known_pyop2_safe
def prolong(coarse, fine):
    cfs = coarse.function_space()
    hierarchy, lvl = utils.get_level(cfs)
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(fine.function_space())
    if flvl < lvl:
        raise ValueError("Cannot prolong from level %d to level %d" %
                         (lvl, flvl))
    # if the same level, return current function
    if lvl == flvl:
        fine.assign(coarse)
        return
    # number of recursive prolongs
    slvl = flvl - lvl
    if hierarchy is not fhierarchy:
        raise ValueError("Can't prolong between functions from different hierarchies")
    if isinstance(hierarchy, firedrake.MixedFunctionSpaceHierarchy):
        for c, f in zip(coarse.split(), fine.split()):
            prolong(c, f)
        return
    # carry out recursive prolongs
    coarser = coarse
    for j in range(slvl):
        if j == slvl - 1:  # at the top
            intermediate = fine
        else:
            intermediate = Function(fhierarchy._full_hierarchy[lvl + j + 1])
        op2.par_loop(fhierarchy._prolong_kernel,
                     fhierarchy._cell_sets[lvl + j],
                     intermediate.dat(op2.WRITE, fhierarchy.cell_node_map(lvl + j)[op2.i[0]]),
                     coarser.dat(op2.READ, coarser.cell_node_map()))
        if j < slvl - 1:
            coarser = intermediate
    fine.assign(intermediate)


@firedrake.utils.known_pyop2_safe
def restrict(fine, coarse):
    cfs = coarse.function_space()
    hierarchy, lvl = utils.get_level(cfs)
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(fine.function_space())
    if flvl < lvl:
        raise ValueError("Cannot restrict from level %d to level %d" %
                         (flvl, lvl))
    # if the same level, return current function
    if lvl == flvl:
        fine.assign(coarse)
        return
    # number of recursive prolongs
    slvl = flvl - lvl
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

    finer = fine
    # carry out recursive restrictions
    for j in range(slvl):
        if j == slvl - 1:  # at the bottom
            intermediate = coarse
        else:
            intermediate = Function(hierarchy._full_hierarchy[flvl - j - 1])

        args = [intermediate.dat(op2.INC, intermediate.cell_node_map()[op2.i[0]]),
                finer.dat(op2.READ, hierarchy.cell_node_map(flvl - j - 1))]

        if not hierarchy._discontinuous:
            weight = weights[flvl - j]
            args.append(weight.dat(op2.READ, hierarchy._restriction_weights.cell_node_map(flvl - j - 1)))
        intermediate.dat.zero()
        op2.par_loop(hierarchy._restrict_kernel, hierarchy._cell_sets[flvl - j - 1],
                     *args)
        if j < slvl - 1:
            finer = intermediate
    coarse.assign(intermediate)


@firedrake.utils.known_pyop2_safe
def inject(fine, coarse):
    cfs = coarse.function_space()
    hierarchy, lvl = utils.get_level(cfs)
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(fine.function_space())
    if lvl > flvl:
        raise ValueError("Cannot inject from level %d to level %d" %
                         (flvl, lvl))
    # if the same level, return current function
    if lvl == flvl:
        coarse.assign(fine)
        return
    # number of recursive injections
    slvl = flvl - lvl
    if hierarchy is not fhierarchy:
        raise ValueError("Can't prolong between functions from different hierarchies")
    if isinstance(hierarchy, firedrake.MixedFunctionSpaceHierarchy):
        for f, c in zip(fine.split(), coarse.split()):
            inject(f, c)
        return
    # carry out recursive injections
    finer = fine
    for j in range(slvl):
        if j == slvl - 1:  # at the bottom
            intermediate = coarse
        else:
            intermediate = Function(hierarchy._full_hierarchy[flvl - j - 1])
        op2.par_loop(hierarchy._inject_kernel, hierarchy._cell_sets[flvl - j - 1],
                     intermediate.dat(op2.WRITE, intermediate.cell_node_map()[op2.i[0]]),
                     finer.dat(op2.READ, hierarchy.cell_node_map(flvl - j - 1)))
        if j < slvl - 1:
            finer = intermediate
    coarse.assign(intermediate)
