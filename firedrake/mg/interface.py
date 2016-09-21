from __future__ import absolute_import

from pyop2 import op2

import firedrake
import firedrake.utils
from . import utils


__all__ = ["prolong", "restrict", "inject", "FunctionHierarchy",
           "FunctionSpaceHierarchy", "VectorFunctionSpaceHierarchy",
           "TensorFunctionSpaceHierarchy", "MixedFunctionSpaceHierarchy"]


@firedrake.utils.known_pyop2_safe
def prolong(coarse, fine):
    cfs = coarse.function_space()
    ffs = fine.function_space()
    hierarchy, lvl = utils.get_level(cfs.mesh())
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(ffs.mesh())
    if flvl != lvl + 1:
        raise ValueError("Can only prolong from level %d to level %d, not %d" %
                         (lvl, lvl + 1, flvl))
    if hierarchy is not fhierarchy:
        raise ValueError("Can't prolong between functions from different hierarchies")
    if len(cfs) > 1:
        assert len(ffs) == len(cfs)
        for c, f in zip(coarse.split(), fine.split()):
            prolong(c, f)
        return
    c2f_map = utils.coarse_to_fine_node_map(cfs, ffs)
    op2.par_loop(utils.get_transfer_kernel(cfs, ffs, typ="prolong"),
                 c2f_map.iterset,
                 fine.dat(op2.WRITE, c2f_map[op2.i[0]]),
                 coarse.dat(op2.READ, coarse.cell_node_map()))


@firedrake.utils.known_pyop2_safe
def restrict(fine, coarse):
    cfs = coarse.function_space()
    ffs = fine.function_space()
    hierarchy, lvl = utils.get_level(cfs.mesh())
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(ffs.mesh())
    if flvl != lvl + 1:
        raise ValueError("Can only restrict from level %d to level %d, not %d" %
                         (flvl, flvl - 1, lvl))
    if hierarchy is not fhierarchy:
        raise ValueError("Can't restrict between functions from different hierarchies")
    if len(cfs) > 1:
        assert len(ffs) == len(cfs)
        for f, c in zip(fine.split(), coarse.split()):
            restrict(f, c)
        return

    kernel = utils.get_transfer_kernel(cfs, ffs, typ="restrict")

    c2f_map = utils.coarse_to_fine_node_map(cfs, ffs)
    weights = utils.get_restriction_weights(cfs, ffs)

    args = [coarse.dat(op2.INC, coarse.cell_node_map()[op2.i[0]]),
            fine.dat(op2.READ, c2f_map)]

    if weights is not None:
        args.append(weights.dat(op2.READ, c2f_map))
    coarse.dat.zero()
    op2.par_loop(kernel, c2f_map.iterset, *args)


@firedrake.utils.known_pyop2_safe
def inject(fine, coarse):
    cfs = coarse.function_space()
    ffs = fine.function_space()
    hierarchy, lvl = utils.get_level(cfs.mesh())
    if hierarchy is None:
        raise RuntimeError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(ffs.mesh())
    if flvl != lvl + 1:
        raise ValueError("Can only inject from level %d to level %d, not %d" %
                         (flvl, flvl - 1, lvl))
    if hierarchy is not fhierarchy:
        raise ValueError("Can't prolong between functions from different hierarchies")
    if len(cfs) > 1:
        assert len(ffs) == len(cfs)
        for f, c in zip(fine.split(), coarse.split()):
            inject(f, c)
        return
    kernel = utils.get_transfer_kernel(cfs, ffs, typ="inject")
    c2f_map = utils.coarse_to_fine_node_map(cfs, ffs)
    op2.par_loop(kernel, c2f_map.iterset,
                 coarse.dat(op2.WRITE, coarse.cell_node_map()[op2.i[0]]),
                 fine.dat(op2.READ, c2f_map))


def FunctionHierarchy(fs_hierarchy, functions=None):
    """ outdated and returns warning & list of functions corresponding to each level
    of a functionspace hierarchy

        :arg fs_hierarchy: the :class:`~.FunctionSpaceHierarchy` to build on.
        :arg functions: optional :class:`~.Function` for each level.

    """
    from firedrake.logging import warning, RED
    warning(RED % "FunctionHierarchy is obsolete. Falls back by returning list of functions")

    if functions is not None:
        assert len(functions) == len(fs_hierarchy)
        for f, V in zip(functions, fs_hierarchy):
            assert f.function_space() == V
        return tuple(functions)
    else:
        return tuple([firedrake.Function(f) for f in fs_hierarchy])


def FunctionSpaceHierarchy(mesh_hierarchy, *args, **kwargs):
    from firedrake.logging import warning, RED
    warning(RED % "FunctionSpaceHierarchy is obsolete. Just build a FunctionSpace on the relevant mesh")

    return tuple(firedrake.FunctionSpace(mesh, *args, **kwargs) for mesh in mesh_hierarchy)


def VectorFunctionSpaceHierarchy(mesh_hierarchy, *args, **kwargs):
    from firedrake.logging import warning, RED
    warning(RED % "VectorFunctionSpaceHierarchy is obsolete. Just build a FunctionSpace on the relevant mesh")

    return tuple(firedrake.VectorFunctionSpace(mesh, *args, **kwargs) for mesh in mesh_hierarchy)


def TensorFunctionSpaceHierarchy(mesh_hierarchy, *args, **kwargs):
    from firedrake.logging import warning, RED
    warning(RED % "TensorFunctionSpaceHierarchy is obsolete. Just build a FunctionSpace on the relevant mesh")

    return tuple(firedrake.TensorFunctionSpace(mesh, *args, **kwargs) for mesh in mesh_hierarchy)


def MixedFunctionSpaceHierarchy(mesh_hierarchy, *args, **kwargs):
    from firedrake.logging import warning, RED
    warning(RED % "TensorFunctionSpaceHierarchy is obsolete. Just build a FunctionSpace on the relevant mesh")

    kwargs.pop("mesh", None)
    return tuple(firedrake.MixedFunctionSpace(*args, mesh=mesh, **kwargs) for mesh in mesh_hierarchy)
