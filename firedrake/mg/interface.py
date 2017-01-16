from __future__ import absolute_import, print_function, division

from functools import partial

from pyop2 import op2

import firedrake
import firedrake.utils
from . import utils


__all__ = ["prolong", "restrict", "inject", "FunctionHierarchy",
           "FunctionSpaceHierarchy", "VectorFunctionSpaceHierarchy",
           "TensorFunctionSpaceHierarchy", "MixedFunctionSpaceHierarchy"]


def check_arguments(coarse, fine):
    cfs = coarse.function_space()
    ffs = fine.function_space()
    hierarchy, lvl = utils.get_level(cfs.mesh())
    if hierarchy is None:
        raise ValueError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(ffs.mesh())
    if lvl >= flvl:
        raise ValueError("Coarse function must be from coarser space")
    if hierarchy is not fhierarchy:
        raise ValueError("Can't transfer between functions from different hierarchies")


@firedrake.utils.known_pyop2_safe
def transfer(input, output, typ=None):
    if len(input.function_space()) > 1:
        if len(output.function_space()) != len(input.function_space()):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(input.split(), output.split()):
            transfer(in_, out, typ=typ)
        return

    if typ == "prolong":
        coarse, fine = input, output
    elif typ in ["inject", "restrict"]:
        coarse, fine = output, input
    else:
        raise ValueError("Unknown transfer type '%s'" % typ)
    check_arguments(coarse, fine)

    hierarchy, coarse_level = utils.get_level(coarse.ufl_domain())
    _, fine_level = utils.get_level(fine.ufl_domain())
    ref_per_level = hierarchy.refinements_per_level
    all_meshes = hierarchy._unskipped_hierarchy

    kernel = None
    element = input.ufl_element()
    repeat = (fine_level - coarse_level)*ref_per_level
    if typ == "prolong":
        next_level = coarse_level*ref_per_level
    else:
        next_level = fine_level*ref_per_level

    for j in range(repeat):
        if typ == "prolong":
            next_level += 1
        else:
            next_level -= 1
        if j == repeat - 1:
            next = output
        else:
            V = firedrake.FunctionSpace(all_meshes[next_level], element)
            next = firedrake.Function(V)
        if typ == "prolong":
            coarse, fine = input, next
        else:
            coarse, fine = next, input

        coarse_V = coarse.function_space()
        fine_V = fine.function_space()
        if kernel is None:
            kernel = utils.get_transfer_kernel(coarse_V, fine_V, typ=typ)
        c2f_map = utils.coarse_to_fine_node_map(coarse_V, fine_V)
        args = [kernel, c2f_map.iterset]
        if typ == "prolong":
            args.append(next.dat(op2.WRITE, c2f_map[op2.i[0]]))
            args.append(input.dat(op2.READ, input.cell_node_map()))
        elif typ == "inject":
            args.append(next.dat(op2.WRITE, next.cell_node_map()[op2.i[0]]))
            args.append(input.dat(op2.READ, c2f_map))
        else:
            next.dat.zero()
            args.append(next.dat(op2.INC, next.cell_node_map()[op2.i[0]]))
            args.append(input.dat(op2.READ, c2f_map))
            weights = utils.get_restriction_weights(coarse_V, fine_V)
            if weights is not None:
                args.append(weights.dat(op2.READ, c2f_map))

        op2.par_loop(*args)
        input = next


prolong = partial(transfer, typ="prolong")

restrict = partial(transfer, typ="restrict")

inject = partial(transfer, typ="inject")


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
