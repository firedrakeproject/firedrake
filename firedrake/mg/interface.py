from pyop2 import op2

import firedrake
from . import utils
from . import kernels


__all__ = ["prolong", "restrict", "inject"]


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
    if coarse.ufl_shape != fine.ufl_shape:
        raise ValueError("Mismatching function space shapes")


def prolong(coarse, fine):
    check_arguments(coarse, fine)
    Vc = coarse.function_space()
    Vf = fine.function_space()
    if len(Vc) > 1:
        if len(Vc) != len(Vf):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(coarse.split(), fine.split()):
            prolong(in_, out)
        return

    if Vc.ufl_element().family() == "Real" or Vf.ufl_element().family() == "Real":
        assert Vc.ufl_element().family() == "Real"
        assert Vf.ufl_element().family() == "Real"
        with fine.dat.vec_wo as dest, coarse.dat.vec_ro as src:
            src.copy(dest)
        return

    hierarchy, coarse_level = utils.get_level(coarse.ufl_domain())
    _, fine_level = utils.get_level(fine.ufl_domain())
    refinements_per_level = hierarchy.refinements_per_level
    repeat = (fine_level - coarse_level)*refinements_per_level
    next_level = coarse_level * refinements_per_level

    element = Vc.ufl_element()
    meshes = hierarchy._meshes
    for j in range(repeat):
        next_level += 1
        if j == repeat - 1:
            next = fine
            Vf = fine.function_space()
        else:
            Vf = firedrake.FunctionSpace(meshes[next_level], element)
            next = firedrake.Function(Vf)

        coarse_coords = Vc.ufl_domain().coordinates
        fine_to_coarse = utils.fine_node_to_coarse_node_map(Vf, Vc)
        fine_to_coarse_coords = utils.fine_node_to_coarse_node_map(Vf, coarse_coords.function_space())
        kernel = kernels.prolong_kernel(coarse)

        # XXX: Should be able to figure out locations by pushing forward
        # reference cell node locations to physical space.
        # x = \sum_i c_i \phi_i(x_hat)
        node_locations = utils.physical_node_locations(Vf)
        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        for d in [coarse, coarse_coords]:
            d.dat._force_evaluation(read=True, write=False)
            d.dat.global_to_local_begin(op2.READ)
            d.dat.global_to_local_end(op2.READ)
        op2.par_loop(kernel, next.node_set,
                     next.dat(op2.WRITE),
                     coarse.dat(op2.READ, fine_to_coarse[op2.i[0]]),
                     node_locations.dat(op2.READ),
                     coarse_coords.dat(op2.READ, fine_to_coarse_coords[op2.i[0]]))
        coarse = next
        Vc = Vf
    return fine


def restrict(fine_dual, coarse_dual):
    check_arguments(coarse_dual, fine_dual)
    Vf = fine_dual.function_space()
    Vc = coarse_dual.function_space()
    if len(Vc) > 1:
        if len(Vc) != len(Vf):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(fine_dual.split(), coarse_dual.split()):
            restrict(in_, out)
        return

    if Vc.ufl_element().family() == "Real" or Vf.ufl_element().family() == "Real":
        assert Vc.ufl_element().family() == "Real"
        assert Vf.ufl_element().family() == "Real"
        with coarse_dual.dat.vec_wo as dest, fine_dual.dat.vec_ro as src:
            src.copy(dest)
        return

    hierarchy, coarse_level = utils.get_level(coarse_dual.ufl_domain())
    _, fine_level = utils.get_level(fine_dual.ufl_domain())
    refinements_per_level = hierarchy.refinements_per_level
    repeat = (fine_level - coarse_level)*refinements_per_level
    next_level = fine_level * refinements_per_level

    element = Vc.ufl_element()
    meshes = hierarchy._meshes

    for j in range(repeat):
        next_level -= 1
        if j == repeat - 1:
            coarse_dual.dat.zero()
            next = coarse_dual
            Vc = next.function_space()
        else:
            Vc = firedrake.FunctionSpace(meshes[next_level], element)
            next = firedrake.Function(Vc)
        # XXX: Should be able to figure out locations by pushing forward
        # reference cell node locations to physical space.
        # x = \sum_i c_i \phi_i(x_hat)
        node_locations = utils.physical_node_locations(Vf)

        coarse_coords = Vc.ufl_domain().coordinates
        fine_to_coarse = utils.fine_node_to_coarse_node_map(Vf, Vc)
        fine_to_coarse_coords = utils.fine_node_to_coarse_node_map(Vf, coarse_coords.function_space())
        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        for d in [coarse_coords]:
            d.dat._force_evaluation(read=True, write=False)
            d.dat.global_to_local_begin(op2.READ)
            d.dat.global_to_local_end(op2.READ)
        kernel = kernels.restrict_kernel(Vf, Vc)
        op2.par_loop(kernel, fine_dual.node_set,
                     next.dat(op2.INC, fine_to_coarse[op2.i[0]]),
                     fine_dual.dat(op2.READ),
                     node_locations.dat(op2.READ),
                     coarse_coords.dat(op2.READ, fine_to_coarse_coords[op2.i[0]]))
        fine_dual = next
        Vf = Vc
    return coarse_dual


def inject(fine, coarse):
    check_arguments(coarse, fine)
    Vf = fine.function_space()
    Vc = coarse.function_space()
    if len(Vc) > 1:
        if len(Vc) != len(Vf):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(fine.split(), coarse.split()):
            inject(in_, out)
        return

    if Vc.ufl_element().family() == "Real" or Vf.ufl_element().family() == "Real":
        assert Vc.ufl_element().family() == "Real"
        assert Vf.ufl_element().family() == "Real"
        with coarse.dat.vec_wo as dest, fine.dat.vec_ro as src:
            src.copy(dest)
        return

    # Algorithm:
    # Loop over coarse nodes
    # Have list of candidate fine cells for each coarse node
    # For each fine cell, pull back to reference space, determine if
    # coarse node location is inside.
    # With candidate cell found, evaluate fine dofs from relevant
    # function at coarse node location.
    #
    # For DG, for each coarse cell, instead:
    # solve inner(u_c, v_c)*dx_c == inner(f, v_c)*dx_c

    kernel, dg = kernels.inject_kernel(Vf, Vc)
    hierarchy, coarse_level = utils.get_level(coarse.ufl_domain())
    if dg and not hierarchy.nested:
        raise NotImplementedError("Sorry, we can't do supermesh projections yet!")
    _, fine_level = utils.get_level(fine.ufl_domain())
    refinements_per_level = hierarchy.refinements_per_level
    repeat = (fine_level - coarse_level)*refinements_per_level
    next_level = fine_level * refinements_per_level

    element = Vc.ufl_element()
    meshes = hierarchy._meshes

    for j in range(repeat):
        next_level -= 1
        if j == repeat - 1:
            coarse.dat.zero()
            next = coarse
            Vc = next.function_space()
        else:
            Vc = firedrake.FunctionSpace(meshes[next_level], element)
            next = firedrake.Function(Vc)
        if not dg:
            node_locations = utils.physical_node_locations(Vc)

            fine_coords = Vf.ufl_domain().coordinates
            coarse_node_to_fine_nodes = utils.coarse_node_to_fine_node_map(Vc, Vf)
            coarse_node_to_fine_coords = utils.coarse_node_to_fine_node_map(Vc, fine_coords.function_space())

            # Have to do this, because the node set core size is not right for
            # this expanded stencil
            for d in [fine, fine_coords]:
                d.dat._force_evaluation(read=True, write=False)
                d.dat.global_to_local_begin(op2.READ)
                d.dat.global_to_local_end(op2.READ)
            op2.par_loop(kernel, next.node_set,
                         next.dat(op2.INC),
                         node_locations.dat(op2.READ),
                         fine.dat(op2.READ, coarse_node_to_fine_nodes[op2.i[0]]),
                         fine_coords.dat(op2.READ, coarse_node_to_fine_coords[op2.i[0]]))
        else:
            coarse_coords = Vc.mesh().coordinates
            fine_coords = Vf.mesh().coordinates
            coarse_cell_to_fine_nodes = utils.coarse_cell_to_fine_node_map(Vc, Vf)
            coarse_cell_to_fine_coords = utils.coarse_cell_to_fine_node_map(Vc, fine_coords.function_space())
            # Have to do this, because the node set core size is not right for
            # this expanded stencil
            for d in [fine, fine_coords]:
                d.dat._force_evaluation(read=True, write=False)
                d.dat.global_to_local_begin(op2.READ)
                d.dat.global_to_local_end(op2.READ)
            op2.par_loop(kernel, Vc.mesh().cell_set,
                         next.dat(op2.INC, next.cell_node_map()[op2.i[0]]),
                         fine.dat(op2.READ, coarse_cell_to_fine_nodes[op2.i[0]]),
                         fine_coords.dat(op2.READ, coarse_cell_to_fine_coords[op2.i[0]]),
                         coarse_coords.dat(op2.READ, coarse_coords.cell_node_map()[op2.i[0]]))
        fine = next
        Vf = Vc
    return coarse
