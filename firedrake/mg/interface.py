from pyop2 import op2

import numpy
from finat.ufl import FiniteElement, TensorElement
from finat.quadrature import QuadratureRule
from firedrake import ufl_expr, dmhooks
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.petsc import PETSc
from ufl.duals import is_dual
from . import utils
from . import kernels


__all__ = ["prolong", "restrict", "inject"]


def check_arguments(coarse, fine, needs_dual=False):
    if is_dual(coarse) != needs_dual:
        expected_type = Cofunction if needs_dual else Function
        raise TypeError("Coarse argument is a %s, not a %s" % (type(coarse).__name__, expected_type.__name__))
    if is_dual(fine) != needs_dual:
        expected_type = Cofunction if needs_dual else Function
        raise TypeError("Fine argument is a %s, not a %s" % (type(fine).__name__, expected_type.__name__))
    cfs = coarse.function_space()
    ffs = fine.function_space()
    hierarchy, lvl = utils.get_level(cfs.mesh())
    if hierarchy is None:
        raise ValueError("Coarse argument not from hierarchy")
    fhierarchy, flvl = utils.get_level(ffs.mesh())
    if lvl >= flvl:
        raise ValueError("Coarse argument must be from coarser space")
    if hierarchy is not fhierarchy:
        raise ValueError("Can't transfer between functions from different hierarchies")
    if coarse.ufl_shape != fine.ufl_shape:
        raise ValueError("Mismatching function space shapes")


def get_quadrature_element(V):
    """Returns a ufl element with the quadrature points required by the dual of V."""
    ufl_element = V.ufl_element()
    finat_element = V.finat_element
    if finat_element.has_pointwise_dual_basis:
        return ufl_element

    # Construct a Quadrature element
    Q, ps = finat_element.dual_basis
    degree = finat_element.degree
    wts = numpy.full(len(ps.points), numpy.nan)
    quad_scheme = QuadratureRule(ps, wts, finat_element.cell)
    element = FiniteElement("Quadrature", cell=ufl_element.cell, degree=degree, quad_scheme=quad_scheme)
    if V.value_shape:
        element = TensorElement(element, shape=V.value_shape)
    return element


@PETSc.Log.EventDecorator()
def prolong(coarse, fine):
    check_arguments(coarse, fine)
    Vc = coarse.function_space()
    Vf = fine.function_space()
    if len(Vc) > 1:
        if len(Vc) != len(Vf):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(coarse.subfunctions, fine.subfunctions):
            manager = dmhooks.get_transfer_manager(in_.function_space().dm)
            manager.prolong(in_, out)
        return fine

    if Vc.ufl_element().family() == "Real" or Vf.ufl_element().family() == "Real":
        assert Vc.ufl_element().family() == "Real"
        assert Vf.ufl_element().family() == "Real"
        with fine.dat.vec_wo as dest, coarse.dat.vec_ro as src:
            src.copy(dest)
        return fine

    hierarchy, coarse_level = utils.get_level(ufl_expr.extract_unique_domain(coarse))
    _, fine_level = utils.get_level(ufl_expr.extract_unique_domain(fine))
    refinements_per_level = hierarchy.refinements_per_level
    repeat = (fine_level - coarse_level)*refinements_per_level
    next_level = coarse_level * refinements_per_level

    if needs_quadrature := not Vf.finat_element.has_pointwise_dual_basis:
        # Introduce an intermidiate quadrature target space
        Vf = Vf.collapse().reconstruct(element=get_quadrature_element(Vf))

    finest = fine
    Vfinest = finest.function_space()
    meshes = hierarchy._meshes
    for j in range(repeat):
        next_level += 1
        if j == repeat - 1 and not needs_quadrature:
            fine = finest
        else:
            fine = Function(Vf.reconstruct(mesh=meshes[next_level]))
        Vf = fine.function_space()
        Vc = coarse.function_space()

        coarse_coords = get_coordinates(Vc)
        fine_to_coarse = utils.fine_node_to_coarse_node_map(Vf, Vc)
        fine_to_coarse_coords = utils.fine_node_to_coarse_node_map(Vf, coarse_coords.function_space())
        kernel = kernels.prolong_kernel(coarse, Vf)

        # XXX: Should be able to figure out locations by pushing forward
        # reference cell node locations to physical space.
        # x = \sum_i c_i \phi_i(x_hat)
        node_locations = utils.physical_node_locations(Vf)
        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        for d in [coarse, coarse_coords]:
            d.dat.global_to_local_begin(op2.READ)
            d.dat.global_to_local_end(op2.READ)
        op2.par_loop(kernel, fine.node_set,
                     fine.dat(op2.WRITE),
                     coarse.dat(op2.READ, fine_to_coarse),
                     node_locations.dat(op2.READ),
                     coarse_coords.dat(op2.READ, fine_to_coarse_coords))

        if needs_quadrature:
            # Transfer to the actual target space
            new_fine = finest if j == repeat-1 else Function(Vfinest.reconstruct(mesh=meshes[next_level]))
            fine = new_fine.interpolate(fine)

        coarse = fine
    return fine


@PETSc.Log.EventDecorator()
def restrict(fine_dual, coarse_dual):
    check_arguments(coarse_dual, fine_dual, needs_dual=True)
    Vf = fine_dual.function_space()
    Vc = coarse_dual.function_space()
    if len(Vc) > 1:
        if len(Vc) != len(Vf):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(fine_dual.subfunctions, coarse_dual.subfunctions):
            manager = dmhooks.get_transfer_manager(in_.function_space().dm)
            manager.restrict(in_, out)
        return coarse_dual

    if Vc.ufl_element().family() == "Real" or Vf.ufl_element().family() == "Real":
        assert Vc.ufl_element().family() == "Real"
        assert Vf.ufl_element().family() == "Real"
        with coarse_dual.dat.vec_wo as dest, fine_dual.dat.vec_ro as src:
            src.copy(dest)
        return coarse_dual

    hierarchy, coarse_level = utils.get_level(ufl_expr.extract_unique_domain(coarse_dual))
    _, fine_level = utils.get_level(ufl_expr.extract_unique_domain(fine_dual))
    refinements_per_level = hierarchy.refinements_per_level
    repeat = (fine_level - coarse_level)*refinements_per_level
    next_level = fine_level * refinements_per_level

    if needs_quadrature := not Vf.finat_element.has_pointwise_dual_basis:
        # Introduce an intermidiate quadrature source space
        Vq = Vf.collapse().reconstruct(element=get_quadrature_element(Vf))

    coarsest = coarse_dual.zero()
    meshes = hierarchy._meshes
    for j in range(repeat):
        if needs_quadrature:
            # Transfer to the quadrature source space
            fine_dual = Function(Vq.reconstruct(mesh=meshes[next_level])).interpolate(fine_dual)

        next_level -= 1
        if j == repeat - 1:
            coarse_dual = coarsest
        else:
            coarse_dual = Function(Vc.reconstruct(mesh=meshes[next_level]))
        Vf = fine_dual.function_space()
        Vc = coarse_dual.function_space()

        # XXX: Should be able to figure out locations by pushing forward
        # reference cell node locations to physical space.
        # x = \sum_i c_i \phi_i(x_hat)
        node_locations = utils.physical_node_locations(Vf.dual())

        coarse_coords = get_coordinates(Vc.dual())
        fine_to_coarse = utils.fine_node_to_coarse_node_map(Vf, Vc)
        fine_to_coarse_coords = utils.fine_node_to_coarse_node_map(Vf, coarse_coords.function_space())
        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        for d in [coarse_coords]:
            d.dat.global_to_local_begin(op2.READ)
            d.dat.global_to_local_end(op2.READ)
        kernel = kernels.restrict_kernel(Vf, Vc)
        op2.par_loop(kernel, fine_dual.node_set,
                     coarse_dual.dat(op2.INC, fine_to_coarse),
                     fine_dual.dat(op2.READ),
                     node_locations.dat(op2.READ),
                     coarse_coords.dat(op2.READ, fine_to_coarse_coords))

        fine_dual = coarse_dual
    return coarse_dual


@PETSc.Log.EventDecorator()
def inject(fine, coarse):
    check_arguments(coarse, fine)
    Vf = fine.function_space()
    Vc = coarse.function_space()
    if len(Vc) > 1:
        if len(Vc) != len(Vf):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(fine.subfunctions, coarse.subfunctions):
            manager = dmhooks.get_transfer_manager(in_.function_space().dm)
            manager.inject(in_, out)
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

    hierarchy, coarse_level = utils.get_level(ufl_expr.extract_unique_domain(coarse))
    _, fine_level = utils.get_level(ufl_expr.extract_unique_domain(fine))
    refinements_per_level = hierarchy.refinements_per_level
    repeat = (fine_level - coarse_level)*refinements_per_level
    next_level = fine_level * refinements_per_level

    if needs_quadrature := not Vc.finat_element.has_pointwise_dual_basis:
        # Introduce an intermidiate quadrature target space
        Vc = Vc.collapse().reconstruct(element=get_quadrature_element(Vc))

    kernel, dg = kernels.inject_kernel(Vf, Vc)
    if dg and not hierarchy.nested:
        raise NotImplementedError("Sorry, we can't do supermesh projections yet!")

    coarsest = coarse.zero()
    Vcoarsest = coarsest.function_space()
    meshes = hierarchy._meshes
    for j in range(repeat):
        next_level -= 1
        if j == repeat - 1 and not needs_quadrature:
            coarse = coarsest
        else:
            coarse = Function(Vc.reconstruct(mesh=meshes[next_level]))
        Vc = coarse.function_space()
        Vf = fine.function_space()
        if not dg:
            fine_coords = get_coordinates(Vf)
            coarse_to_fine = utils.coarse_node_to_fine_node_map(Vc, Vf)
            coarse_to_fine_coords = utils.coarse_node_to_fine_node_map(Vc, fine_coords.function_space())

            node_locations = utils.physical_node_locations(Vc)
            # Have to do this, because the node set core size is not right for
            # this expanded stencil
            for d in [fine, fine_coords]:
                d.dat.global_to_local_begin(op2.READ)
                d.dat.global_to_local_end(op2.READ)
            op2.par_loop(kernel, coarse.node_set,
                         coarse.dat(op2.WRITE),
                         fine.dat(op2.READ, coarse_to_fine),
                         node_locations.dat(op2.READ),
                         fine_coords.dat(op2.READ, coarse_to_fine_coords))
        else:
            coarse_coords = get_coordinates(Vc)
            fine_coords = get_coordinates(Vf)
            coarse_cell_to_fine_nodes = utils.coarse_cell_to_fine_node_map(Vc, Vf)
            coarse_cell_to_fine_coords = utils.coarse_cell_to_fine_node_map(Vc, fine_coords.function_space())
            # Have to do this, because the node set core size is not right for
            # this expanded stencil
            for d in [fine, fine_coords]:
                d.dat.global_to_local_begin(op2.READ)
                d.dat.global_to_local_end(op2.READ)
            op2.par_loop(kernel, Vc.mesh().cell_set,
                         coarse.dat(op2.INC, coarse.cell_node_map()),
                         fine.dat(op2.READ, coarse_cell_to_fine_nodes),
                         fine_coords.dat(op2.READ, coarse_cell_to_fine_coords),
                         coarse_coords.dat(op2.READ, coarse_coords.cell_node_map()))

        if needs_quadrature:
            # Transfer to the actual target space
            new_coarse = coarsest if j == repeat - 1 else Function(Vcoarsest.reconstruct(mesh=meshes[next_level]))
            coarse = new_coarse.interpolate(coarse)

        fine = coarse
    return coarse


def get_coordinates(V):
    coords = V.mesh().coordinates
    if V.boundary_set:
        W = V.reconstruct(element=coords.function_space().ufl_element())
        coords = Function(W).interpolate(coords)
    return coords
