import ufl
from itertools import repeat
from pyop2 import op2

from firedrake import ufl_expr, dmhooks
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.petsc import PETSc
from ufl.duals import is_dual
from ufl.algorithms.analysis import extract_coefficients
from ufl.domain import extract_unique_domain
from . import utils
from . import kernels


__all__ = ["prolong", "restrict", "inject"]


def check_arguments(coarse, fine, needs_dual=False):
    if coarse.ufl_shape != fine.ufl_shape:
        raise ValueError("Mismatching function space shapes")
    coarse, = extract_coefficients(coarse)
    fine, = extract_coefficients(fine)
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


def multigrid_transfer(ufl_interpolate, tensor=None):
    if tensor is None:
        tensor = Function(ufl_interpolate.ufl_function_space())

    coefficients = extract_coefficients(ufl_interpolate)
    if is_dual(ufl_interpolate.ufl_function_space()):
        kernel = kernels.restrict_kernel(ufl_interpolate)
        access = op2.INC
        source, = ufl_interpolate.arguments()
    else:
        kernel = kernels.prolong_kernel(ufl_interpolate)
        access = op2.WRITE
        source, = coefficients

    dual_arg, operand = ufl_interpolate.argument_slots()
    Vtarget = dual_arg.ufl_function_space().dual()
    source_mesh = extract_unique_domain(operand)
    target_mesh = Vtarget.mesh()
    if utils.get_level(target_mesh)[1] > utils.get_level(source_mesh)[1]:
        node_map = utils.fine_node_to_coarse_node_map
    else:
        node_map = utils.coarse_node_to_fine_node_map

    # XXX: Should be able to figure out locations by pushing forward
    # reference cell node locations to physical space.
    # x = \sum_i c_i \phi_i(x_hat)
    target_coords = utils.physical_node_locations(Vtarget)
    source_coords = get_coordinates(source.ufl_function_space())
    # Have to do this, because the node set core size is not right for
    # this expanded stencil
    for d in [source_coords, *coefficients]:
        if d.function_space().mesh() is not target_mesh:
            d.dat.global_to_local_begin(op2.READ)
            d.dat.global_to_local_end(op2.READ)

    def parloop_arg(c, access):
        m_ = None if c.function_space().mesh() is target_mesh else node_map(Vtarget, c.function_space())
        return c.dat(access, m_)

    op2.par_loop(kernel, Vtarget.node_set,
                 parloop_arg(tensor, access),
                 *map(parloop_arg, (*coefficients, target_coords, source_coords), repeat(op2.READ)))
    return tensor


@PETSc.Log.EventDecorator()
def prolong(coarse, fine):
    check_arguments(coarse, fine)
    coarse_expr = coarse
    coarse, = extract_coefficients(coarse_expr)
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
    refine = (fine_level - coarse_level)*refinements_per_level
    next_level = coarse_level * refinements_per_level

    if needs_quadrature := not Vf.finat_element.has_pointwise_dual_basis:
        # Introduce an intermediate quadrature target space
        Vf = Vf.quadrature_space()

    finest = fine
    Vfinest = finest.function_space()
    meshes = hierarchy._meshes
    for j in range(refine):
        next_level += 1
        if j == refine - 1 and not needs_quadrature:
            tensor = finest
        else:
            tensor = None

        fine_dual = ufl.TestFunction(Vf.reconstruct(mesh=meshes[next_level]).dual())
        ufl_interpolate = ufl.Interpolate(coarse_expr, fine_dual)
        fine = multigrid_transfer(ufl_interpolate, tensor=tensor)

        if needs_quadrature:
            # Transfer to the actual target space
            new_fine = finest if j == refine-1 else Function(Vfinest.reconstruct(mesh=meshes[next_level]))
            fine = new_fine.interpolate(fine)

        coarse = fine
        coarse_expr = coarse
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
    refine = (fine_level - coarse_level)*refinements_per_level
    next_level = fine_level * refinements_per_level

    if needs_quadrature := not Vf.finat_element.has_pointwise_dual_basis:
        # Introduce an intermediate quadrature source space
        Vq = Vf.quadrature_space()

    coarsest = coarse_dual.zero()
    meshes = hierarchy._meshes
    for j in range(refine):
        if needs_quadrature:
            # Transfer to the quadrature source space
            fine_dual = Function(Vq.reconstruct(mesh=meshes[next_level])).interpolate(fine_dual)

        next_level -= 1
        if j == refine - 1:
            coarse_dual = coarsest
        else:
            coarse_dual = Function(Vc.reconstruct(mesh=meshes[next_level]))
        Vf = fine_dual.function_space()
        Vc = coarse_dual.function_space()

        coarse_expr = ufl.TestFunction(Vc.dual())
        ufl_interpolate = ufl.Interpolate(coarse_expr, fine_dual)
        multigrid_transfer(ufl_interpolate, tensor=coarse_dual)
        fine_dual = coarse_dual
    return coarse_dual


@PETSc.Log.EventDecorator()
def inject(fine, coarse):
    check_arguments(coarse, fine)
    fine_expr = fine
    fine, = extract_coefficients(fine)
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
    refine = (fine_level - coarse_level)*refinements_per_level
    next_level = fine_level * refinements_per_level

    if needs_quadrature := not Vc.finat_element.has_pointwise_dual_basis:
        # Introduce an intermediate quadrature target space
        Vc = Vc.quadrature_space()

    coarsest = coarse.zero()
    Vcoarsest = coarsest.function_space()
    meshes = hierarchy._meshes
    for j in range(refine):
        next_level -= 1
        if j == refine - 1 and not needs_quadrature:
            coarse = coarsest
        else:
            coarse = Function(Vc.reconstruct(mesh=meshes[next_level]))
        Vc = coarse.function_space()
        Vf = fine.function_space()

        ufl_interpolate = ufl.Interpolate(fine_expr, ufl.TestFunction(Vc.dual()))
        if not Vf.finat_element.is_dg():
            multigrid_transfer(ufl_interpolate, tensor=coarse)
        else:
            kernel, dg = kernels.inject_kernel(ufl_interpolate)
            if dg and not hierarchy.nested:
                raise NotImplementedError("Sorry, we can't do supermesh projections yet!")
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
            new_coarse = coarsest if j == refine - 1 else Function(Vcoarsest.reconstruct(mesh=meshes[next_level]))
            coarse = new_coarse.interpolate(coarse)
        fine = coarse
        fine_expr = fine
    return coarse


def get_coordinates(V):
    coords = V.mesh().coordinates
    if V.boundary_set:
        W = V.reconstruct(element=coords.function_space().ufl_element())
        coords = Function(W).interpolate(coords)
    return coords
