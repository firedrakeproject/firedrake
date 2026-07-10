from pyop2 import op2

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
        # Introduce an intermediate quadrature target space
        Vf = Vf.quadrature_space()

    finest = fine
    Vfinest = finest.function_space()
    meshes = hierarchy._meshes
    for j in range(repeat):
        next_level += 1
        fine_mesh = meshes[next_level]
        redist = getattr(fine_mesh, "redist", None)
        transfer_mesh = redist.orig if redist is not None else fine_mesh
        if j == repeat - 1 and not needs_quadrature and redist is None:
            fine = finest
        else:
            fine = Function(Vf.reconstruct(mesh=transfer_mesh))
        Vf = fine.function_space()
        Vc = coarse.function_space()
        compose_map = lambda u: utils.fine_node_to_coarse_node_map(Vf, u.function_space())

        # XXX: Should be able to figure out locations by pushing forward
        # reference cell node locations to physical space.
        # x = \sum_i c_i \phi_i(x_hat)
        node_locations = utils.physical_node_locations(Vf)

        kernel = kernels.prolong_kernel(coarse, Vf)
        kernel_args = [
            fine.dat(op2.WRITE),
            coarse.dat(op2.READ, compose_map(coarse)),
            node_locations.dat(op2.READ),
        ]
        # source mesh quantities
        source_mesh = Vc.mesh()
        coarse_coords = source_mesh.coordinates
        kernel_args.append(coarse_coords.dat(op2.READ, compose_map(coarse_coords)))
        if kernel.oriented:
            co = source_mesh.cell_orientations()
            kernel_args.append(co.dat(op2.READ, compose_map(co)))
        if kernel.needs_cell_sizes:
            cs = source_mesh.cell_sizes
            kernel_args.append(cs.dat(op2.READ, compose_map(cs)))
        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        for d in [coarse, coarse_coords]:
            d.dat.global_to_local_begin(op2.READ)
            d.dat.global_to_local_end(op2.READ)
        op2.par_loop(kernel, fine.node_set, *kernel_args)

        if needs_quadrature:
            # Transfer to the actual target space
            target_mesh = transfer_mesh if redist is not None else meshes[next_level]
            new_fine = (finest if j == repeat-1 and redist is None
                        else Function(Vfinest.reconstruct(mesh=target_mesh)))
            fine = new_fine.interpolate(fine)
        if redist is not None:
            # Move from original mesh into the distributed one
            fine_orig = fine
            fine = (finest if j == repeat - 1
                    else Function(Vfinest.reconstruct(mesh=fine_mesh)))
            redist.orig2redist(fine_orig, fine)
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
        # Introduce an intermediate quadrature source space
        Vq = Vf.quadrature_space()

    coarsest = coarse_dual.zero()
    meshes = hierarchy._meshes
    for j in range(repeat):
        fine_mesh = meshes[next_level]
        redist = getattr(fine_mesh, "redist", None)
        if redist is not None:
            # Move from redist mesh to original one, so we can restrict
            Vf_orig = fine_dual.function_space().reconstruct(mesh=redist.orig)
            fine_dual_orig = Function(Vf_orig)
            redist.redist2orig(fine_dual, fine_dual_orig)
            fine_dual = fine_dual_orig
        if needs_quadrature:
            # Transfer to the quadrature source space
            fine_dual = Function(
                Vq.reconstruct(mesh=fine_dual.function_space().mesh())
            ).interpolate(fine_dual)

        next_level -= 1
        if j == repeat - 1:
            coarse_dual = coarsest
        else:
            coarse_dual = Function(Vc.reconstruct(mesh=meshes[next_level]))
        Vf = fine_dual.function_space()
        Vc = coarse_dual.function_space()
        compose_map = lambda u: utils.fine_node_to_coarse_node_map(Vf, u.function_space())

        # XXX: Should be able to figure out locations by pushing forward
        # reference cell node locations to physical space.
        # x = \sum_i c_i \phi_i(x_hat)
        node_locations = utils.physical_node_locations(Vf.dual())

        kernel = kernels.restrict_kernel(Vf, Vc)
        kernel_args = [
            coarse_dual.dat(op2.INC, compose_map(coarse_dual)),
            fine_dual.dat(op2.READ),
            node_locations.dat(op2.READ),
        ]
        # source mesh quantities
        source_mesh = Vc.mesh()
        coarse_coords = source_mesh.coordinates
        kernel_args.append(coarse_coords.dat(op2.READ, compose_map(coarse_coords)))
        if kernel.oriented:
            co = source_mesh.cell_orientations()
            kernel_args.append(co.dat(op2.READ, compose_map(co)))
        if kernel.needs_cell_sizes:
            cs = source_mesh.cell_sizes
            kernel_args.append(cs.dat(op2.READ, compose_map(cs)))
        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        for d in [coarse_coords]:
            d.dat.global_to_local_begin(op2.READ)
            d.dat.global_to_local_end(op2.READ)
        op2.par_loop(kernel, fine_dual.node_set, *kernel_args)
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
        # Introduce an intermediate quadrature target space
        Vc = Vc.quadrature_space()

    kernel, dg = kernels.inject_kernel(Vf, Vc)
    if dg and not hierarchy.nested:
        raise NotImplementedError("Sorry, we can't do supermesh projections yet!")

    coarsest = coarse.zero()
    Vcoarsest = coarsest.function_space()
    meshes = hierarchy._meshes
    for j in range(repeat):
        fine_mesh = meshes[next_level]
        redist = getattr(fine_mesh, "redist", None)
        if redist is not None:
            # Move from redist mesh to original one, so we can inject
            Vf_orig = fine.function_space().reconstruct(mesh=redist.orig)
            fine_orig = Function(Vf_orig)
            redist.redist2orig(fine, fine_orig)
            fine = fine_orig
        next_level -= 1
        if j == repeat - 1 and not needs_quadrature:
            coarse = coarsest
        else:
            coarse = Function(Vc.reconstruct(mesh=meshes[next_level]))
        Vc = coarse.function_space()
        Vf = fine.function_space()
        if not dg:
            compose_map = lambda u: utils.coarse_node_to_fine_node_map(Vc, u.function_space())
            node_locations = utils.physical_node_locations(Vc)
            kernel_args = [
                coarse.dat(op2.WRITE),
                fine.dat(op2.READ, compose_map(fine)),
                node_locations.dat(op2.READ),
            ]
            # source mesh quantities
            source_mesh = Vf.mesh()
            fine_coords = source_mesh.coordinates
            kernel_args.append(fine_coords.dat(op2.READ, compose_map(fine_coords)))
            if kernel.oriented:
                co = source_mesh.cell_orientations()
                kernel_args.append(co.dat(op2.READ, compose_map(co)))
            if kernel.needs_cell_sizes:
                cs = source_mesh.cell_sizes
                kernel_args.append(cs.dat(op2.READ, compose_map(cs)))
            # Have to do this, because the node set core size is not right for
            # this expanded stencil
            for d in [fine, fine_coords]:
                d.dat.global_to_local_begin(op2.READ)
                d.dat.global_to_local_end(op2.READ)
            op2.par_loop(kernel, coarse.node_set, *kernel_args)
        else:
            compose_map = lambda u: utils.coarse_cell_to_fine_node_map(Vc, u.function_space())
            coarse_coords = Vc.mesh().coordinates
            fine_coords = Vf.mesh().coordinates
            # Have to do this, because the node set core size is not right for
            # this expanded stencil
            for d in [fine, fine_coords]:
                d.dat.global_to_local_begin(op2.READ)
                d.dat.global_to_local_end(op2.READ)
            op2.par_loop(kernel, Vc.mesh().cell_set,
                         coarse.dat(op2.INC, coarse.cell_node_map()),
                         fine.dat(op2.READ, compose_map(fine)),
                         fine_coords.dat(op2.READ, compose_map(fine_coords)),
                         coarse_coords.dat(op2.READ, coarse_coords.cell_node_map()))

        if needs_quadrature:
            # Transfer to the actual target space
            new_coarse = coarsest if j == repeat - 1 else Function(Vcoarsest.reconstruct(mesh=meshes[next_level]))
            coarse = new_coarse.interpolate(coarse)
        fine = coarse
    return coarse
