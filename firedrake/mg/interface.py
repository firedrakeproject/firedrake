import pyop3 as op3

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
        if j == repeat - 1 and not needs_quadrature:
            fine = finest
        else:
            fine = Function(Vf.reconstruct(mesh=meshes[next_level]))
        Vf = fine.function_space()
        Vc = coarse.function_space()

        # XXX: Should be able to figure out locations by pushing forward
        # reference cell node locations to physical space.
        # x = \sum_i c_i \phi_i(x_hat)
        node_locations = utils.physical_node_locations(Vf)

        kernel, oriented, needs_cell_sizes = kernels.prolong_kernel(coarse, Vf)
        n = Vf.nodal_axes.blocked(Vf.shape).free.iter()
        compose_map = lambda u: utils.fine_node_to_coarse_node_map(Vf, u.function_space())(n)
        kernel_args = [
            _regionless(fine.dat)[n],
            _regionless(coarse.dat)[compose_map(coarse)],
            _regionless(node_locations.dat)[n],
        ]
        # source mesh quantities
        source_mesh = Vc.mesh()
        coarse_coords = source_mesh.coordinates
        kernel_args.append(_regionless(coarse_coords.dat)[compose_map(coarse_coords)])
        if oriented:
            co = source_mesh.cell_orientations()
            kernel_args.append(_regionless(co.dat)[compose_map(co)])
        if needs_cell_sizes:
            cs = source_mesh.cell_sizes
            kernel_args.append(_regionless(cs.dat)[compose_map(cs)])

        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        for d in [coarse, coarse_coords]:
            d.dat.buffer.reduce_leaves_to_roots_begin()
        for d in [coarse, coarse_coords]:
            d.dat.buffer.reduce_leaves_to_roots_end()
        op3.loop(n, kernel(*kernel_args), eager=True)

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
        # Introduce an intermediate quadrature source space
        Vq = Vf.quadrature_space()

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

        kernel, oriented, needs_cell_sizes = kernels.restrict_kernel(Vf, Vc)
        n = Vf.nodal_axes.blocked(Vf.shape).free.iter()
        compose_map = lambda u: utils.fine_node_to_coarse_node_map(Vf, u.function_space())(n)

        kernel_args = [
            _regionless(coarse_dual.dat)[compose_map(coarse_dual)],
            _regionless(fine_dual.dat)[n],
            _regionless(node_locations.dat)[n],
        ]
        # source mesh quantities
        source_mesh = Vc.mesh()
        coarse_coords = source_mesh.coordinates
        kernel_args.append(_regionless(coarse_coords.dat)[compose_map(coarse_coords)])
        if oriented:
            co = source_mesh.cell_orientations()
            kernel_args.append(_regionless(co.dat)[compose_map(co)])
        if needs_cell_sizes:
            cs = source_mesh.cell_sizes
            kernel_args.append(_regionless(cs.dat)[compose_map(cs)])

        # Have to do this, because the node set core size is not right for
        # this expanded stencil
        coarse_coords.dat.buffer.reduce_leaves_to_roots()
        op3.loop(n, kernel(*kernel_args), eager=True)
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

    (kernel, oriented, needs_cell_sizes), dg = kernels.inject_kernel(Vf, Vc)
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
            node_locations = utils.physical_node_locations(Vc)

            n = Vc.nodal_axes.blocked(Vc.shape).free.iter()
            compose_map = lambda u: utils.coarse_node_to_fine_node_map(Vc, u.function_space())(n)
            kernel_args = [
                _regionless(coarse.dat)[n],
                _regionless(fine.dat)[compose_map(fine)],
                _regionless(node_locations.dat)[n],
            ]
            # source mesh quantities
            source_mesh = Vf.mesh()
            fine_coords = source_mesh.coordinates
            kernel_args.append(_regionless(fine_coords.dat)[compose_map(fine_coords)])
            if oriented:
                co = source_mesh.cell_orientations()
                kernel_args.append(_regionless(co.dat)[compose_map(co)])
            if needs_cell_sizes:
                cs = source_mesh.cell_sizes
                kernel_args.append(_regionless(cs.dat)[compose_map(cs)])

            # Have to do this, because the node set core size is not right for
            # this expanded stencil
            for d in [fine, fine_coords]:
                d.dat.buffer.reduce_leaves_to_roots_begin()
            for d in [fine, fine_coords]:
                d.dat.buffer.reduce_leaves_to_roots_end()
            op3.loop(n, kernel(*kernel_args), eager=True)
        else:
            c = Vc.mesh().cells.owned.iter()
            compose_map = lambda u: utils.coarse_cell_to_fine_node_map(Vc, u.function_space())(c)
            coarse_coords = Vc.mesh().coordinates
            fine_coords = Vf.mesh().coordinates

            # Have to do this, because the node set core size is not right for
            # this expanded stencil
            for d in [fine, fine_coords]:
                d.dat.buffer.reduce_leaves_to_roots_begin()
            for d in [fine, fine_coords]:
                d.dat.buffer.reduce_leaves_to_roots_end()
            op3.loop(
                c,
                kernel(
                    coarse.dat[coarse.function_space().cell_node_map(c)],
                    fine.dat[compose_map(fine)],
                    fine_coords.dat[compose_map(fine_coords)],
                    coarse_coords.dat[coarse_coords.function_space().cell_node_map(c)],
                ),
                eager=True,
            )

        if needs_quadrature:
            # Transfer to the actual target space
            new_coarse = coarsest if j == repeat - 1 else Function(Vcoarsest.reconstruct(mesh=meshes[next_level]))
            coarse = new_coarse.interpolate(coarse)
        fine = coarse
    return coarse


# Think this isnt needed any more
def _regionless(dat):
    """Drop all region (i.e. unconstrained vs constrained) information from a dat.

    This is needed for multigrid because otherwise the node-wise loops fail.

    """
    return dat
    return dat.with_axes(dat.axes.regionless())
