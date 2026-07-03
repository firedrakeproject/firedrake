import numpy

from pyop2 import op2
from pyop2.mpi import MPI

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
            target = (finest if j == repeat - 1
                      else Function(Vfinest.reconstruct(mesh=fine_mesh)))
            redist.orig2redist(fine, target)
            fine = target
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
            fine_dual_transfer = Cofunction(
                fine_dual.function_space().reconstruct(mesh=redist.orig)
            )
            redist.redist2orig(fine_dual, fine_dual_transfer)
            fine_dual = fine_dual_transfer
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


def _simplex_cell_volumes(mesh):
    coords = mesh.coordinates
    cell_coords = coords.dat.data_ro_with_halos[coords.cell_node_map().values]
    tdim = mesh.topological_dimension

    if tdim == 2:
        a = cell_coords[:, 1, :] - cell_coords[:, 0, :]
        b = cell_coords[:, 2, :] - cell_coords[:, 0, :]
        if mesh.geometric_dimension == 2:
            return numpy.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]) / 2
        return numpy.linalg.norm(numpy.cross(a, b), axis=1) / 2
    elif tdim == 3:
        a = cell_coords[:, 1, :] - cell_coords[:, 0, :]
        b = cell_coords[:, 2, :] - cell_coords[:, 0, :]
        c = cell_coords[:, 3, :] - cell_coords[:, 0, :]
        return numpy.abs(numpy.einsum("ij,ij->i", numpy.cross(a, b), c)) / 6
    raise NotImplementedError("Padded DG injection is only implemented in dimension 2 and 3")


def _inject_padded_dg(fine, coarse, level, hierarchy):
    Vf = fine.function_space()
    Vc = coarse.function_space()
    meshf = Vf.mesh()
    meshc = Vc.mesh()
    if Vf.finat_element.space_dimension() != 1 or Vc.finat_element.space_dimension() != 1:
        raise NotImplementedError("Padded DG injection is only implemented for DG0 spaces")

    coarse_to_fine = hierarchy.coarse_to_fine_cells[level]
    valid = coarse_to_fine >= 0
    safe_cells = numpy.where(valid, coarse_to_fine, 0)
    fine_nodes = fine.cell_node_map().values[:, 0]
    coarse_nodes = coarse.cell_node_map().values[:, 0]

    fine_volume = _simplex_cell_volumes(meshf)
    coarse_volume = _simplex_cell_volumes(meshc)[:coarse_to_fine.shape[0]]

    fine_values = fine.dat.data_ro_with_halos[fine_nodes]
    scalar = fine_values.ndim == 1
    if scalar:
        fine_values = fine_values[:, None]

    child_values = fine_values[safe_cells]
    child_volumes = fine_volume[safe_cells]
    weighted = child_values * child_volumes[..., None] * valid[..., None]
    injected = weighted.sum(axis=1) / coarse_volume[:, None]

    if scalar:
        coarse.dat.data_wo[coarse_nodes[:coarse_to_fine.shape[0]]] = injected[:, 0]
    else:
        coarse.dat.data_wo[coarse_nodes[:coarse_to_fine.shape[0]]] = injected


def _is_linear_lagrange(V):
    element = V.ufl_element()
    try:
        return (element.family() == "Lagrange"
                and element.degree() == 1
                and element.mapping() == "identity")
    except AttributeError:
        return False


def _inject_adaptive_pointwise(fine, coarse):
    Vf = fine.function_space()
    Vc = coarse.function_space()
    if not (_is_linear_lagrange(Vf) and _is_linear_lagrange(Vc)):
        return False

    meshf = Vf.mesh()
    meshc = Vc.mesh()
    if meshf.coordinates.function_space().ufl_element().degree() != 1:
        return False
    if meshc.coordinates.function_space().ufl_element().degree() != 1:
        return False

    candidates = utils.coarse_node_to_fine_node_map(Vc, Vf).values[:Vc.node_set.size]
    if candidates.shape[0] == 0:
        return True
    coarse_x = meshc.coordinates.dat.data_ro_with_halos[:Vc.node_set.size]
    fine_x = meshf.coordinates.dat.data_ro_with_halos
    distances = numpy.linalg.norm(fine_x[candidates] - coarse_x[:, None, :],
                                  ord=numpy.inf, axis=2)
    rows = numpy.arange(candidates.shape[0])
    best = distances.argmin(axis=1)
    if (distances[rows, best] > 1e-10).any():
        return False

    fine_nodes = candidates[rows, best]
    coarse.dat.data_wo[:Vc.node_set.size] = fine.dat.data_ro_with_halos[fine_nodes]
    return True


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

    coarsest = coarse
    coarsest.dat.data_wo_with_halos[...] = 0
    Vcoarsest = coarsest.function_space()
    meshes = hierarchy._meshes
    for j in range(repeat):
        fine_mesh = meshes[next_level]
        redist = getattr(fine_mesh, "redist", None)
        if redist is not None:
            fine_transfer = Function(
                fine.function_space().reconstruct(mesh=redist.orig)
            )
            redist.redist2orig(fine, fine_transfer)
            fine = fine_transfer
        next_level -= 1
        if j == repeat - 1 and not needs_quadrature:
            coarse = coarsest
        else:
            coarse = Function(Vc.reconstruct(mesh=meshes[next_level]))
        Vc = coarse.function_space()
        Vf = fine.function_space()
        _, level = utils.get_level(Vc.mesh())
        adaptive_parallel = (Vc.mesh().comm.size > 1
                             and type(hierarchy).__name__ == "AdaptiveMeshHierarchy")
        if dg:
            has_padding = adaptive_parallel or Vc.mesh().comm.allreduce(
                bool((hierarchy.coarse_to_fine_cells[level] < 0).any()), op=MPI.LOR
            )
            if has_padding:
                _inject_padded_dg(fine, coarse, level, hierarchy)
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
        elif adaptive_parallel:
            pointwise_ok = Vc.mesh().comm.allreduce(
                _inject_adaptive_pointwise(fine, coarse), op=MPI.LAND
            )
            if not pointwise_ok:
                raise NotImplementedError(
                    "Adaptive parallel injection is only implemented natively for DG0 and CG1"
                )
        else:
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
        if needs_quadrature:
            # Transfer to the actual target space
            new_coarse = coarsest if j == repeat - 1 else Function(Vcoarsest.reconstruct(mesh=meshes[next_level]))
            coarse = new_coarse.interpolate(coarse)
        fine = coarse
    return coarse
