"""
This file was copied from ngsPETSc.
"""
import time
from fractions import Fraction

import numpy as np
import ufl
from packaging.version import Version
from petsc4py import PETSc

import firedrake as fd
from firedrake.mesh import DISTRIBUTION_PARAMETERS_NOOP
from firedrake.cython import mgimpl as impl, dmcommon
from firedrake import dmhooks
from firedrake.logging import logger

# Netgen and ngsPETSc are not available when the documentation is getting built
# because they do not have ARM wheels.
try:
    from netgen.meshing import MeshingParameters
    from ngsPETSc.plex import MeshMapping
except ImportError:
    pass


def trim_util(T):
    """
    Trim zeros from a connectivity array T.
    """
    if Version(np.__version__) >= Version("2.2"):
        T = np.trim_zeros(T, "b", axis=1).astype(np.int32) - 1
    else:
        T = (np.array([list(np.trim_zeros(a, "b")) for a in list(T)], dtype=np.int32) - 1)
    return T


def snapToNetgenDMPlex(ngmesh, petscPlex):
    '''
    This function snaps the coordinates of a DMPlex mesh to the coordinates of a Netgen mesh.
    '''
    logger.info(f"\t\t\t[{time.time()}]Snapping the DMPlex to NETGEN mesh")

    gdim = petscPlex.getCoordinateDim()
    if gdim == 1:
        ng_coelement = ngmesh.Elements0D()
    elif gdim == 2:
        ng_coelement = ngmesh.Elements1D()
    elif gdim == 3:
        ng_coelement = ngmesh.Elements2D()
    # When we create a netgen mesh from a refined plex,
    # the netgen mesh represents the local submesh.
    # Therefore, there is no need to distribute the netgen data
    nodes_to_correct = ng_coelement.NumPy()["nodes"]
    nodes_to_correct = trim_util(nodes_to_correct)
    nodes_to_correct_sorted = nodes_to_correct.flatten()
    nodes_to_correct_index = np.unique(nodes_to_correct_sorted)
    logger.info(f"\t\t\t[{time.time()}]Nodes have been corrected")
    tic = time.time()
    ngCoordinates = ngmesh.Coordinates()
    petscCoordinates = petscPlex.getCoordinatesLocal().getArray()
    petscCoordinates = petscCoordinates.reshape(-1, petscPlex.getCoordinateDim())
    petscCoordinates[nodes_to_correct_index] = ngCoordinates[nodes_to_correct_index]
    petscPlexCoordinates = petscPlex.getCoordinatesLocal()
    petscPlexCoordinates.setArray(petscCoordinates.flatten())
    petscPlex.setCoordinatesLocal(petscPlexCoordinates)
    toc = time.time()
    logger.info(f"\t\t\tSnap the DMPlex to NETGEN mesh. Time taken: {toc - tic} seconds")


def snapToCoarse(coarse, linear, degree, snap_smoothing, cg):
    '''
    This function snaps the coordinates of a DMPlex mesh to the coordinates of a Netgen mesh.
    '''
    dim = linear.geometric_dimension
    if dim == 2:
        space = fd.VectorFunctionSpace(linear, "CG", degree)
        ho = fd.assemble(fd.interpolate(coarse, space))
        if snap_smoothing == "hyperelastic":
            # Hyperelastic Smoothing
            bcs = [fd.DirichletBC(space, ho, "on_boundary")]
            quad_degree = 2*(degree+1)-1
            d = linear.topological_dimension
            Q = fd.TensorFunctionSpace(linear, "DG", degree=0)
            Jinv = ufl.JacobianInverse(linear)
            hinv = fd.Function(Q)
            hinv.interpolate(Jinv)
            G = ufl.Jacobian(linear) * hinv
            ijac = 1/abs(ufl.det(G))

            def ref_grad(u):
                return ufl.dot(ufl.grad(u), G)

            params = {
                "snes_type": "newtonls",
                "snes_linesearch_type": "l2",
                "snes_max_it": 50,
                "snes_rtol": 1E-8,
                "snes_atol": 1E-8,
                "snes_ksp_ew": True,
                "snes_ksp_ew_rtol0": 1E-2,
                "snes_ksp_ew_rtol_max": 1E-2,
            }
            params["mat_type"] = "aij"
            coarse = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_mat_factor_type": "mumps",
            }
            gmg = {
                "pc_type": "mg",
                "mg_coarse": coarse,
                "mg_levels": {
                    "ksp_max_it": 2,
                    "ksp_type": "chebyshev",
                    "pc_type": "jacobi",
                },
            }
            l = fd.mg.utils.get_level(linear)[1]
            pc = gmg if l else coarse
            params.update(pc)
            ksp = {
                "ksp_rtol": 1E-8,
                "ksp_atol": 0,
                "ksp_type": "minres",
                "ksp_norm_type": "preconditioned",
            }
            params.update(ksp)
            u = ho
            F = ref_grad(u)
            J = ufl.det(F)
            psi = (1/2) * (ufl.inner(F, F)-d - ufl.ln(J**2))
            U = (psi * ijac)*fd.dx(degree=quad_degree)
            dU = ufl.derivative(U, u, fd.TestFunction(space))
            problem = fd.NonlinearVariationalProblem(dU, u, bcs)
            solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)
            solver.set_transfer_manager(None)
            ctx = solver._ctx
            for c in problem.F.coefficients():
                dm = c.function_space().dm
                dmhooks.push_appctx(dm, ctx)
            solver.solve()
        if not cg:
            ho = fd.Function(ho.function_space().broken_space()).interpolate(ho)
    else:
        raise NotImplementedError("Snapping to Netgen meshes is only implemented for 2D meshes.")
    return reconstruct_mesh(linear, ho)


def uniformRefinementRoutine(ngmesh, cdm):
    '''
    Routine called inside of NetgenHierarchy to compute refined ngmesh and plex.
    '''
    # We refine the DMPlex mesh uniformly
    logger.info(f"\t\t\t[{time.time()}]Refining the plex")
    cdm.setRefinementUniform(True)
    rdm = cdm.refine()
    rdm.removeLabel("pyop2_core")
    rdm.removeLabel("pyop2_owned")
    rdm.removeLabel("pyop2_ghost")
    logger.info(f"\t\t\t[{time.time()}]Mapping the mesh to Netgen mesh")
    tic = time.time()
    mapping = MeshMapping(rdm, geo=ngmesh.GetGeometry())
    toc = time.time()
    logger.info(f"\t\t\t[{time.time()}]Mapped the mesh to Netgen. Time taken: {toc-tic}")
    return (rdm, mapping.ngMesh)


def uniformMapRoutine(meshes, lgmaps):
    '''
    This function computes the coarse to fine and fine to coarse maps
    for a uniform mesh hierarchy.
    '''
    refinements_per_level = 1
    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    for (coarse, fine), (clgmaps, flgmaps) in zip(
        zip(meshes[:-1], meshes[1:]),
        zip(lgmaps[:-1], lgmaps[1:])
    ):
        c2f, f2c = impl.coarse_to_fine_cells(coarse, fine, clgmaps, flgmaps)
        coarse_to_fine_cells.append(c2f)
        fine_to_coarse_cells.append(f2c)

    coarse_to_fine_cells = dict((Fraction(i, refinements_per_level), c2f)
                                for i, c2f in enumerate(coarse_to_fine_cells))
    fine_to_coarse_cells = dict((Fraction(i, refinements_per_level), f2c)
                                for i, f2c in enumerate(fine_to_coarse_cells))
    return (coarse_to_fine_cells, fine_to_coarse_cells)


def alfeldRefinementRoutine(ngmesh, cdm):
    '''
    Routing called inside of NetgenHierarchy to compute refined ngmesh and plex.
    '''
    # We refine the netgen mesh alfeld
    ngmesh.SplitAlfeld()
    # We refine the DMPlex mesh alfeld
    tr = PETSc.DMPlexTransform().create(comm=PETSc.COMM_WORLD)
    tr.setType(PETSc.DMPlexTransformType.REFINEREGULAR)
    tr.setDM(cdm)
    tr.setUp()
    rdm = tr.apply(cdm)
    return (rdm, ngmesh)


def alfeldMapRoutine(meshes):
    '''
    This function computes the coarse to fine and fine to coarse maps
    for a alfeld mesh hierarchy.
    '''
    raise NotImplementedError("Alfeld refinement is not implemented yet.")


refinementTypes = {"uniform": (uniformRefinementRoutine, uniformMapRoutine),
                   "Alfeld": (alfeldRefinementRoutine, alfeldMapRoutine)}


def NetgenHierarchy(mesh, levs, flags, distribution_parameters=None):
    """Create a Firedrake mesh hierarchy from Netgen/NGSolve meshes.

    :arg mesh: the Netgen/NGSolve mesh
    :arg levs: the number of levels in the hierarchy
    :arg flags: either a bool or a dictionary containing options for Netgen.
        If not False the hierachy is constructed using ngsPETSc, if None hierarchy
        constructed in a standard manner. Netgen flags includes:

            - degree, either an integer denoting the degree of curvature of all levels of
              the mesh or a list of levs+1 integers denoting the degree of curvature of
              each level of the mesh.
            - tol, geometric tolerance adopted in snapToNetgenDMPlex.
            - refinement_type, the refinment type to be used: uniform (default), Alfeld
    :kwarg distribution_parameters: a dict of options controlling mesh distribution.
        If ``None``, use the same distribution parameters as were used to distribute
        the coarse mesh, otherwise, these options override the default.

    """
    tdim = mesh.topological_dimension
    # Parse netgen flags
    if not isinstance(flags, dict):
        flags = mesh.netgen_flags
    order = flags.get("degree", 1)
    if isinstance(order, int):
        order = [order]*(levs+1)
    permutation_tol = flags.get("permutation_tol", 1e-8)
    refType = flags.get("refinement_type", "uniform")
    optMoves = flags.get("optimisation_moves", False)
    snap = flags.get("snap_to", "geometry")
    snap_smoothing = flags.get("snap_smoothing", "hyperelastic")
    cg = flags.get("cg", not mesh.coordinates.function_space().finat_element.is_dg())
    nested = flags.get("nested", snap in ["coarse"])
    logger.info(f"Creating a Netgen hierarchy with {levs} levels.")
    logger.info(f"\tOrder of the hierarchy: {order}")
    logger.info(f"\tRefinement type: {refType}")
    logger.info(f"\tSnap to {snap} using {snap_smoothing} smoothing (if snapping to coarse)")
    # Firedrake quantities
    meshes = []
    lgmaps = []
    # Curve the mesh
    if order[0] != mesh.coordinates.function_space().ufl_element().degree():
        coordinates = mesh.curve_field(
            order=order[0],
            permutation_tol=permutation_tol,
            cg_field=cg,
        )
        mesh = reconstruct_mesh(mesh, coordinates)
    # Make a plex (cdm) without overlap.
    dm_cell_type, = mesh.dm_cell_types
    cdm = dmcommon.submesh_create(mesh.topology_dm, tdim, "celltype", dm_cell_type, True)
    cdm.removeLabel("pyop2_core")
    cdm.removeLabel("pyop2_owned")
    cdm.removeLabel("pyop2_ghost")
    no = impl.create_lgmap(cdm)
    o = impl.create_lgmap(mesh.topology_dm)
    lgmaps.append((no, o))
    mesh.topology_dm.setRefineLevel(0)
    meshes.append(mesh)
    ngmesh = mesh.netgen_mesh
    for l in range(1, levs+1):
        # Straighten the mesh
        ngmesh.Curve(1)
        rdm, ngmesh = refinementTypes[refType][0](ngmesh, cdm)
        cdm = rdm
        if optMoves:
            # Optimises the mesh, for example smoothing
            if tdim == 2:
                ngmesh.OptimizeMesh2d(MeshingParameters(optimize2d=optMoves))
            elif tdim == 3:
                ngmesh.OptimizeVolumeMesh(MeshingParameters(optimize3d=optMoves))
            else:
                raise ValueError("Only 2D and 3D meshes can be optimised.")
        # Snap the mesh to the Netgen mesh
        if snap == "geometry":
            snapToNetgenDMPlex(ngmesh, rdm)

        # We construct a Firedrake mesh from the DMPlex mesh
        parameters = {}
        if distribution_parameters is not None:
            parameters.update(distribution_parameters)
        else:
            parameters.update(mesh._distribution_parameters)
        parameters["partition"] = False
        mesh = fd.Mesh(rdm, dim=mesh.geometric_dimension,
                       reorder=False,
                       distribution_parameters=parameters,
                       tolerance=mesh.tolerance)
        mesh.netgen_mesh = ngmesh
        mesh.netgen_flags = flags

        no = impl.create_lgmap(rdm)
        o = impl.create_lgmap(mesh.topology_dm)
        lgmaps.append((no, o))

        # Curve the mesh
        if order[l] != mesh.coordinates.function_space().ufl_element().degree():
            logger.info("\t\t\tCurving the mesh ...")
            tic = time.time()
            if snap == "geometry":
                coordinates = mesh.curve_field(
                    order=order[l],
                    permutation_tol=permutation_tol,
                    cg_field=cg,
                )
                mesh = reconstruct_mesh(mesh, coordinates)
            elif snap == "coarse":
                mesh = snapToCoarse(meshes[0].coordinates, mesh, order[l], snap_smoothing, cg)
            toc = time.time()
            logger.info(f"\t\t\tMeshed curved. Time taken: {toc-tic}")
        logger.info(f"\t\tLevel {l}: with {ngmesh.Coordinates().shape[0]}\
                vertices, with order {order[l]}, snapping to {snap}\
                and optimisation moves {optMoves}.")
        mesh.topology_dm.setRefineLevel(l)
        meshes.append(mesh)
    # Populate the coarse to fine map
    coarse_to_fine_cells, fine_to_coarse_cells = refinementTypes[refType][1](meshes, lgmaps)
    return fd.HierarchyBase(meshes, coarse_to_fine_cells, fine_to_coarse_cells, 1, nested=nested)


def reconstruct_mesh(mesh, *args, **kwargs):
    """Reconstruct a mesh."""
    kwargs.setdefault("dim", mesh.geometric_dimension)
    kwargs.setdefault("reorder", False)
    kwargs.setdefault("distribution_parameters", DISTRIBUTION_PARAMETERS_NOOP)
    kwargs.setdefault("comm", mesh.comm)
    kwargs.setdefault("tolerance", mesh.tolerance)
    kwargs.setdefault("perm_is", mesh._dm_renumbering)

    tmesh = fd.Mesh(*args, **kwargs)
    tmesh._distribution_parameters = mesh._distribution_parameters
    tmesh._did_reordering = mesh._did_reordering
    tmesh.netgen_mesh = mesh.netgen_mesh
    tmesh.netgen_flags = mesh.netgen_flags
    tmesh.sfBC_orig = mesh.sfBC_orig
    return tmesh
