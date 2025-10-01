'''
This module contains all the functions related
'''
try:
    import firedrake as fd
    from firedrake.cython import mgimpl as impl, dmcommon
    from firedrake.__future__ import interpolate
    from firedrake import dmhooks
    import ufl
except ImportError:
    fd = None

from fractions import Fraction
import numpy as np
from petsc4py import PETSc

from netgen.meshing import MeshingParameters

from ngsPETSc.utils.firedrake.meshes import flagsUtils

def snapToNetgenDMPlex(ngmesh, petscPlex):
    '''
    This function snaps the coordinates of a DMPlex mesh to the coordinates of a Netgen mesh.
    '''
    if petscPlex.getDimension() == 2:
        ngCoordinates = ngmesh.Coordinates()
        petscCoordinates = petscPlex.getCoordinatesLocal().getArray().reshape(-1, ngmesh.dim)
        for i, pt in enumerate(petscCoordinates):
            j = np.argmin(np.sum((ngCoordinates - pt)**2, axis=1))
            petscCoordinates[i] = ngCoordinates[j]
        petscPlexCoordinates = petscPlex.getCoordinatesLocal()
        petscPlexCoordinates.setArray(petscPlexCoordinates)
        petscPlex.setCoordinatesLocal(petscPlexCoordinates)
    else:
        raise NotImplementedError("Snapping to Netgen meshes is only implemented for 2D meshes.")

def snapToCoarse(coarse, linear, degree, snap_smoothing, cg):
    '''
    This function snaps the coordinates of a DMPlex mesh to the coordinates of a Netgen mesh.
    '''
    dim = linear.geometric_dimension()
    if dim == 2:
        space = fd.VectorFunctionSpace(linear, "CG", degree)
        ho = fd.assemble(interpolate(coarse, space))
        if snap_smoothing == "hyperelastic":
            #Hyperelastic Smoothing
            bcs = [fd.DirichletBC(space, ho, "on_boundary")]
            quad_degree = 2*(degree+1)-1
            d = linear.topological_dimension()
            Q = fd.TensorFunctionSpace(linear, "DG", degree=0)
            Jinv = ufl.JacobianInverse(linear)
            hinv = fd.Function(Q)
            hinv.interpolate(Jinv)
            G = ufl.Jacobian(linear) * hinv
            ijac = 1/abs(ufl.det(G))

            def ref_grad(u):
                return ufl.dot(ufl.grad(u),G)

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
            element = ho.function_space().ufl_element().sub_elements[0].reconstruct(degree=degree)
            space = fd.VectorFunctionSpace(linear, fd.BrokenElement(element))
            ho = fd.Function(space).interpolate(ho)
    else:
        raise NotImplementedError("Snapping to Netgen meshes is only implemented for 2D meshes.")
    return fd.Mesh(ho, comm=linear.comm, distribution_parameters=linear._distribution_parameters)

def uniformRefinementRoutine(ngmesh, cdm):
    '''
    Routing called inside of NetgenHierarchy to compute refined ngmesh and plex.
    '''
    #We refine the netgen mesh uniformly
    ngmesh.Refine(adaptive=False)
    #We refine the DMPlex mesh uniformly
    cdm.setRefinementUniform(True)
    rdm = cdm.refine()
    rdm.removeLabel("pyop2_core")
    rdm.removeLabel("pyop2_owned")
    rdm.removeLabel("pyop2_ghost")
    return (rdm, ngmesh)

def uniformMapRoutine(meshes, lgmaps):
    '''
    This function computes the coarse to fine and fine to coarse maps
    for a uniform mesh hierarchy.
    '''
    refinements_per_level = 1
    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    for (coarse, fine), (clgmaps, flgmaps) in zip(zip(meshes[:-1], meshes[1:]),
                                                zip(lgmaps[:-1], lgmaps[1:])):
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
    #We refine the netgen mesh alfeld
    ngmesh.SplitAlfeld()
    #We refine the DMPlex mesh alfeld
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

def NetgenHierarchy(mesh, levs, flags):
    '''
    This function creates a Firedrake mesh hierarchy from Netgen/NGSolve meshes.

    :arg mesh: the Netgen/NGSolve mesh
    :arg levs: the number of levels in the hierarchy
    :arg netgen_flags: either a bool or a dictionray containing options for Netgen.
    If not False the hierachy is constructed using ngsPETSc, if None hierarchy
    constructed in a standard manner. Netgen flags includes:
        -degree, either an integer denoting the degree of curvature of all levels of
        the mesh or a list of levs+1 integers denoting the degree of curvature of
        each level of the mesh.
        -tol, geometric tolerance adopted in snapToNetgenDMPlex.
        -refinement_type, the refinment type to be used: uniform (default), Alfeld
    '''
    if mesh.geometric_dimension() == 3:
        raise NotImplementedError("Netgen hierachies are only implemented for 2D meshes.")
    comm = mesh.comm
    #Parsing netgen flags
    if not isinstance(flags, dict):
        flags = {}
    order = flagsUtils(flags, "degree", 1)
    if isinstance(order, int):
        order= [order]*(levs+1)
    permutation_tol = flagsUtils(flags, "tol", 1e-8)
    refType = flagsUtils(flags, "refinement_type", "uniform")
    optMoves = flagsUtils(flags, "optimisation_moves", False)
    snap = flagsUtils(flags, "snap_to", "geometry")
    snap_smoothing = flagsUtils(flags, "snap_smoothing", "hyperelastic")
    cg = flagsUtils(flags, "cg", False)
    nested = flagsUtils(flags, "nested", snap in ["coarse"])
    #Firedrake quoantities
    meshes = []
    lgmaps = []
    params = {"partition": False}
    #We curve the mesh
    if order[0]>1:
        ho_field = mesh.curve_field(
            order=order[0],
            permutation_tol=permutation_tol,
            cg_field=cg
        )
        temp = fd.Mesh(ho_field,distribution_parameters=params, comm=comm)
        temp.netgen_mesh = mesh.netgen_mesh
        temp._tolerance = mesh.tolerance
        mesh = temp
    # Make a plex (cdm) without overlap.
    dm_cell_type, = mesh.dm_cell_types
    tdim = mesh.topology_dm.getDimension()
    cdm = dmcommon.submesh_create(mesh.topology_dm, tdim, "celltype", dm_cell_type, True)
    cdm.removeLabel("pyop2_core")
    cdm.removeLabel("pyop2_owned")
    cdm.removeLabel("pyop2_ghost")
    no = impl.create_lgmap(cdm)
    o = impl.create_lgmap(mesh.topology_dm)
    lgmaps.append((no, o))
    mesh.topology_dm.setRefineLevel(0)
    meshes += [mesh]
    ngmesh = mesh.netgen_mesh
    for l in range(levs):
        #Streightening the mesh
        ngmesh.Curve(1)
        rdm, ngmesh = refinementTypes[refType][0](ngmesh, cdm)
        cdm = rdm
        #We snap the mesh to the Netgen mesh
        if snap == "geometry":
            snapToNetgenDMPlex(ngmesh, rdm)
        #We construct a Firedrake mesh from the DMPlex mesh
        no = impl.create_lgmap(rdm)
        mesh = fd.Mesh(rdm, dim=meshes[-1].geometric_dimension(), reorder=False,
                       distribution_parameters=params, comm=comm)
        o = impl.create_lgmap(mesh.topology_dm)
        lgmaps.append((no, o))
        if optMoves:
            #Optimises the mesh, for example smoothing
            if ngmesh.dim == 2:
                ngmesh.OptimizeMesh2d(MeshingParameters(optimize2d=optMoves))
            elif mesh.dim == 3:
                ngmesh.OptimizeVolumeMesh(MeshingParameters(optimize3d=optMoves))
            else:
                raise ValueError("Only 2D and 3D meshes can be optimised.")
        mesh.netgen_mesh = ngmesh
        #We curve the mesh
        if order[l+1] > 1:
            if snap == "geometry":
                mesh = fd.Mesh(
                    mesh.curve_field(order=order[l+1], permutation_tol=permutation_tol),
                    distribution_parameters=params,
                    comm=comm
                )
            elif snap == "coarse":
                mesh = snapToCoarse(ho_field, mesh, order[l+1], snap_smoothing, cg)
        mesh.topology_dm.setRefineLevel(1 + l)
        meshes += [mesh]
    #We populate the coarse to fine map
    coarse_to_fine_cells, fine_to_coarse_cells = refinementTypes[refType][1](meshes, lgmaps)
    return fd.HierarchyBase(meshes, coarse_to_fine_cells, fine_to_coarse_cells,
                            1, nested=nested)
