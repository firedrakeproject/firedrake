# Utility functions to derive global and local numbering from DMPlex
from petsc4py import PETSc
from pyop2 import MPI
import numpy as np
from operator import itemgetter

def facet_numbering(plex, vertex_numbering, facet):
    """Derive local facet number according to Fenics"""
    cells = plex.getSupport(facet)
    local_facet = []
    for c in cells:
        closure = plex.getTransitiveClosure(c)[0]

        # Local vertex numbering according to universal vertex numbering
        vStart, vEnd = plex.getDepthStratum(0)   # vertices
        is_vertex = lambda v: vStart <= v < vEnd
        vertices = filter(is_vertex, closure)
        v_glbl = [vertex_numbering.getOffset(v) for v in vertices]
        v_glbl = [v if v >= 0 else -(v+1) for v in v_glbl]
        vertices, v_glbl = zip(*sorted(zip(vertices, v_glbl), key=itemgetter(1)))

        # Local facet number := local number of non-incident vertex
        v_incident = filter(is_vertex, plex.getTransitiveClosure(facet)[0])
        v_non_incident = [v for v in vertices if v not in v_incident ][0]
        local_facet.append(np.where(vertices==v_non_incident)[0][0])
    return local_facet

def closure_numbering(plex, vertex_numbering, closure, dofs_per_entity):
    """Apply Fenics local numbering to a cell closure.

    Vertices    := Ordered according to global/universal
                   vertex numbering
    Edges/faces := Ordered according to lexicographical
                   ordering of non-incident vertices
    """
    dim = plex.getDimension()
    local_numbering = np.empty(len(closure), dtype=np.int32)
    vStart, vEnd = plex.getDepthStratum(0)   # vertice
    is_vertex = lambda v: vStart <= v < vEnd

    # Vertices := Ordered according to vertex numbering
    vertices = filter(is_vertex, closure)
    v_glbl = [vertex_numbering.getOffset(v) for v in vertices]

    # Plex defines non-owned universal numbers as negative,
    # correct with N = -(N+1)
    v_glbl = [v if v >= 0 else -(v+1) for v in v_glbl]

    vertices, v_glbl = zip(*sorted(zip(vertices, v_glbl), key=itemgetter(1)))
    # Correct 1D edge numbering
    if dim == 1:
        vertices = vertices[::-1]
    local_numbering[:len(vertices)] = vertices
    offset = len(vertices)

    # Local edge/face numbering := lexicographical ordering
    #                              of non-incident vertices

    for d in range(1, dim):
        pStart, pEnd = plex.getDepthStratum(d)
        points = filter(lambda p: pStart <= p < pEnd, closure)

        # Re-order edge/facet points only if they have DoFs associated
        if dofs_per_entity[d] > 0:
            v_lcl = []   # local no. of non-incident vertices
            for p in points:
                p_closure = plex.getTransitiveClosure(p)[0]
                v_incident = filter(is_vertex, p_closure)
                v_non_inc = [v for v in vertices if v not in v_incident ]
                v_lcl.append([np.where(vertices==v)[0][0] for v in v_non_inc])
            points, v_lcl = zip(*sorted(zip(points, v_lcl), key=itemgetter(1)))

        local_numbering[offset:offset+len(points)] = points
        offset += len(points)

    # Add the cell itself
    cStart, cEnd = plex.getHeightStratum(0)  # cells
    cells = filter(lambda c: cStart <= c < cEnd, closure)
    local_numbering[offset:offset+len(cells)] = cells
    return local_numbering

def mark_entity_classes(plex):
    """Mark all points in a given Plex according to the PyOP2 entity classes:
    core      : owned and not in send halo
    non_core  : owned and in send halo
    exec_halo : in halo, but touch owned entity
    """
    plex.createLabel("op2_core")
    plex.createLabel("op2_non_core")
    plex.createLabel("op2_exec_halo")

    if MPI.comm.size > 1:
        # Mark exec_halo from point overlap SF
        point_sf = plex.getPointSF()
        nroots, nleaves, local, remote = point_sf.getGraph()
        for p in local:
            depth = plex.getLabelValue("depth", p)
            plex.setLabelValue("op2_exec_halo", p, depth)
    else:
        # If sequential mark all points as core
        pStart, pEnd = plex.getChart()
        for p in range(pStart, pEnd):
            depth = plex.getLabelValue("depth", p)
            plex.setLabelValue("op2_core", p, depth)
        return

    # Mark all unmarked points in the closure of adjacent cells as non_core
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)
    dim = plex.getDimension()
    halo_cells = plex.getStratumIS("op2_exec_halo", dim).getIndices()
    halo_vertices = plex.getStratumIS("op2_exec_halo", 0).getIndices()
    adjacent_cells = []
    for c in halo_cells:
        halo_closure = plex.getTransitiveClosure(c)[0]
        for vertex in filter(lambda x: x>=vStart and x<vEnd, halo_closure):
            star = plex.getTransitiveClosure(vertex, useCone=False)[0]
            for adj in filter(lambda x: x>=cStart and x<cEnd, star):
                if plex.getLabelValue("op2_exec_halo", adj) < 0:
                    adjacent_cells.append(adj)

    for adj_cell in adjacent_cells:
        for p in plex.getTransitiveClosure(adj_cell)[0]:
            if plex.getLabelValue("op2_exec_halo", p) < 0:
                depth = plex.getLabelValue("depth", p)
                plex.setLabelValue("op2_non_core", p, depth)

    # Mark all remaining points as core
    pStart, pEnd = plex.getChart()
    for p in range(pStart, pEnd):
        exec_halo = plex.getLabelValue("op2_exec_halo", p)
        non_core = plex.getLabelValue("op2_non_core", p)
        if exec_halo < 0 and non_core < 0:
            depth = plex.getLabelValue("depth", p)
            plex.setLabelValue("op2_core", p, depth)

def get_entities_by_class(plex, depth, condition=None):
    """Get a list of Plex entities sorted by the PyOP2 entity classes"""
    entity_classes = [0, 0, 0 ,0]
    entities = np.array([], dtype=np.int32)
    if plex.getStratumSize("op2_core", depth) > 0:
        core = plex.getStratumIS("op2_core", depth).getIndices()
        if condition:
            core = filter(condition, core)
        entities = np.concatenate([entities, core])
    entity_classes[0] = entities.size
    if plex.getStratumSize("op2_non_core", depth) > 0:
        non_core = plex.getStratumIS("op2_non_core", depth).getIndices()
        if condition:
            non_core = filter(condition, non_core)
        entities = np.concatenate([entities, non_core])
    entity_classes[1] = entities.size
    if plex.getStratumSize("op2_exec_halo", depth) > 0:
        exec_halo = plex.getStratumIS("op2_exec_halo", depth).getIndices()
        if condition:
            exec_halo = filter(condition, exec_halo)
        entities = np.concatenate([entities, exec_halo])
    entity_classes[2] = entities.size
    entity_classes[3] = entities.size
    return entities, entity_classes

def permute_global_numbering(plex):
    """Permute the global/universal DoF numbering according to a
    depth-first traversal of the Plex graph."""
    dim = plex.getDimension()
    glbl = plex.getDefaultSection()
    univ = plex.getDefaultGlobalSection()
    pStart, pEnd = glbl.getChart()

    entity_classes = [0, 0, 0, 0]
    permutation = -1 * np.ones(pEnd-pStart, dtype=np.int)
    glbl_num = 0

    # Create new numbering sections
    glbl_new = PETSc.Section().create()
    glbl_new.setChart(pStart, pEnd)
    glbl_new.setUp()
    univ_new = PETSc.Section().create()
    univ_new.setChart(pStart, pEnd)
    univ_new.setUp()

    # Get a list of current universal DoFs
    universal_dofs = []
    for p in range(pStart, pEnd):
        for c in range(univ.getDof(p)):
            universal_dofs.append(univ.getOffset(p)+c)

    # Renumber core DoFs
    seen = set()
    core_is = plex.getStratumIS("op2_core", dim)
    if plex.getStratumSize("op2_core", dim) > 0:
        for cell in plex.getStratumIS("op2_core", dim).getIndices():
            for p in plex.getTransitiveClosure(cell)[0]:
                if p in seen: continue

                seen.add(p)
                dof = glbl.getDof(p)
                if dof > 0 and plex.getLabelValue("op2_core", p) >= 0:
                    glbl_new.setDof(p, dof)
                    glbl_new.setOffset(p, glbl_num)
                    univ_new.setDof(p, dof)
                    univ_new.setOffset(p, universal_dofs[glbl_num])
                    permutation[p] = glbl_num
                    glbl_num += dof
    entity_classes[0] = glbl_num

    # Renumber non-core DoFs
    seen = set()
    non_core_is = plex.getStratumIS("op2_non_core", dim)
    if plex.getStratumSize("op2_non_core", dim) > 0:
        for cell in plex.getStratumIS("op2_non_core", dim).getIndices():
            for p in plex.getTransitiveClosure(cell)[0]:
                if p in seen: continue

                seen.add(p)
                dof = glbl.getDof(p)
                if dof > 0 and plex.getLabelValue("op2_non_core", p) >= 0:
                    glbl_new.setDof(p, dof)
                    glbl_new.setOffset(p, glbl_num)
                    univ_new.setDof(p, dof)
                    univ_new.setOffset(p, universal_dofs[glbl_num])
                    permutation[p] = glbl_num
                    glbl_num += dof
    entity_classes[1] = glbl_num

    # We need to propagate the new global numbers for owned points to
    # all ranks to get the correct universal numbers (unn) for the halo.
    unn_global = plex.createGlobalVec()
    unn_global.assemblyBegin()
    for p in range(pStart,pEnd):
        if univ_new.getDof(p) > 0:
            unn_global.setValue(univ.getOffset(p), univ_new.getOffset(p))
    unn_global.assemblyEnd()
    unn_local = plex.createLocalVec()
    plex.globalToLocal(unn_global, unn_local)

    # Renumber exec-halo DoFs
    seen = set()
    halo_is = plex.getStratumIS("op2_exec_halo", dim)
    if plex.getStratumSize("op2_exec_halo", dim) > 0:
        for cell in plex.getStratumIS("op2_exec_halo", dim).getIndices():
            for p in plex.getTransitiveClosure(cell)[0]:
                if p in seen: continue

                seen.add(p)
                ldof = glbl.getDof(p)
                gdof = univ.getDof(p)
                if ldof > 0 and plex.getLabelValue("op2_exec_halo", p) >= 0:
                    glbl_new.setDof(p, ldof)
                    glbl_new.setOffset(p, glbl_num)
                    univ_new.setDof(p, gdof)
                    remote_unn = unn_local.getValue(glbl.getOffset(p))
                    univ_new.setOffset(p, -(remote_unn+1) )
                    permutation[p] = glbl_num
                    glbl_num += ldof
    entity_classes[2] = glbl_num

    # L2 halos not supported
    entity_classes[3] = glbl_num

    plex.setDefaultSection(glbl_new)
    plex.setDefaultGlobalSection(univ_new)
    return entity_classes, permutation
