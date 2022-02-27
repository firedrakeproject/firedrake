from firedrake import *
from .mesh import MeshHierarchy, HierarchyBase
from .interface import prolong

import numpy
import subprocess
import os
import warnings


__all__ = ("OpenCascadeMeshHierarchy",)


def OpenCascadeMeshHierarchy(
    stepfile,
    levels,
    element_size=None,
    meshfile=None,
    comm=COMM_WORLD,
    distribution_parameters=None,
    callbacks=None,
    order=1,
    mh_constructor=MeshHierarchy,
    cache=True,
    verbose=True,
    gmsh="gmsh",
    project_refinements_to_cad=True,
    reorder=None,
):

    # OpenCascade doesn't give a nice error message if stepfile
    # doesn't exist, it segfaults ...

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Extend.TopologyUtils import TopologyExplorer
    except ImportError:
        raise ImportError(
            "To use OpenCascadeMeshHierarchy, you must install firedrake with the OpenCascade python bindings (firedrake-update --opencascade)."
        )

    if not os.path.isfile(stepfile):
        raise OSError("%s does not exist" % stepfile)

    step_reader = STEPControl_Reader()
    step_reader.ReadFile(stepfile)
    step_reader.TransferRoot()
    shape = step_reader.Shape()
    cad = TopologyExplorer(shape)
    dim = 3 if cad.number_of_solids() > 0 else 2
    if meshfile:
        coarse = Mesh(
            meshfile, comm=comm, distribution_parameters=distribution_parameters
        )
        # create_coloring(coarse, cad)
        project_to_cad = (
            project_mesh_to_cad_2dx if dim == 2 else project_mesh_to_cad_3dx
        )
    else:
        coarse = make_coarse_mesh(
            stepfile,
            cad,
            element_size,
            dim,
            comm=comm,
            distribution_parameters=distribution_parameters,
            cache=cache,
            verbose=verbose,
            gmsh=gmsh,
        )
        project_to_cad = project_mesh_to_cad_2d if dim == 2 else project_mesh_to_cad_3d

    mh = mh_constructor(
        coarse,
        levels,
        distribution_parameters=distribution_parameters,
        callbacks=callbacks,
        reorder=reorder,
    )

    if project_refinements_to_cad:
        for mesh in mh:
            project_to_cad(mesh, cad)
        mh.nested = False

    if order > 1:
        VFS = VectorFunctionSpace
        Ts = [
            Function(VFS(mesh, "CG", order)).interpolate(mesh.coordinates)
            for mesh in mh
        ]
        ho_meshes = [Mesh(T) for T in Ts]
        mh = HierarchyBase(
            ho_meshes,
            mh.coarse_to_fine_cells,
            mh.fine_to_coarse_cells,
            refinements_per_level=mh.refinements_per_level,
            nested=mh.nested,
        )
        if project_refinements_to_cad:
            for mesh in mh:
                project_to_cad(mesh, cad)
        else:
            project_to_cad(mh[0], cad)
            for i in range(1, len(mh)):
                prolong(Ts[i - 1], Ts[i])
    return mh


def make_coarse_mesh(
    stepfile,
    cad,
    element_size,
    dim,
    comm=COMM_WORLD,
    distribution_parameters=None,
    cache=True,
    verbose=True,
    gmsh="gmsh",
):

    curdir = os.path.dirname(stepfile) or os.getcwd()
    stepname = os.path.basename(os.path.splitext(stepfile)[0])
    geopath = os.path.join(curdir, "coarse-%s.geo" % stepname)
    mshpath = os.path.join(curdir, "coarse-%s.msh" % stepname)

    if not os.path.isfile(mshpath) or not cache:

        if comm.rank == 0:
            geostr = 'SetFactory("OpenCASCADE");\n'
            geostr += 'a() = ShapeFromFile("%s");\n' % os.path.abspath(stepfile)
            if isinstance(element_size, tuple):
                assert len(element_size) == 2

                geostr += """
Mesh.CharacteristicLengthMin = %s;
Mesh.CharacteristicLengthMax = %s;
                """ % (
                    element_size[0],
                    element_size[1],
                )
            elif isinstance(element_size, int) or isinstance(element_size, float):
                geostr += """
Mesh.CharacteristicLengthMin = %s;
Mesh.CharacteristicLengthMax = %s;
                """ % (
                    element_size,
                    element_size,
                )
            elif isinstance(element_size, str):
                geostr += element_size
            else:
                raise NotImplementedError(
                    "element_size has to be a tuple, a number or a string"
                )

            if dim == 2:
                for i in range(1, cad.number_of_edges() + 1):
                    geostr += "Physical Line(%d) = {%d};\n" % (i, i)
                for i in range(1, cad.number_of_faces() + 1):
                    geostr += "Physical Surface(%d) = {%d};\n" % (
                        i + cad.number_of_edges(),
                        i,
                    )
                if cad.number_of_faces() > 1:
                    surfs = "".join(
                        [
                            "Surface{%d}; " % i
                            for i in range(2, cad.number_of_faces() + 1)
                        ]
                    )
                    geostr += "BooleanUnion{ Surface{1}; Delete;}{" + surfs + "Delete;}"
            elif dim == 3:
                for i in range(1, cad.number_of_faces() + 1):
                    geostr += "Physical Surface(%d) = {%d};\n" % (i, i)

                geostr += 'Physical Volume("Combined volume", %d) = {a()};\n' % (
                    cad.number_of_faces() + 1
                )

            logging.debug(geostr)

            with open(geopath, "w") as f:
                f.write(geostr)

            try:
                os.remove(mshpath)
            except OSError:
                pass

            if verbose:
                stdout = None
            else:
                stdout = subprocess.DEVNULL

            gmsh = subprocess.Popen(
                gmsh.split(" ") + ["-%d" % dim, geopath], stdout=stdout
            )
            gmsh.wait()

        comm.barrier()

    coarse = Mesh(mshpath, distribution_parameters=distribution_parameters, comm=comm)
    return coarse



def project_mesh_to_cad_2d(mesh, cad):

    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve

    coorddata = mesh.coordinates.dat.data
    ids = mesh.exterior_facets.unique_markers

    filt = lambda arr: arr[numpy.where(arr < mesh.coordinates.dof_dset.size)[0]]
    boundary_nodes = {
        id: filt(mesh.coordinates.function_space().boundary_nodes(int(id)))
        for id in ids
    }

    for (id, edge) in zip(ids, cad.edges()):
        owned_nodes = boundary_nodes[id]
        for other_id in ids:
            if id == other_id:
                continue
            owned_nodes = numpy.setdiff1d(owned_nodes, boundary_nodes[other_id])

        curve = BRepAdaptor_Curve(edge)

        for node in owned_nodes:
            pt = gp_Pnt(*coorddata[node, :], 0)
            proj = GeomAPI_ProjectPointOnCurve(pt, curve.Curve().Curve())
            if proj.NbPoints() > 0:
                projpt = proj.NearestPoint()
                coorddata[node, :] = projpt.Coord()[0:2]
            else:
                warnings.warn(
                    "Projection of point %s onto curve failed" % coorddata[node, :]
                )

def project_mesh_to_cad_3d(mesh, cad):

    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf, GeomAPI_ProjectPointOnCurve

    coorddata = mesh.coordinates.dat.data
    ids = mesh.exterior_facets.unique_markers

    filt = lambda arr: arr[numpy.where(arr < mesh.coordinates.dof_dset.size)[0]]
    boundary_nodes = {
        id: filt(mesh.coordinates.function_space().boundary_nodes(int(id)))
        for id in ids
    }

    for (id, face) in zip(ids, cad.faces()):
        owned_nodes = boundary_nodes[id]
        for other_id in ids:
            if id == other_id:
                continue
            owned_nodes = numpy.setdiff1d(owned_nodes, boundary_nodes[other_id])

        surf = BRepAdaptor_Surface(face)

        for node in owned_nodes:
            pt = gp_Pnt(*coorddata[node, :])

            proj = GeomAPI_ProjectPointOnSurf(pt, surf.Surface().Surface())
            if proj.NbPoints() > 0:
                projpt = proj.NearestPoint()
                coorddata[node, :] = projpt.Coord()
            else:
                warnings.warn(
                    "Projection of point %s onto face %d failed"
                    % (coorddata[node, :], id)
                )

        edges = set(cad.edges_from_face(face))

        for (other_id, other_face) in zip(ids, cad.faces()):
            if other_id <= id:
                continue

            intersecting_nodes = numpy.intersect1d(
                boundary_nodes[id], boundary_nodes[other_id]
            )
            if len(intersecting_nodes) == 0:
                continue

            other_edges = set(cad.edges_from_face(other_face))

            intersecting_edges = []
            for edge in edges:
                s = str(edge)  # FIXME: is there a more elegant way to get the OCC id?
                for other_edge in other_edges:
                    other_s = str(other_edge)
                    if s == other_s:
                        intersecting_edges.append(edge)

            if len(intersecting_edges) == 0:
                warnings.warn(
                    "face: %s other_face: %s intersecting_edges: %s"
                    % (face, other_face, intersecting_edges)
                )
                warnings.warn(
                    "Warning: no intersecting edges in CAD, even though vertices on both faces?"
                )
                continue

            for node in intersecting_nodes:
                pt = gp_Pnt(*coorddata[node, :])

                projections = []
                for edge in intersecting_edges:
                    curve = BRepAdaptor_Curve(edge)

                    proj = GeomAPI_ProjectPointOnCurve(pt, curve.Curve().Curve())
                    if proj.NbPoints() > 0:
                        projpt = proj.NearestPoint()
                        sqdist = projpt.SquareDistance(pt)
                        projections.append((projpt, sqdist))
                    else:
                        warnings.warn(
                            "Projection of point %s onto curve failed"
                            % coorddata[node, :]
                        )

                (projpt, sqdist) = min(projections, key=lambda x: x[1])
                coorddata[node, :] = projpt.Coord()


def project_mesh_to_cad_2dx(mesh, cad):
    """
    Slight improvement to project_mesh_to_cad_2d.
    Sometimes when projecting 2D meshes and letting open cascade decide which
    edges it should project to causes strange distortions.
    Rather than relying on the closest projection. This method loops through all
    edges and chooses to project to the edge that moves the point the smallest distance.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve

    coorddata = mesh.coordinates.dat.data
    ids = mesh.exterior_facets.unique_markers

    filt = lambda arr: arr[numpy.where(arr < mesh.coordinates.dof_dset.size)[0]]
    boundary_nodes = {
        id: filt(mesh.coordinates.function_space().boundary_nodes(int(id)))
        for id in ids
    }

    for id_ in ids:
        for node in boundary_nodes[id_]:
            # print(node)
            coords = coorddata[node, :]
            best_coords = coords
            dist_old = np.inf
            for edge in cad.edges():
                curve = BRepAdaptor_Curve(edge)
                pt = gp_Pnt(*coorddata[node, :], 0)
                proj = GeomAPI_ProjectPointOnCurve(pt, curve.Curve().Curve())
                if proj.NbPoints() > 0:
                    projpt = proj.NearestPoint()
                    projected_coords = np.array(projpt.Coord()[0:2])
                    dist = np.linalg.norm(coords - projected_coords)
                    if dist_old > dist:
                        best_coords = projected_coords
                        dist_old = dist
            coorddata[node, :] = best_coords
    return

def project_mesh_to_cad_3dx(mesh, cad):
    """
    Similar to the 2D version.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf, GeomAPI_ProjectPointOnCurve

    coorddata = mesh.coordinates.dat.data
    ids = mesh.exterior_facets.unique_markers

    boundary_nodes = mesh.coordinates.function_space().boundary_nodes("on_boundary")
    for node in boundary_nodes:
        candidate_faces = []
        og_coord = coorddata[node, :]
        pt = gp_Pnt(*og_coord)
        for face in cad.faces():
            surf = BRepAdaptor_Surface(face)
            # Check all faces
            proj = GeomAPI_ProjectPointOnSurf(pt, surf.Surface().Surface())
            if proj.NbPoints() > 0:
                projpt = proj.NearestPoint()
                projected_coordinate = projpt.Coord()
                candidate_faces.append(
                    (
                        projected_coordinate,
                        np.linalg.norm(og_coord - projected_coordinate),
                        face,
                    )
                )
        candidate_faces.sort(key=lambda x: x[1])
        # Assuming if three faces intersect then they can only do so at a point
        candidate_faces = candidate_faces[:2]
        intersecting_edges = []
        for edge in cad.edges_from_face(candidate_faces[0][2]):
            s = str(edge)
            for other_edge in cad.edges_from_face(candidate_faces[1][2]):
                other_s = str(other_edge)
                if s == other_s:
                    intersecting_edges.append(edge)
        if len(intersecting_edges) == 0:
            coorddata[node, :] = candidate_faces[0][0]
        elif len(intersecting_edges) == 1:
            edge = intersecting_edges[0]
            curve = BRepAdaptor_Curve(edge)
            proj = GeomAPI_ProjectPointOnCurve(pt, curve.Curve().Curve())
            candidate_edges = []
            if proj.NbPoints() > 0:
                projpt = proj.NearestPoint()
                projected_coordinate = projpt.Coord()
                candidate_edges.append(
                    (
                        projected_coordinate,
                        np.linalg.norm(og_coord - projected_coordinate),
                        edge,
                    )
                )
            # print("Distance_to", candidate_faces[0][1], candidate_faces[1][1],candidate_edges[0][1])
            # coorddata[node, :] = candidate_edges[0][0]
        else:
            warnings.warn("Warning: Too many edges are intersecting")

        coorddata[node, :] = candidate_faces[0][0]

    return



def project_mesh_to_cad_3d_from_coloring(mesh, cad):
    """
    # FIXME (em):
    Boundary projection needs more information than distances to boundaries.
    A greedy projection will lead to elements with zero volume near edges.
    Potential fix is to color the coarse plex and then propagate the labels
    to the finer meshes. This will tells us exactly where we need to project.
    Note that this feature already exists in petsc but requires additional
    libraries (egads, egadslite)
    """
    pass


def create_coloring(mesh, cad):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf, GeomAPI_ProjectPointOnCurve

    plex = mesh.topology_dm
    coords = plex.getCoordinatesLocal()
    sec = plex.getCoordinateSection()
    label_face = "face_coloring"
    label_edge = "edge_coloring"
    plex.createLabel(label_face)
    plex.createLabel(label_edge)

    nfaces = plex.getStratumSize("exterior_facets", 1)
    assert nfaces != 0
    faces = plex.getStratumIS("exterior_facets", 1).getIndices()
    num_cadfaces = cad.number_of_faces()
    num_edges = cad.number_of_edges()
    for face in faces[:-1]:
        vertices = plex.vecGetClosure(sec, coords, face).reshape((3, 3))
        normal = np.cross(
            vertices[1, :] - vertices[0, :], vertices[2, :] - vertices[0, :]
        )
        normal = normal / np.linalg.norm(normal)
        vertex = vertices.mean(axis=0)  # Midpoint of faces
        pt = gp_Pnt(*vertex)
        projections = []
        for id_, cadface in enumerate(cad.faces()):
            surf = BRepAdaptor_Surface(cadface)
            proj = GeomAPI_ProjectPointOnSurf(pt, surf.Surface().Surface())
            if proj.NbPoints() > 0:
                projpt = proj.NearestPoint()
                sqdist = projpt.SquareDistance(pt)
                projections.append((projpt.Coord(), sqdist, id_))
        projections.sort(key=lambda x: x[1])
        projections = projections[:2]  # How many faces should we check?
        tol = 1e-6
        for candidate in projections:
            v1 = candidate[0] - vertex
            # FIXME: Point lies on face or we project in the normal direction
            if candidate[1] < 1e-16:
                plex.setLabelValue(label_face, face, candidate[2])
                break
            elif abs(np.dot(v1 / np.linalg.norm(v1), normal)) > tol:
                plex.setLabelValue(label_face, face, candidate[2])
                break

    return
