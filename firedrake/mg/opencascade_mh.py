from firedrake import *
from .mesh import MeshHierarchy

import numpy
import subprocess
import os
import warnings


__all__ = ("OpenCascadeMeshHierarchy",)


def OpenCascadeMeshHierarchy(stepfile, mincoarseh, maxcoarseh, levels, comm=COMM_WORLD, distribution_parameters=None, callbacks=None, order=1, mh_constructor=MeshHierarchy, cache=True, verbose=True):

    # OpenCascade doesn't give a nice error message if stepfile
    # doesn't exist, it segfaults ...

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Extend.TopologyUtils import TopologyExplorer
    except ImportError:
        raise ImportError("To use OpenCascadeMeshHierarchy, you must install firedrake with the OpenCascade python bindings (firedrake-update --opencascade).")

    if not os.path.isfile(stepfile):
        raise OSError("%s does not exist" % stepfile)

    step_reader = STEPControl_Reader()
    step_reader.ReadFile(stepfile)
    step_reader.TransferRoot()
    shape = step_reader.Shape()
    cad = TopologyExplorer(shape)

    coarse = make_coarse_mesh(stepfile, cad, mincoarseh, maxcoarseh, comm=comm, distribution_parameters=distribution_parameters, cache=cache, verbose=verbose)

    if order > 1:
        V = VectorFunctionSpace(coarse, "CG", order)
        newcoords = Function(V)
        newcoords.interpolate(coarse.coordinates)
        coarse = Mesh(newcoords)

    mh = mh_constructor(coarse, levels, distribution_parameters=distribution_parameters, callbacks=callbacks)
    for mesh in mh:
        project_mesh_to_cad(mesh, cad)

    return mh


def make_coarse_mesh(stepfile, cad, mincoarseh, maxcoarseh, comm=COMM_WORLD, distribution_parameters=None, cache=True, verbose=True):

    curdir = os.path.dirname(stepfile) or os.getcwd()
    stepname = os.path.basename(os.path.splitext(stepfile)[0])
    geopath = os.path.join(curdir, "coarse-%s-%s-%s.geo" % (stepname, mincoarseh, maxcoarseh))
    mshpath = geopath.replace("geo", "msh")

    if not os.path.isfile(mshpath) or not cache:

        if comm.rank == 0:
            geostr = """
SetFactory("OpenCASCADE");

Mesh.CharacteristicLengthMin = %s;
Mesh.CharacteristicLengthMax = %s;

a() = ShapeFromFile("%s");
""" % (mincoarseh, maxcoarseh, os.path.abspath(stepfile))

            for i in range(1, cad.number_of_faces()+1):
                geostr += "Physical Surface(%d) = {%d};\n" % (i, i)

            geostr += ('Physical Volume("Combined volume", %d) = {a()};\n' % (cad.number_of_faces()+1))

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

            gmsh = subprocess.Popen(["gmsh", "-3", geopath], stdout=stdout)
            gmsh.wait()

        comm.barrier()

    coarse = Mesh(mshpath, distribution_parameters=distribution_parameters, comm=comm)
    return coarse


def push_coordinates_to_plex(mesh):
    plex = mesh._plex
    x = mesh.coordinates
    with x.dat.vec_ro as x_:
        plex.setCoordinates(x_)
        plex.setCoordinatesLocal(x_)


def project_mesh_to_cad(mesh, cad):

    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf, GeomAPI_ProjectPointOnCurve

    coorddata = mesh.coordinates.dat.data
    ids = mesh.exterior_facets.unique_markers

    filt = lambda arr: arr[numpy.where(arr < mesh.coordinates.dof_dset.size)[0]]
    boundary_nodes = {id: filt(mesh.coordinates.function_space().boundary_nodes(int(id), "topological")) for id in ids}

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
                warnings.warn("Projection of point %s onto face %d failed" % (coordata[node, :], id))

        edges = set(cad.edges_from_face(face))

        for (other_id, other_face) in zip(ids, cad.faces()):
            if other_id <= id:
                continue

            intersecting_nodes = numpy.intersect1d(boundary_nodes[id], boundary_nodes[other_id])
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
                warnings.warn("face: %s other_face: %s intersecting_edges: %s" % (face, other_face, intersecting_edges))
                warnings.warn("Warning: no intersecting edges in CAD, even though vertices on both faces?")
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
                        warnings.warn("Projection of point %s onto curve failed" % coordata[node, :])

                (projpt, sqdist) = min(projections, key=lambda x: x[1])
                coorddata[node, :] = projpt.Coord()

    push_coordinates_to_plex(mesh)
