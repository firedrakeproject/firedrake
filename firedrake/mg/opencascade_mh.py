from firedrake import *
from .mesh import MeshHierarchy, HierarchyBase
from .interface import prolong

import numpy
import subprocess
import os
import warnings


__all__ = ("OpenCascadeMeshHierarchy",)


def OpenCascadeMeshHierarchy(stepfile, element_size, levels, comm=COMM_WORLD, distribution_parameters=None, callbacks=None, order=1, mh_constructor=MeshHierarchy, cache=True, verbose=True, gmsh="gmsh", project_refinements_to_cad=True, reorder=None):

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
    dim = 3 if cad.number_of_solids() > 0 else 2
    coarse = make_coarse_mesh(stepfile, cad, element_size, dim, comm=comm, distribution_parameters=distribution_parameters, cache=cache, verbose=verbose, gmsh=gmsh)

    mh = mh_constructor(coarse, levels, distribution_parameters=distribution_parameters, callbacks=callbacks, reorder=reorder)
    project_to_cad = project_mesh_to_cad_2d if dim == 2 else project_mesh_to_cad_3d

    if project_refinements_to_cad:
        for mesh in mh:
            project_to_cad(mesh, cad)
        mh.nested = False

    if order > 1:
        VFS = VectorFunctionSpace
        Ts = [
            Function(VFS(mesh, "CG", order)).interpolate(mesh.coordinates)
            for mesh in mh]
        ho_meshes = [Mesh(T) for T in Ts]
        from collections import defaultdict
        for i, m in enumerate(ho_meshes):
            m._shared_data_cache = defaultdict(dict)
            for k in mh[i]._shared_data_cache:
                if k != "hierarchy_physical_node_locations":
                    m._shared_data_cache[k] = mh[i]._shared_data_cache[k]
        mh = HierarchyBase(
            ho_meshes, mh.coarse_to_fine_cells,
            mh.fine_to_coarse_cells,
            refinements_per_level=mh.refinements_per_level, nested=mh.nested
        )
        if project_refinements_to_cad:
            for mesh in mh:
                project_to_cad(mesh, cad)
        else:
            project_to_cad(mh[0], cad)
            for i in range(1, len(mh)):
                prolong(Ts[i-1], Ts[i])
    return mh


def make_coarse_mesh(stepfile, cad, element_size, dim, comm=COMM_WORLD, distribution_parameters=None, cache=True, verbose=True, gmsh="gmsh"):

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
                """ % (element_size[0], element_size[1])
            elif isinstance(element_size, int) or isinstance(element_size, float):
                geostr += """
Mesh.CharacteristicLengthMin = %s;
Mesh.CharacteristicLengthMax = %s;
                """ % (element_size, element_size)
            elif isinstance(element_size, str):
                geostr += element_size
            else:
                raise NotImplementedError("element_size has to be a tuple, a number or a string")

            if dim == 2:
                for i in range(1, cad.number_of_edges()+1):
                    geostr += "Physical Line(%d) = {%d};\n" % (i, i)
                for i in range(1, cad.number_of_faces()+1):
                    geostr += ('Physical Surface(%d) = {%d};\n' % (i+cad.number_of_edges(), i))
                if cad.number_of_faces() > 1:
                    surfs = "".join(["Surface{%d}; " % i for i in range(2, cad.number_of_faces()+1)])
                    geostr += ('BooleanUnion{ Surface{1}; Delete;}{' + surfs + 'Delete;}')
            elif dim == 3:
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

            gmsh = subprocess.Popen(gmsh.split(" ") + ["-%d" % dim, geopath], stdout=stdout)
            gmsh.wait()

        comm.barrier()

    coarse = Mesh(mshpath, distribution_parameters=distribution_parameters, comm=comm)
    return coarse


def project_mesh_to_cad_3d(mesh, cad):

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
                warnings.warn("Projection of point %s onto face %d failed" % (coorddata[node, :], id))

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
                        warnings.warn("Projection of point %s onto curve failed" % coorddata[node, :])

                (projpt, sqdist) = min(projections, key=lambda x: x[1])
                coorddata[node, :] = projpt.Coord()


def project_mesh_to_cad_2d(mesh, cad):

    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve

    coorddata = mesh.coordinates.dat.data
    ids = mesh.exterior_facets.unique_markers

    filt = lambda arr: arr[numpy.where(arr < mesh.coordinates.dof_dset.size)[0]]
    boundary_nodes = {id: filt(mesh.coordinates.function_space().boundary_nodes(int(id), "topological")) for id in ids}

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
                warnings.warn("Projection of point %s onto curve failed" % coorddata[node, :])
