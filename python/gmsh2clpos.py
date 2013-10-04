#!/usr/bin/env python

"""
Module for generating gmsh post processing files for use as background meshes.
Elements have the form:
  [[coords], data]
or
  [[coords], [data]]
"""

import sys


def ReadMsh(filename):
    """
    Reads a gmsh msh file. Returns a list of node coordinates, and a dictionary
    of elements each with data element equal to the physical group ID.

    Uses code from gmsh2triangle.
    """

    mshfile = file(filename, "r")

    # Header section
    assert(mshfile.readline().strip() == "$MeshFormat")
    assert(mshfile.readline().strip() == "2 0 8")
    assert(mshfile.readline().strip() == "$EndMeshFormat")

    # Nodes section
    assert(mshfile.readline().strip() == "$Nodes")
    nodecount = int(mshfile.readline())
    nodes = {}
    for i in range(nodecount):
        node = mshfile.readline()
        nodes[node.split()[0]] = [float(coord) for coord in node.split()[1:]]
    assert(mshfile.readline().strip() == "$EndNodes")

    # Elements section
    assert(mshfile.readline().strip() == "$Elements")
    elementcount = int(mshfile.readline())
    elements = {}
    elements["linear_edges"] = []
    elements["linear_triangles"] = []
    elements["linear_quads"] = []
    elements["linear_tets"] = []
    elements["linear_hexes"] = []
    for i in range(elementcount):
        element = mshfile.readline().split()
        if element[1] == "1":
            elements["linear_edges"].append([element[-2:], element[3]])
        elif element[1] == "2":
            elements["linear_triangles"].append([element[-3:], element[3]])
        elif element[1] == "3":
            elements["linear_quads"].append([element[-4:], element[3]])
        elif element[1] == "4":
            elements["linear_tets"].append([element[-4:], element[3]])
        elif element[1] == "5":
            elements["linear_hexes"].append([element[-8:], element[3]])
        elif element[1] == "15":
            pass
        else:
            sys.stderr.write("Warning - Unknown element of type " +
                             element[1] + " encountered in gmsh msh file \"" +
                             filename + "\"\n")
            pass
    assert(mshfile.readline().strip() == "$EndElements")

    if not len(elements["linear_triangles"]) == 0 \
            or not len(elements["linear_tets"]) == 0:
        if not len(elements["linear_quads"]) == 0 \
                or not len(elements["linear_hexes"]) == 0:
            sys.stderr.write("Warning - gmsh msh file \"" + filename +
                             "\" contains a mix of triangles/tets and quads/hexes\n")

    mshfile.close()

    return nodes, elements


def ElementCoords(nodes, element):
    """
    Returns a list of node coordinates for the supplied element.
    """

    return [nodes[i] for i in element[0]]


def ElementData(element):
    """
    Returns a list of data for the supplied element
    """

    if isinstance(element[1], list):
        return element[1]
    else:
        return [element[1]]


def GenerateClPos(clfunc, nodes, elements={}):
    """Replaces the data in all supplied elements with characteristic lengths
    as defined by the supplied characteristic length function clfunc. clfunc
    must be of the form:

      clfunc(x)

    where x is a list defining a spatial coordinate.
    """

    newElements = {}

    for key in elements.keys():
        newElements[key] = []
        for element in elements[key]:
            newElements[key].append([element[0], [clfunc(coord) for coord in
                                                  ElementCoords(nodes, element)]])

    return newElements


def WriteClPos(filename, clfunc, nodes, elements):
    """Generates a gmsh post processing file suitable for use as a background
    mesh, with characteristic lengths as defined by the supplied characteristic
    length function clfunc. See GenerateClPos."""

    def ElementCoordsAndData(nodes, element):
        coordsAndData = "("
        elementCoords = ElementCoords(nodes, element)
        for i in range(len(elementCoords)):
            for j in range(len(elementCoords[i])):
                coordsAndData += str(elementCoords[i][j])
                if j < len(elementCoords[i]) - 1:
                    coordsAndData += ", "
            if i < len(elementCoords) - 1:
                coordsAndData += ", "
        coordsAndData += ") {"
        data = ElementData(element)
        for i in range(len(data)):
            coordsAndData += str(data[i])
            if i < len(data) - 1:
                coordsAndData += ", "
        coordsAndData += "}"

        return coordsAndData

    elements = GenerateClPos(clfunc, nodes, elements)

    posfile = file(filename, "w")

    posfile.write("// Produced by gmsh2clpos\n" +
                  "View \"CharactisticLengths\"\n" +
                  "{\n")

    for key in elements.keys():
        if key == "linear_edges":
            for linear_edge in elements["linear_edges"]:
                posfile.write(
                    "  SL" + ElementCoordsAndData(nodes, linear_edge) + ";\n")
        elif key == "linear_triangles":
            for linear_triangle in elements["linear_triangles"]:
                posfile.write(
                    "  ST" + ElementCoordsAndData(nodes, linear_triangle) + ";\n")
        elif key == "linear_quads":
            for linear_quad in elements["linear_quads"]:
                posfile.write(
                    "  SQ" + ElementCoordsAndData(nodes, linear_quad) + ";\n")
        elif key == "linear_tets":
            for linear_tet in elements["linear_tets"]:
                posfile.write(
                    "  SS" + ElementCoordsAndData(nodes, linear_tet) + ";\n")
        elif key == "linear_hexes":
            for linear_hex in elements["linear_hexes"]:
                posfile.write(
                    "  SH" + ElementCoordsAndData(nodes, linear_hex) + ";\n")
        else:
            sys.stderr.write("Warning: Unknown element of type \"" + str(
                key) + "\" encountered in element dictionary\n")
            pass

    posfile.write("};\n")

    posfile.flush()
    posfile.close()

    return


def Gmsh2ClPos(input_filename, clfunc, output_filename):
    """Reads the gmsh mesh file specified by input_filename and generates a
    gmsh post processing file suitable for use as a background mesh."""

    nodes, elements = ReadMsh(input_filename)
    WriteClPos(output_filename, clfunc, nodes, elements)

    return
