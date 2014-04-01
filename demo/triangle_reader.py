# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Provides functions for reading triangle files into OP2 data structures."""

from pyop2 import op2
import numpy as np


def read_triangle(f, layers=None):
    """Read the triangle file with prefix f into OP2 data strctures. Presently
    only .node and .ele files are read, attributes are ignored, and there may
    be bugs. The dat structures are returned as:

        (nodes, coords, elements, elem_node)

    These items have type:

        (Set, Dat, Set, Map)

    The Layers argument allows the reading of data for extruded meshes.
    It is to be used when dealing with extruded meshes.
    """
    # Read nodes
    with open(f + '.node') as h:
        num_nodes = int(h.readline().split(' ')[0])
        node_values = np.zeros((num_nodes, 2), dtype=np.float64)
        for line in h:
            if line[0] == '#':
                continue
            node, x, y = line.split()[:3]
            node_values[int(node) - 1, :] = [float(x), float(y)]

    nodes = op2.Set(num_nodes, "nodes")
    coords = op2.Dat(nodes ** 2, node_values, name="coords")

    # Read elements
    with open(f + '.ele') as h:
        num_tri, nodes_per_tri, num_attrs = [int(col) for col in h.readline().split()]
        map_values = np.zeros((num_tri, nodes_per_tri), dtype=np.int32)
        for line in h:
            if line[0] == '#':
                continue
            vals = [int(v) - 1 for v in line.split()]
            map_values[vals[0], :] = vals[1:nodes_per_tri + 1]

    if layers is not None:
        elements = op2.ExtrudedSet(op2.Set(num_tri, "elements"), layers=layers)
    else:
        elements = op2.Set(num_tri, "elements")
    elem_node = op2.Map(elements, nodes, nodes_per_tri, map_values, "elem_node")

    return nodes, coords, elements, elem_node
