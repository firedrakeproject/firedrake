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

"""PyOP2 Stupid MPI demo

This demo repeatidily computes the input mesh geometric center by two means
and scaling the mesh around its center.

The domain read in from a pickle dump.
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from cPickle import load
import gzip

from pyop2 import op2, utils


def main(opt):
    valuetype = np.float64

    f = gzip.open(opt['mesh'] + '.' + str(op2.MPI.comm.rank) + '.pickle.gz')

    elements, nodes, elem_node, coords = load(f)
    f.close()
    coords = op2.Dat(nodes ** 2, coords.data, np.float64, "coords")
    varea = op2.Dat(nodes, np.zeros((nodes.total_size, 1), valuetype), valuetype, "varea")

    mesh_center = op2.Kernel("""\
void
mesh_center(double* coords, double* center, int* count)
{
  center[0] += coords[0];
  center[1] += coords[1];
  *count += 1;
}""", "mesh_center")

    mesh_scale = op2.Kernel("""\
void
mesh_scale(double* coords, double* center, double* scale)
{
  coords[0] = (coords[0] - center[0]) * scale[0] + center[0];
  coords[1] = (coords[1] - center[1]) * scale[1] + center[1];
}""", "mesh_scale")

    elem_center = op2.Kernel("""\
void
elem_center(double* center, double* vcoords[3], int* count)
{
  center[0] += (vcoords[0][0] + vcoords[1][0] + vcoords[2][0]) / 3.0f;
  center[1] += (vcoords[0][1] + vcoords[1][1] + vcoords[2][1]) / 3.0f;
  *count += 1;
}""", "elem_center")

    dispatch_area = op2.Kernel("""\
void
dispatch_area(double* vcoords[3], double* area[3])
{
  double a = 0;
  a += vcoords[0][0] * ( vcoords[1][1] - vcoords[2][1] );
  a += vcoords[1][0] * ( vcoords[2][1] - vcoords[0][1] );
  a += vcoords[2][0] * ( vcoords[0][1] - vcoords[1][1] );
  a = fabs(a) / 6.0;

  *area[0] += a;
  *area[1] += a;
  *area[2] += a;
}""", "dispatch_area")

    collect_area = op2.Kernel("""\
void
collect_area(double* varea, double* area)
{
    *area += *varea;
}""", "collect_area")

    expected_area = 1.0
    for i, s in enumerate([[1, 2], [2, 1], [3, 3], [2, 5], [5, 2]]):
        center1 = op2.Global(2, [0.0, 0.0], valuetype, name='center1')
        center2 = op2.Global(2, [0.0, 0.0], valuetype, name='center2')
        node_count = op2.Global(1, [0], np.int32, name='node_count')
        elem_count = op2.Global(1, [0], np.int32, name='elem_count')
        scale = op2.Global(2, s, valuetype, name='scale')
        area = op2.Global(1, [0.0], valuetype, name='area')

        op2.par_loop(mesh_center, nodes,
                     coords(op2.READ),
                     center1(op2.INC),
                     node_count(op2.INC))
        center1.data[:] = center1.data[:] / node_count.data[:]

        op2.par_loop(elem_center, elements,
                     center2(op2.INC),
                     coords(op2.READ, elem_node),
                     elem_count(op2.INC))
        center2.data[:] = center2.data[:] / elem_count.data[:]

        op2.par_loop(mesh_scale, nodes,
                     coords(op2.RW),
                     center1(op2.READ),
                     scale(op2.READ))

        varea.data.fill(0.0)
        op2.par_loop(dispatch_area, elements,
                     coords(op2.READ, elem_node),
                     varea(op2.INC, elem_node))

        op2.par_loop(collect_area, nodes,
                     varea(op2.READ),
                     area(op2.INC))

        expected_area *= s[0] * s[1]

        if opt['print_output']:
            print "Rank: %d: [%f, %f] [%f, %f] |%f (%f)|" % \
                (op2.MPI.comm.rank,
                 center1.data[0], center1.data[1],
                 center2.data[0], center2.data[1],
                 area.data[0], expected_area)

        if opt['test_output']:
            assert_allclose(center1.data, [0.5, 0.5])
            assert_allclose(center2.data, center1.data)
            assert_almost_equal(area.data[0], expected_area)

if __name__ == '__main__':
    parser = utils.parser(group=True, description=__doc__)
    parser.add_argument('-m', '--mesh', required=True,
                        help='Base name of mesh pickle \
                              (excluding the process number and .pickle extension)')
    parser.add_argument('--print-output', action='store_true', help='Print output')
    parser.add_argument('--test-output', action='store_true', help='Test output')

    opt = vars(parser.parse_args())
    op2.init(**opt)

    if op2.MPI.comm.size != 3:
        print "Stupid demo only works on 3 processes"
        op2.MPI.comm.Abort(1)

    main(opt)
