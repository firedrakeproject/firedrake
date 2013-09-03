#!/usr/bin/env python
from optparse import OptionParser
import os
from subprocess import call
import sys

meshtemplate = """
Point(1) = {0, 0, 0, %(dx)f};
Extrude {1, 0, 0} {
  Point{1}; Layers{%(layers)d};
}
Extrude {0, 1, 0} {
  Line{1}; Layers{%(layers)d};
}
"""


def generate_meshfile(name, layers):
    with open(name + ".geo", 'w') as f:
        f.write(meshtemplate % {'dx': 1. / layers, 'layers': layers})

    meshdir, name = os.path.split(name)
    meshdir = meshdir if meshdir != "" else None
    call(["gmsh", "-2", name + ".geo"], cwd=meshdir)
    path = os.path.dirname(os.path.abspath(__file__))
    call([path + "/gmsh2triangle", "--2d", name + ".msh"], cwd=meshdir)


if __name__ == '__main__':
    optparser = OptionParser(usage='usage: %prog [options] <name> <layers>',
                             add_help_option=True,
                             description="""Generate the mesh files for a given
                             number of layers of elements in the channel.""")
    (options, argv) = optparser.parse_args()

    try:
        name = argv[0]
        layers = int(argv[1])
    except:
        optparser.print_help()
        sys.exit(1)

    generate_meshfile(name, layers)
