from finat.ufl.elementlist import ufl_elements
# ~ from ufl.finiteelement.elementlist import ufl_elements
from finat.element_factory import supported_elements
import csv

shape_names = {
    0: 'scalar',
    1: 'vector',
    2: 'tensor'
    }

def cells(cellnames):

    firedrake_cells = ("interval",
                       "triangle",
                       "tetrahedron",
                       "quadrilateral",
                       "hexahedron")

    cells = [c for c in cellnames if c in firedrake_cells]

    return(", ".join(cells))


with open("element_list.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    for element in supported_elements:
        family, short_name, value_rank, sobolev_space, \
            mapping, degree_range, cellnames = ufl_elements[element]

        short_name = short_name if short_name != family else ""
        cellnames = cells(cellnames)
        shape = shape_names[value_rank]

        csvwriter.writerow((family, short_name, shape, cellnames))
