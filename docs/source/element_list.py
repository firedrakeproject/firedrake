from finat.ufl.elementlist import ufl_elements
from finat.element_factory import supported_elements, create_element
from finat.ufl import FiniteElement
from ufl.cell import TensorProductCell
from ufl import interval, quadrilateral
import csv

shape_names = {
    0: 'scalar',
    1: 'vector',
    2: 'tensor'
    }

firedrake_cells = ("interval",
                    "triangle",
                    "tetrahedron",
                    "quadrilateral",
                    "hexahedron")


def cells(cell_list):
    return(", ".join(cell_list))


with open("element_list.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    for element in supported_elements:
        family, short_name, value_rank, sobolev_space, \
            mapping, degree_range, cell_list = ufl_elements[element]

        cell_list = [c for c in cell_list if c in firedrake_cells]
        short_name = short_name if short_name != family else ""
        cellnames = cells(cell_list)
        shape = shape_names[value_rank]

        if family in {"Q", "DQ", "DQ L2"}:
            cell = cell_list[-1]
        elif family in {"NCE", "NCF"}:
            cell = TensorProductCell(quadrilateral, interval)
        else:
            cell = cell_list[0]

        ufl_elem = FiniteElement(family, cell=cell, degree=degree_range[0])
        finat_element = create_element(ufl_elem)

        if short_name in {"BDMCF", "BDMCE"}:
            interpolatable = "No"
        else:
            try:
                finat_element.dual_basis
                interpolatable = "Yes"
            except NotImplementedError:
                interpolatable = "No"

        csvwriter.writerow((family, short_name, shape, cellnames, interpolatable))
