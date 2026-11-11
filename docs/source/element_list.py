from finat.ufl.elementlist import aliases, ufl_elements
from finat.element_factory import supported_elements
import csv

shape_names = {
    0: 'scalar',
    1: 'vector',
    2: 'tensor'
    }

def simple_aliases():
    """Return the aliases that always select the same element family.

    Some aliases select a family that depends on the cell dimension or on
    the form degree, for example the FEEC notation ``"P-"``.  These aliases
    cannot be listed against a single family, so they are left out.

    Returns
    -------
    dict
        A map from each full family name to the sorted list of its simple
        aliases.
    """
    result = {}
    for alias, resolve in aliases.items():
        targets = set()
        for dim in (1, 2, 3):
            try:
                family, _ = resolve(alias, dim, 1, None)
            except TypeError:
                # The alias needs a form degree, so it is not simple.
                targets = None
                break
            targets.add(family)
        if targets is not None and len(targets) == 1:
            result.setdefault(targets.pop(), []).append(alias)
    return {family: sorted(names) for family, names in result.items()}


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

    family_aliases = simple_aliases()

    rows = []
    for element in supported_elements:
        family, short_name, value_rank, sobolev_space, \
            mapping, degree_range, cellnames = ufl_elements[element]

        # The other names are the abbreviation followed by the simple
        # aliases.  An empty cell tells the reader that only the full name
        # is available for this family.
        other_names = [short_name] if short_name not in (None, family) else []
        other_names += [a for a in family_aliases.get(family, [])
                        if a not in other_names]
        other_names = ", ".join(other_names)
        cellnames = cells(cellnames)
        shape = shape_names[value_rank]

        rows.append((family, other_names, shape, cellnames))

    # Sort alphabetically by name, ignoring case.
    for row in sorted(rows, key=lambda row: row[0].casefold()):
        csvwriter.writerow(row)
