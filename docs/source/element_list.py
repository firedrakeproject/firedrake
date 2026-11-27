from finat.ufl.elementlist import aliases, ufl_elements
from finat.element_factory import defelement_names, supported_elements
import csv

shape_names = {0: "scalar", 1: "vector", 2: "tensor"}

firedrake_cells = ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron")


def simple_aliases():
    """Return the short names and aliases that always select the same element family.

    Some aliases select a family that depends on the cell dimension or on
    the form degree, for example the FEEC notation ``"P-"``.  These aliases
    cannot be listed against a single family, so they are left out.

    Returns
    -------
    dict
        A map from each full family name to the sorted list of its short
        name and simple aliases.
    """
    result = {}
    for family, short_name, *_ in ufl_elements.values():
        if short_name is not None and short_name != family:
            result.setdefault(family, set()).add(short_name)
    for alias, resolve in aliases.items():
        targets = set()
        for dim in (1, 2, 3):
            try:
                family, _ = resolve(alias, dim, 1, None)
            except TypeError:
                # The alias needs a form degree, so it is not simple.
                break
            targets.add(family)
        else:
            if len(targets) == 1:
                result.setdefault(targets.pop(), set()).add(alias)
    return {family: sorted(names) for family, names in result.items()}


def name_cell(family):
    """Return the family name, linked to its DefElement page if there is one."""
    page = defelement_names[family]
    if page is None:
        return family
    return f"`{family} <https://defelement.org/elements/{page}.html>`__"


family_aliases = simple_aliases()
rows = []
# Sort alphabetically by family name, ignoring case.
for element in sorted(supported_elements, key=lambda e: ufl_elements[e][0].casefold()):
    family, _, value_rank, _, _, _, cellnames = ufl_elements[element]
    rows.append((
        name_cell(family),
        ", ".join(family_aliases.get(family, [])),
        shape_names[value_rank],
        ", ".join(c for c in cellnames if c in firedrake_cells),
    ))

with open("element_list.csv", "w", newline="") as csvfile:
    csv.writer(csvfile).writerows(rows)
