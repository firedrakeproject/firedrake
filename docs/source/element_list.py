from finat.ufl.elementlist import aliases, ufl_elements
from finat.element_factory import supported_elements
import csv
import sys

shape_names = {
    0: 'scalar',
    1: 'vector',
    2: 'tensor'
    }

# The DefElement page for each element family, as the page name in
# https://defelement.org/elements/<name>.html.  A value of None means that
# DefElement has no page for that family.  The build prints a warning for
# any family that is missing from this dictionary.
defelement_names = {
    "Alfeld C2": None,
    "Alfeld-Sorokina": "alfeld-sorokina",
    "Argyris": "argyris",
    "Arnold-Qin": None,
    "Bell": "bell",
    "Bernardi-Raugel": "bernardi-raugel",
    "Bernardi-Raugel Bubble": None,
    "Bernstein": "bernstein",
    "Bramble-Zlamal C2": None,
    "Brezzi-Douglas-Fortin-Marini": "brezzi-douglas-fortin-marini",
    "Brezzi-Douglas-Marini": "brezzi-douglas-marini",
    "Brezzi-Douglas-Marini Cube Edge": "scurl",
    "Brezzi-Douglas-Marini Cube Face": "sdiv",
    "Bubble": "bubble",
    "Christiansen-Hu": None,
    "Conforming Arnold-Winther": "arnold-winther",
    "Crouzeix-Raviart": "crouzeix-raviart",
    "Direct Serendipity": "direct-serendipity",
    "Discontinuous Lagrange": "discontinuous-lagrange",
    "Discontinuous Lagrange L2": "discontinuous-lagrange",
    "Discontinuous Raviart-Thomas": None,
    "Discontinuous Taylor": "taylor",
    "DPC": "dpc",
    "DPC L2": "dpc",
    "DQ": "discontinuous-lagrange",
    "DQ L2": "discontinuous-lagrange",
    "FacetBubble": None,
    "Gauss-Legendre": "gauss-legendre",
    "Gauss-Legendre L2": "gauss-legendre",
    "Gauss-Lobatto-Legendre": "lagrange",
    "Gopalakrishnan-Lederer-Schoberl 1st kind": "gopalakrishnan-lederer-schoberl",
    "Gopalakrishnan-Lederer-Schoberl 2nd kind": None,
    "Guzman-Neilan 1st kind H1": "guzman-neilan",
    "Guzman-Neilan 2nd kind H1": "guzman-neilan2",
    "Guzman-Neilan Bubble": None,
    "Guzman-Neilan H1(div)": None,
    "HDiv Trace": "lagrange",
    "Hellan-Herrmann-Johnson": "hellan-herrmann-johnson",
    "Hermite": "hermite",
    "Hsieh-Clough-Tocher": "hsieh-clough-tocher",
    "Hu-Zhang": None,
    "Johnson-Mercier": "johnson-mercier",
    "Kong-Mulder-Veldhuizen": "kong-mulder-veldhuizen",
    "Lagrange": "lagrange",
    "Mardal-Tai-Winther": "mardal-tai-winther",
    "Morley": "morley",
    "NCE": "nedelec1",
    "NCF": "raviart-thomas",
    "Nedelec 1st kind H(curl)": "nedelec1",
    "Nedelec 2nd kind H(curl)": "nedelec2",
    "Nonconforming Arnold-Winther": "nonconforming-arnold-winther",
    "Nonconforming Robust Wu-Xu": None,
    "Nonconforming Wu-Xu": "wu-xu",
    "Q": "lagrange",
    "QuadraticPowellSabin12": None,
    "QuadraticPowellSabin6": None,
    "Raviart-Thomas": "raviart-thomas",
    "Real": None,
    "Reduced-Arnold-Qin": None,
    "Reduced-Hsieh-Clough-Tocher": "reduced-hsieh-clough-tocher",
    "Regge": "regge",
    "RTCE": "nedelec1",
    "RTCF": "raviart-thomas",
    "S": "serendipity",
    "SminusCurl": "trimmed-serendipity-curl",
    "SminusDiv": "trimmed-serendipity-div",
    "SminusE": "trimmed-serendipity-curl",
    "SminusF": "trimmed-serendipity-div",
    "Walkington": None,
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

        other_names = [short_name] if short_name not in (None, family) else []
        other_names += [a for a in family_aliases.get(family, [])
                        if a not in other_names]
        other_names = ", ".join(other_names)
        cellnames = cells(cellnames)
        shape = shape_names[value_rank]

        rows.append((family, other_names, shape, cellnames))

    def name_cell(family):
        page = defelement_names.get(family)
        if page is None:
            return family
        return f"`{family} <https://defelement.org/elements/{page}.html>`__"

    # Sort alphabetically by name, ignoring case.
    for family, *rest in sorted(rows, key=lambda row: row[0].casefold()):
        csvwriter.writerow((name_cell(family), *rest))
