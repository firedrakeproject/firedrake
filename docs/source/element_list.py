from finat.ufl.elementlist import aliases, ufl_elements
from finat.element_factory import supported_elements
import csv

shape_names = {0: "scalar", 1: "vector", 2: "tensor"}

firedrake_cells = ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron")

# The DefElement page for each element family, as the page name in
# https://defelement.org/elements/<name>.html.  A value of None means that
# DefElement has no page for that family.  The build fails if this dictionary
# does not have exactly one entry for each supported family.
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


families = {ufl_elements[element][0] for element in supported_elements}
if missing := families - defelement_names.keys():
    raise KeyError(f"defelement_names has no entry for {sorted(missing)}")
if extra := defelement_names.keys() - families:
    raise ValueError(f"defelement_names has entries for unsupported families {sorted(extra)}")

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
