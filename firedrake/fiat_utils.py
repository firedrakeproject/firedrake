from __future__ import absolute_import
from weakref import WeakKeyDictionary
import FIAT
import ffc


_fiat_element_cache = WeakKeyDictionary()


_cells = {
    1: {2: "interval"},
    2: {3: "triangle", 4: "quadrilateral"},
    3: {4: "tetrahedron"}
}


def fiat_from_ufl_element(ufl_element):
    try:
        return _fiat_element_cache[ufl_element]
    except KeyError:
        fiat_element = ffc.create_actual_fiat_element(ufl_element)
        _fiat_element_cache[ufl_element] = fiat_element
        return fiat_element


class FlattenedElement(FIAT.FiniteElement):
    """Class for flattening tensor-like finite elements."""

    def __init__(self, tfe):
        """Initialises flattened element from tensor-like element.

        :arg tfe: tensor-like finite element to flatten
        """

        # save tensor-like element
        self._element = tfe

        if isinstance(tfe.get_reference_element(), FIAT.reference_element.two_product_cell):
            self.ref_el = tfe.get_reference_element().A
        else:
            raise TypeError("Cannot flatten non-product cell.")

        # set up entity_ids
        # Return the flattened (w.r.t. 2nd component) map
        # of topological entities to degrees of freedom.
        # Assumes product is something crossed with an interval
        dofs = tfe.entity_dofs()
        self.entity_ids = {}

        for dimA, dimB in dofs:
            # dimB = 0 or 1.  only look at the 1s, then grab the data from 0s
            if dimB == 0:
                continue
            self.entity_ids[dimA] = {}
            for ent in dofs[(dimA, dimB)]:
                # this line is fairly magic.
                # it works because an interval has two points.
                # we pick up the dofs from the bottom point,
                # then the dofs from the interior of the interval,
                # then finally the dofs from the top point
                self.entity_ids[dimA][ent] = \
                    dofs[(dimA, 0)][2*ent] + dofs[(dimA, 1)][ent] + dofs[(dimA, 0)][2*ent+1]

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self._element.degree()

    def entity_dofs(self):
        """Return the map of topological entities to degrees of
        freedom for the finite element."""
        return self.entity_ids

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        return self._element.space_dimension()
