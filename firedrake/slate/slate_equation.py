"""The SlateEquation class for SLATE objects. This is used to express
equations like A == B."""


__all__ = ['SlateEquation']


class SlateEquation(object):
    """This class is used to abstractly represent equations of SLATE objects
    via the "==" operator, such as A == B where A and B are SLATE expressions (tensors)."""

    def __init__(self, LHS, RHS):
        """Creates an equation LHS == RHS."""

        self.LHS = LHS
        self.RHS = RHS

    def __bool__(self):
        """Evalues the boolean expression: LHS == RHS."""

        if type(self.LHS) != type(self.RHS):
            return False
        if hasattr(self.LHS, "equals"):
            return self.LHS.equals(self.RHS)

        return repr(self.LHS) == repr(self.RHS)

    __nonzero__ = __bool__

    def __eq__(self, other):
        """Compare two equations by comparing LHS and RHS."""
        return (isinstance(other, SlateEquation) and
                bool(self.LHS == self.other.LHS) and
                bool(self.RHS == self.other.RHS))

    def __hash__(self):
        return hash((hash(self.LHS), hash(self.RHS)))

    def __repr__(self):
        return "SlateEquation(%r, %r)" % (self.LHS, self.RHS)
