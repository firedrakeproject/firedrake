# -*- coding: utf-8 -*-
# Copyright (C) 2011-2015 Martin Sandve Alnæs
#
# This file is part of UFLACS.
#
# UFLACS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFLACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFLACS. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Miklós Homolya, 2016.

"""Definitions of 'modified terminals', a core concept in uflacs."""

from __future__ import print_function  # used in some debugging

from ufl.classes import (ReferenceValue, ReferenceGrad,
                         Restricted, FacetAvg, CellAvg)


class ModifiedTerminal(object):

    """A modified terminal expression is an object of a Terminal subtype, wrapped in terminal modifier types.

    The variables of this class are:

        expr - The original UFL expression

        terminal           - the underlying Terminal object
        local_derivatives  - tuple of ints, each meaning derivative in that local direction
        reference_value    - bool, whether this is represented in reference frame
        averaged           - None, 'facet' or 'cell'
        restriction        - None, '+' or '-'
    """

    def __init__(self, expr, terminal, local_derivatives, averaged, restriction, reference_value):
        # The original expression
        self.expr = expr

        # The underlying terminal expression
        self.terminal = terminal

        # Components
        self.reference_value = reference_value
        self.restriction = restriction

        # Derivatives
        self.local_derivatives = local_derivatives

        # Evaluation method (alternative: { None, 'facet_midpoint', 'cell_midpoint', 'facet_avg', 'cell_avg' })
        self.averaged = averaged

    def as_tuple(self):
        t = self.terminal
        rv = self.reference_value
        ld = self.local_derivatives
        a = self.averaged
        r = self.restriction
        return (t, rv, ld, a, r)

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        return isinstance(other, ModifiedTerminal) and self.as_tuple() == other.as_tuple()

    def __lt__(self, other):
        return self.as_tuple() < other.as_tuple()

    def __str__(self):
        s = []
        s += ["terminal:           {0}".format(self.terminal)]
        s += ["local_derivatives:  {0}".format(self.local_derivatives)]
        s += ["averaged:           {0}".format(self.averaged)]
        s += ["restriction:        {0}".format(self.restriction)]
        return '\n'.join(s)


def is_modified_terminal(v):
    "Check if v is a terminal or a terminal wrapped in terminal modifier types."
    while not v._ufl_is_terminal_:
        if v._ufl_is_terminal_modifier_:
            v = v.ufl_operands[0]
        else:
            return False
    return True


def strip_modified_terminal(v):
    "Extract core Terminal from a modified terminal or return None."
    while not v._ufl_is_terminal_:
        if v._ufl_is_terminal_modifier_:
            v = v.ufl_operands[0]
        else:
            return None
    return v


def analyse_modified_terminal(expr):
    """Analyse a so-called 'modified terminal' expression and return its properties in more compact form.

    A modified terminal expression is an object of a Terminal subtype, wrapped in terminal modifier types.

    The wrapper types can include 0-* Grad or ReferenceGrad objects,
    and 0-1 ReferenceValue, 0-1 Restricted, 0-1 Indexed, and 0-1 FacetAvg or CellAvg objects.
    """
    # Data to determine
    local_derivatives = 0
    reference_value = None
    restriction = None
    averaged = None

    # Start with expr and strip away layers of modifiers
    t = expr
    while not t._ufl_is_terminal_:
        if isinstance(t, ReferenceValue):
            assert reference_value is None, "Got twice pulled back terminal!"
            reference_value = True
            t, = t.ufl_operands

        elif isinstance(t, ReferenceGrad):
            local_derivatives += 1
            t, = t.ufl_operands

        elif isinstance(t, Restricted):
            assert restriction is None, "Got twice restricted terminal!"
            restriction = t._side
            t, = t.ufl_operands

        elif isinstance(t, CellAvg):
            assert averaged is None, "Got twice averaged terminal!"
            averaged = "cell"
            t, = t.ufl_operands

        elif isinstance(t, FacetAvg):
            assert averaged is None, "Got twice averaged terminal!"
            averaged = "facet"
            t, = t.ufl_operands

        elif t._ufl_terminal_modifiers_:
            raise ValueError("Missing handler for terminal modifier type %s, object is %s." % (type(t), repr(t)))

        else:
            raise ValueError("Unexpected type %s object %s." % (type(t), repr(t)))

    # Make reference_value true or false
    if reference_value is None:
        reference_value = False

    mt = ModifiedTerminal(expr, t, local_derivatives, averaged, restriction, reference_value)

    if local_derivatives and not reference_value:
        raise ValueError("Local derivatives of non-local value?")

    return mt
