cell_integral_combined = """\
/// This integral defines the interface for the tabulation of the cell
/// tensor corresponding to the local contribution to a form from
/// the integral over a cell.

void %(classname)s(%(arglist)s)
{
%(tabulate_tensor)s
}"""

exterior_facet_integral_combined = """\
/// This integral defines the interface for the tabulation of the cell
/// tensor corresponding to the local contribution to a form from
/// the integral over an exterior facet.

void %(classname)s(%(arglist)s)
{
%(tabulate_tensor)s
}"""
