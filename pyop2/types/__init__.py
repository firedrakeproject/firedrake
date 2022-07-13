import enum

from .access import *  # noqa: F401
from .data_carrier import *  # noqa: F401
from .dataset import *  # noqa: F401
from .dat import *  # noqa: F401
from .glob import *  # noqa: F401
from .halo import *  # noqa: F401
from .map import *  # noqa: F401
from .mat import *  # noqa: F401
from .set import *  # noqa: F401


class IterationRegion(enum.IntEnum):
    BOTTOM = 1
    TOP = 2
    INTERIOR_FACETS = 3
    ALL = 4


ON_BOTTOM = IterationRegion.BOTTOM
"""Iterate over the cells at the bottom of the column in an extruded mesh."""

ON_TOP = IterationRegion.TOP
"""Iterate over the top cells in an extruded mesh."""

ON_INTERIOR_FACETS = IterationRegion.INTERIOR_FACETS
"""Iterate over the interior facets of an extruded mesh."""

ALL = IterationRegion.ALL
"""Iterate over all cells of an extruded mesh."""
